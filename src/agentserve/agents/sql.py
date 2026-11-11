"""Text-to-SQL policy independent of HTTP routing, vLLM internals and storage."""

import json
import re
from decimal import Decimal

from agentserve.agents.answer_policy import (
    CONCISE_STYLES,
    DIRECT_EVIDENCE_STYLE,
    order_amount_columns,
    order_amount_note,
    render_answer,
)
from agentserve.agents.order_sql import compile_order_sql
from agentserve.agents.retail_sql import RETAIL_DATABASE, use_retail_date_index
from agentserve.agents.sql_policy import intent_violations
from agentserve.agents.sql_prompt import (
    CONCISE_ANSWER_PROMPT,
    FORMAT_REPAIR_PROMPT,
    initial_messages,
)
from agentserve.config import Settings
from agentserve.domain.models import AgentError, QueryResult, RunReport, TaskInput
from agentserve.llm.client import LLMBackend
from agentserve.llm.execution import ModelExecutor
from agentserve.tools.sqlite import FUNCTION_PROFILES, SQLTool


class SQLAgent:
    def __init__(self, backend: LLMBackend, tool: SQLTool, settings: Settings):
        self.backend, self.tool, self.settings = backend, tool, settings
        self.model = ModelExecutor(backend, settings)

    async def run(
        self,
        task: TaskInput,
        report: RunReport,
        *,
        deadline: float,
        emit,
        journal=None,
        memory_text=None,
        workbench_context: tuple[str, dict[str, str]] | None = None,
    ) -> None:
        database = self.settings.databases[task.database_id]
        schema = database.schema_path.read_text(encoding="utf-8")
        plan = (report.memory_context or {}).get("plan") or {}
        state = plan.get("state")
        context, city_aliases = workbench_context or ("", {})
        canonical = None
        if state and city_aliases and (city := state["fields"].get("city")):
            value = city["value"]
            if isinstance(value, str):
                canonical = city_aliases.get(value.removesuffix("市").casefold())
                if canonical is not None:
                    state = {**state, "fields": {**state["fields"], "city": {**city, "value": canonical}}}
        v22 = self.settings.sql_prompt_version == "evidence_v22"
        order_database = task.database_id in {"business_orders_v1", "demo_orders"}
        style = self.settings.answer_style
        compiled = (
            compile_order_sql(state)
            if v22 and order_database and state and state.get("complete_order_intent")
            else None
        )
        model_wording = workbench_context is not None and compiled is None
        explain = task.answer_mode == "explain" or model_wording
        business_question = task.question
        if model_wording and state and plan.get("action") == "refine":
            business_question += " " + state["original_question"]
        settled = bool(model_wording and re.search("成交|交易|营收|营业额", business_question)
                       and not re.search("待处理|待付款|未完成|取消|所有状态|全部状态|pending|cancelled",
                                         business_question, re.IGNORECASE))
        expected_amounts: set[Decimal] = set()
        if compiled is None and plan.get("resolution") == "window_order":
            state = None
        question = task.question
        if state and plan.get("route") != "explicit_independent":
            from agentserve.memory.state_rules import execution_question

            execution_state = state
            if model_wording:
                execution_state = {**state, "fields": {
                    name: value for name, value in state["fields"].items() if name != "metric"
                }}
            question = execution_question(
                execution_state,
                include_source=not compiled if v22 else plan.get("resolution") != "deterministic",
                current_question=task.question if v22 and not compiled else None,
            )
        messages = initial_messages(
            question,
            schema=schema,
            context=database.context + context,
            version=self.settings.sql_prompt_version,
            functions=sorted(FUNCTION_PROFILES[self.settings.sql_function_profile]),
        )
        if memory_text is not None and state is None:
            messages.insert(
                1,
                {
                    "role": "user",
                    "content": "Visible session evidence (historical outputs are unverified; explicit current "
                    "conditions override history). Inherit only the relevant analysis conditions. "
                    "Execute a fresh SQL query; do not answer from historical totals. If the reference "
                    "cannot be resolved from visible evidence, do not call tools: respond only "
                    "CLARIFY: followed by a concrete Chinese clarification question.\n"
                    "Clarify only a genuinely missing business condition or unresolved reference. "
                    "Never ask to confirm explicit conditions, requested grouping, or schema details "
                    "already supplied. Correct SQL errors yourself using Schema and tool errors. "
                    "The Schema above is authoritative; historical or planned column names may be wrong.\n"
                    + memory_text,
                },
            )

        async def execute(sql):
            if journal is not None:
                return await journal.execute_sql(
                    self.tool, task.database_id, sql, deadline=deadline
                )
            await emit("SQL_STARTED", {"sql": sql})
            result = await self.tool.execute(task.database_id, sql, deadline=deadline)
            await emit("SQL_COMPLETED", {"result": result.to_dict()})
            return result

        async def finish_answer(model_answer):
            result = report.query_result
            unit_note = ""
            if "integer cents" in database.context:
                if style in {"evidence_v221", DIRECT_EVIDENCE_STYLE} and order_database:
                    unit_note = order_amount_note(report.sql, result)
                elif style == "evidence_v21" and any(
                    term in " ".join(result.columns).lower()
                    for term in ["cents", "revenue", "营收", "金额", "price", "标价"]
                ):
                    unit_note = "数据库金额的原始单位为分；本回答原样展示查询返回值。"
            answer_style = "bounded_v1" if explain else style
            if expected_amounts:
                mentioned = {
                    Decimal(value.replace(",", "")) for value in re.findall(
                        r"(-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*元", model_answer,
                    )
                }
                if mentioned != expected_amounts:
                    answer_style = DIRECT_EVIDENCE_STYLE
            report.answer, report.answer_source = render_answer(model_answer, result, answer_style, unit_note)
            if report.answer_source == "bounded_summary":
                await emit(
                    "ANSWER_POLICY_APPLIED",
                    {"policy": "bounded_v1", "returned_rows": len(result.rows)},
                )

        query = compiled is None
        if compiled is not None:
            report.sql = compiled
            await emit("SQL_COMPILED", {"compiler": "order_v22", "sql": compiled})
            result = report.query_result = await execute(compiled)
            if result.status != "ok":
                raise AgentError(result.status, result.error)
            call = {
                "id": "order_query",
                "type": "function",
                "function": {"name": "execute_sql", "arguments": json.dumps({"sql": compiled})},
            }
            messages.extend(
                [
                    {"role": "assistant", "tool_calls": [call]},
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result.to_dict(), ensure_ascii=False),
                    },
                    {"role": "user", "content": CONCISE_ANSWER_PROMPT},
                ]
            )
        while True:
            if not query and (
                (explain and report.query_result.truncated)
                or (style == DIRECT_EVIDENCE_STYLE and not explain)
            ):
                await finish_answer("")
                return
            try:
                reply = await self.model.generate(
                    messages,
                    query=query,
                    deadline=deadline,
                    max_tokens=self.settings.sql_max_tokens
                    if query
                    else self.settings.answer_max_tokens,
                    report=report,
                    emit=emit,
                    journal=journal,
                    draft_answer=explain,
                )
            except AgentError as error:
                if (
                    query
                    and error.code == "MODEL_PROTOCOL_ERROR"
                    and report.format_repairs < self.settings.max_format_repairs
                ):
                    report.format_repairs += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": FORMAT_REPAIR_PROMPT,
                        }
                    )
                    continue
                raise
            if not query:
                await finish_answer(reply.message["content"])
                return
            statement = reply.sql
            tool_message = reply.message
            if task.database_id == RETAIL_DATABASE:
                statement = use_retail_date_index(reply.sql)
                if statement != reply.sql:
                    call = tool_message["tool_calls"][0]
                    tool_message = {**tool_message, "tool_calls": [{**call, "function": {
                        **call["function"], "arguments": json.dumps({"sql": statement}),
                    }}]}
            report.sql = statement
            violations = []
            if (v22 or self.settings.sql_prompt_version == "evidence_v21") and order_database:
                grounded = state if plan.get("can_execute") else None
                violations = intent_violations(reply.sql, task.question, grounded, city=canonical, settled=settled)
            if violations:
                report.query_result = QueryResult(
                    status="SQL_INTENT_MISMATCH", error="Not executed: " + " ".join(violations)
                )
                await emit(
                    "SQL_INTENT_CHECK_FAILED",
                    {"executed": False, "violations": violations, "sql": reply.sql},
                )
            else:
                report.query_result = await execute(statement)
            result = report.query_result
            messages.extend(
                [
                    tool_message,
                    {
                        "role": "tool",
                        "tool_call_id": reply.message["tool_calls"][0]["id"],
                        "content": json.dumps(
                            result.to_dict(), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                ]
            )
            if result.status == "ok":
                if explain or style in CONCISE_STYLES:
                    instruction = CONCISE_ANSWER_PROMPT
                    if model_wording:
                        instruction += (" 直接回答当前问题，说明实际查询的城市和月份。"
                                        "成交有无先回答有或没有。"
                                        "不显示 SQL、字段名或计算过程。")
                        if len(result.rows) == 1 and not result.truncated:
                            columns = set(order_amount_columns(report.sql, result))
                            if settled:
                                columns.update(i for i, name in enumerate(result.columns)
                                               if name.lower() in {"revenue", "total_revenue", "revenue_cents"})
                            display = dict(zip(result.columns, result.rows[0], strict=True))
                            for index in sorted(columns):
                                name = result.columns[index]
                                cents = display[name]
                                if type(cents) is int:
                                    yuan = Decimal(cents) / 100
                                    expected_amounts.add(yuan)
                                    display[name] = f"{yuan:,.2f} 元"
                            if expected_amounts:
                                messages = [
                                    {"role": "system", "content": instruction +
                                     " 数据中的金额已经换算成元。回答时逐字复制金额和单位，不要再换算。"},
                                    {"role": "user", "content": f"{question}\n已核实结果："
                                     + json.dumps(display, ensure_ascii=False)},
                                ]
                    if not expected_amounts:
                        messages.append({"role": "user", "content": instruction})
                query = False
            elif result.status in {"SQL_EXECUTION_ERROR", "SQL_INTENT_MISMATCH"} or (
                (v22 or self.settings.sql_prompt_version == "evidence_v21")
                and result.status == "POLICY_REJECTED"
                and (result.error or "").startswith("not authorized to use function: ")
            ):
                if report.sql_repairs >= self.settings.max_sql_repairs:
                    raise AgentError("SQL_REPAIR_EXHAUSTED", result.error)
                report.sql_repairs += 1
                await emit("SQL_REPAIR_REQUESTED", {"repair": report.sql_repairs})
            else:
                raise AgentError(result.status, result.error)
