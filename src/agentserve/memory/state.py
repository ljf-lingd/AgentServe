"""Typed query state: one intent extraction, deterministic version lookup and updates."""

import json
import re
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from agentserve.domain.models import AgentError
from agentserve.memory.models import ContextPack, EncodedMemory, EntryDraft


class StateChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1, max_length=80)
    operation: Literal["set", "delete"]
    operator: Literal["eq", "ne", "in", "not_in", "range", "gte", "lte", "gt", "lt", "describe"] = (
        "eq"
    )
    value: str | int | float | bool | list[str] | None = None
    evidence: str = ""


class StateIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["new", "refine", "resume", "clarify"]
    name: str | None = None
    version_id: str | None = None
    changes: list[StateChange] = Field(default_factory=list, max_length=20)
    clarification: str | None = None


STATE_PROMPT = """Extract the CURRENT user's query intent, not SQL. Return JSON matching the schema.
State is intent, never a database result. Preserve analysis names exactly. new creates an independent
analysis; refine modifies an existing one; resume selects an older analysis/version. Select name from
the supplied catalogue, never silently pick a similar name. version_id is null for latest, or exactly
one supplied immutable version when the user explicitly asks for that version. A first/original
version means the first supplied version of that name. Bare continue refers to active_name.
changes includes ONLY CURRENT explicit new/replaced/deleted fields, not inherited values.
KEEP language such as 保留输出和排序 is not a change. Leave those fields untouched.
Use stable conceptual fields: city, date, status, metric, group_by, outputs, unit, plus other explicit
business filters. date is range [inclusive YYYY-MM-DD, exclusive YYYY-MM-DD]. group_by and outputs
are lists of business labels. metric describes the requested calculation. Use operator describe for
metric and outputs. Preserve filters on refine/resume; remove only when requested using delete.
All statuses explicitly requested means delete status. Do not create permanent session preferences.
Do NOT infer any SQL identifiers or any monetary result. Source SQL/answers are not inputs.
Resolve relative dates using accepted_at and Asia/Shanghai, never the process clock.
Clarify ONLY unresolved references or truly missing constraints, never reconfirm complete conditions.
If the question is independent, use new even if a prior analysis has related vocabulary.
Every change MUST include evidence: an EXACT substring of the CURRENT question supporting that
change. A set requires a NON-NULL value. Omit unspecified fields entirely; NEVER insert null set,
empty unit, active status, guessed preference, or a guessed filter. Delete uses null value.
Keep metric and output labels in the user's natural language, not invented programming identifiers.
Example: question 继续“销售”，城市改为Shanghai，其他不变。
{"action":"refine","name":"销售","version_id":null,"changes":[{"field":"city",
"operation":"set","operator":"eq","value":"Shanghai","evidence":"城市改为Shanghai"}],"clarification":null}
Example: question 新建“产品”：Office类别商品有多少种？
{"action":"new","name":"产品","version_id":null,"changes":[{"field":"category",
"operation":"set","operator":"eq","value":"Office","evidence":"Office类别"},
{"field":"metric","operation":"set","operator":"describe","value":"商品种数","evidence":"商品有多少种"}],"clarification":null}
Example: question 继续“销售”，取消城市筛选。
{"action":"refine","name":"销售","version_id":null,"changes":[{"field":"city",
"operation":"delete","operator":"eq","value":null,"evidence":"取消城市筛选"}],"clarification":null}
metric is ONLY the measured quantity (订单笔数, 商品种数, 营收合计), NEVER a status, price
threshold or other filter. Price thresholds use field price. Date, city, status and category are
separate filters, not part of the metric label. Unspecified fields are omitted.
Example: 当前标价高于1000分的Office商品种数: changes has category eq Office, price gt 1000,
and metric describe 商品种数, each with a supporting verbatim evidence substring.
On refinement use the field names in relevant_states so deletions remove the actual stored filter.
For a new analysis include its explicit intent fields. Return JSON only, no explanation."""


def normalize_date(change, accepted_at):
    """Normalize explicitly supported calendar expressions using the durable acceptance anchor."""
    if change.field != "date" or change.operation != "set":
        return
    match = re.search(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?", change.evidence)
    anchor = datetime.fromtimestamp(accepted_at, ZoneInfo("Asia/Shanghai"))
    if match:
        year, month = int(match[1]), int(match[2])
        if match[3]:
            start = datetime(year, month, int(match[3]))
            end = start + timedelta(days=1)
        else:
            start = datetime(year, month, 1)
            end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1)
    elif change.evidence in {"上个月", "本月", "这个月"}:
        start = anchor.replace(day=1)
        if change.evidence == "上个月":
            start = (start - timedelta(days=1)).replace(day=1)
        end = start.replace(
            year=start.year + (start.month == 12), month=1 if start.month == 12 else start.month + 1
        )
    else:
        return
    change.operator = "range"
    change.value = [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]


def resolve_state(intent, entries, active_name, source_id):
    """Apply changes to an exact immutable version; no semantic retrieval or latest chasing."""
    if intent.action == "clarify":
        raise AgentError(
            "MEMORY_CONTEXT_REQUIRED", intent.clarification or "请明确要恢复哪组分析。"
        )
    name = intent.name
    target = None
    if intent.action != "new":
        name = name or active_name
        matching = [e for e in entries if e["state"]["name"] == name]
        if intent.version_id:
            matching = [e for e in matching if e["state"]["version_id"] == intent.version_id]
        if not matching:
            raise AgentError("MEMORY_CONTEXT_REQUIRED", "本会话没有该分析或指定版本，请补充条件。")
        target = matching[-1]
    fields = dict(target["state"]["fields"]) if target else {}
    for change in intent.changes:
        if change.operation == "delete":
            fields.pop(change.field, None)
        else:
            if change.value is None:
                raise AgentError(
                    "MEMORY_PROTOCOL_ERROR", "A state replacement requires a typed value"
                )
            prior = fields.get(change.field)
            if prior and prior["operator"] == change.operator and prior["value"] == change.value:
                continue
            fields[change.field] = {
                "operator": change.operator,
                "value": change.value,
                "source_task_id": source_id,
            }
    return name, fields, target


def independent_request(question):
    return bool(
        re.search(r"独立|\bindependent\b|\bunrelated\b", question, re.IGNORECASE)
        and not re.search(
            r'[“「"]|\bnamed\b|继续|沿用|恢复|刚才|上次|原来|其余不变|其他不变|'
            r"不是.*独立|不要.*独立|\bcontinue\b|\bresume\b|\bprevious\b|\bnot\b",
            question,
            re.IGNORECASE,
        )
    )


def build_state(intent, entries, active_name, question, turn_no, accepted_at, complete):
    name, fields, target = resolve_state(intent, entries, active_name, None)
    prior = target["state"] if target else None
    removed = set(prior.get("removed_fields", [])) if prior else set()
    for change in intent.changes:
        if change.operation == "delete":
            removed.add(change.field)
        else:
            removed.discard(change.field)
    state = {
        "name": name or f"analysis-turn-{turn_no}",
        "version_id": f"turn:{turn_no}",
        "fields": fields,
        "changed_fields": [c.model_dump() for c in intent.changes],
        "based_on": target["entry_id"] if target else None,
        "accepted_at": accepted_at,
        "timezone": "Asia/Shanghai",
        "removed_fields": sorted(removed),
        "original_question": prior["original_question"] if prior else question,
        "modification_questions": (prior["modification_questions"] if prior else [])
        + ([question] if prior and (intent.changes or not complete) else []),
        "complete_order_intent": bool(
            complete and (prior is None or prior.get("complete_order_intent"))
        ),
    }
    return state, target


def window_order_plan(question, inputs):
    """Reconstruct only complete chains grounded in the bounded visible window."""
    from agentserve.memory.state_rules import complete_order_question, order_intent

    entries = []
    current = {
        "question": question,
        "turn_no": inputs["cutoff_turn"] + 1,
        "created_at": inputs["accepted_at"],
        "task_id": None,
        "status": "COMPLETED",
    }
    for turn in [*inputs["recent"], current]:
        text = turn["question"]
        if turn["status"] != "COMPLETED" or not complete_order_question(text):
            continue
        try:
            intent = order_intent(text, entries, turn["created_at"])
            if intent is None:
                continue
            state, target = build_state(
                intent, entries, None, text, turn["turn_no"], turn["created_at"], True
            )
        except AgentError as error:
            if error.code != "MEMORY_CONTEXT_REQUIRED":
                raise
            continue
        except ValueError:
            continue
        if turn is current:
            return {
                "state": state,
                "can_execute": True,
                "resolution": "window_order",
                "target_id": target["entry_id"] if target else None,
            }
        entries.append({"state": state, "entry_id": turn["task_id"]})
    return None


class StatePolicy:
    version = "query-state-v221-1"
    plan_type = StateIntent
    event_binding = True
    condition_updates = True

    def __init__(self, on_demand=False):
        self.on_demand = on_demand
        self.version += "-on-demand" if on_demand else "-always"

    async def prepare(self, task, inputs, generate, resources, settings, deadline):
        if self.on_demand and task.database_id not in {"business_orders_v1", "demo_orders"}:
            from agentserve.memory.policy import pack_context

            pack = await pack_context(
                inputs, [], None, self.version + "-recent-window", resources, settings, deadline
            )
            pack.plan = {"route": "recent_window", "can_execute": False}
            return pack
        if self.on_demand and independent_request(task.question):
            # Explicitly independent, unnamed queries need neither historical reasoning nor
            # generative rewriting. Preserve the exact current request for a possible later follow-up.
            state = {
                "name": f"analysis-turn-{inputs['cutoff_turn'] + 1}",
                "version_id": f"turn:{inputs['cutoff_turn'] + 1}",
                "fields": {
                    "request": {
                        "operator": "describe",
                        "value": task.question,
                        "source_task_id": None,
                    }
                },
                "original_question": task.question,
                "modification_questions": [],
                "changed_fields": [],
                "based_on": None,
                "accepted_at": inputs["accepted_at"],
                "timezone": "Asia/Shanghai",
            }
            if task.database_id in {"business_orders_v1", "demo_orders"}:
                from agentserve.memory.state_rules import complete_order_question, order_intent

                try:
                    parsed = order_intent(task.question, [], inputs["accepted_at"])
                except ValueError as error:
                    raise AgentError(
                        "MEMORY_CONTEXT_REQUIRED", "日期无效，请明确有效的日历日期。"
                    ) from error
                if parsed is not None:
                    state["fields"] = resolve_state(parsed, [], None, None)[1]
                    state["changed_fields"] = [c.model_dump() for c in parsed.changes]
                    state["removed_fields"] = sorted(
                        {c.field for c in parsed.changes if c.operation == "delete"}
                    )
                    state["complete_order_intent"] = complete_order_question(task.question)
            text = json.dumps(
                {"resolved_query_state": state, "current_question": task.question},
                ensure_ascii=False,
            )
            count = await resources.count(text, deadline)
            if count > settings.context_tokens:
                raise AgentError(
                    "MEMORY_CONTEXT_LIMIT", "Independent request exceeds memory budget"
                )
            plan = {
                "action": "new",
                "target_id": None,
                "state": state,
                "route": "explicit_independent",
                "can_execute": False,
            }
            return ContextPack(
                text, [], inputs["cutoff_turn"], inputs["revision"], self.version, count, plan
            )
        entries = [e for e in inputs["entries"] if e.get("state")]
        active_name = entries[-1]["state"]["name"] if entries else None
        catalogue = [
            {
                "name": e["state"]["name"],
                "version_id": e["state"]["version_id"],
                "turn_no": e["turn_no"],
                "changed_fields": [c["field"] for c in e["state"].get("changed_fields", [])],
            }
            for e in entries
        ]
        mentioned = {e["state"]["name"] for e in entries if e["state"]["name"] in task.question}
        relevant = [
            e["state"] for e in entries if e["state"]["name"] in (mentioned or {active_name})
        ][-4:]
        # Catalogue contains identity only, not gold, future history or SQL observations.
        messages = [
            {"role": "system", "content": STATE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": task.question,
                        "active_name": active_name,
                        "catalogue": catalogue,
                        "relevant_states": relevant,
                        "accepted_at": datetime.fromtimestamp(
                            inputs["accepted_at"], ZoneInfo("Asia/Shanghai")
                        ).isoformat(),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        deterministic = None
        if task.database_id in {"business_orders_v1", "demo_orders"}:
            from agentserve.memory.state_rules import complete_order_question, order_intent

            try:
                deterministic = order_intent(task.question, entries, inputs["accepted_at"])
            except ValueError as error:
                raise AgentError(
                    "MEMORY_CONTEXT_REQUIRED", "日期无效，请明确有效的日历日期。"
                ) from error
        if deterministic is not None:
            intent = deterministic
            messages = []
        else:
            intent = await generate(messages, StateIntent, "memory_plan")
        for change in intent.changes:
            if not change.evidence or change.evidence not in task.question:
                raise AgentError(
                    "MEMORY_PROTOCOL_ERROR", "State change lacks verbatim current-user evidence"
                )
            try:
                normalize_date(change, inputs["accepted_at"])
            except ValueError as error:
                raise AgentError(
                    "MEMORY_CONTEXT_REQUIRED", "日期无效，请明确有效的日历日期。"
                ) from error
        if (
            intent.action == "new"
            and intent.name
            and not any(
                token in task.question
                for token in [
                    f"“{intent.name}”",
                    f"「{intent.name}」",
                    f'"{intent.name}"',
                    f"命名为{intent.name}",
                ]
            )
        ):
            intent.name = None
        state, target = build_state(
            intent,
            entries,
            active_name,
            task.question,
            inputs["cutoff_turn"] + 1,
            inputs["accepted_at"],
            deterministic is not None and complete_order_question(task.question),
        )
        text = json.dumps(
            {
                "resolved_query_state": state,
                "current_question": task.question,
                "state_rule": "Execute these complete conditions. Current explicit modifications win. "
                "The original_question and ordered modification_questions are verbatim user evidence. "
                "Read them together for the complete request. Fields are extracted intent, not SQL. "
                "No reconfirmation is required. Missing state fields are not implicit filters.",
            },
            ensure_ascii=False,
        )
        count = await resources.count(text, deadline)
        if count > settings.context_tokens:
            raise AgentError("MEMORY_CONTEXT_LIMIT", "Required query state exceeds memory budget")
        plan = {
            **intent.model_dump(),
            "target_id": target["entry_id"] if target else None,
            "state": state,
            "can_execute": deterministic is not None,
            "resolution": "deterministic" if deterministic is not None else "model",
        }
        return ContextPack(
            text,
            [target["entry_id"]] if target else [],
            inputs["cutoff_turn"],
            inputs["revision"],
            self.version,
            count,
            plan,
            [target] if target else [],
            messages,
        )

    async def encode(self, observation, context, generate):
        state = context.plan["state"]
        return EncodedMemory(
            entries=[
                EntryDraft(
                    text="Query state " + state["version_id"], topic=state["name"], state=state
                )
            ]
        )
