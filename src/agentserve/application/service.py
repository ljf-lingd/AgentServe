"""Shared task execution, with an optional durable event sink owned by storage."""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from agentserve.agents.sql import SQLAgent
from agentserve.application.execution import execution_snapshot
from agentserve.application.order_workbench import DATABASE as ORDER_DATABASE
from agentserve.application.order_workbench import calendar_bounds, calendar_mentions
from agentserve.domain.models import AgentError, RunReport, TaskInput
from agentserve.llm.client import phase_draft_scope
from agentserve.runtime.journal import StepJournal
from agentserve.storage.steps import StepRepository

COVERAGE_NOTICE = "所问日期超出数据覆盖范围。"


class TaskService:
    def __init__(
        self, agent: SQLAgent, memory=None, knowledge=None, metric=None, analysis=None, *, workbench=None,
    ):
        self.agent = agent
        self.memory = memory
        self.knowledge = knowledge
        self.metric = metric
        self.analysis = analysis
        self.workbench = workbench
        self.execution_config = execution_snapshot(agent.settings)
        identity = self.execution_config["execution_identity"]
        root = Path(__file__).resolve().parents[1]
        if workbench:
            identity["workbench_implementation_sha256"] = sha256(
                (root / "application/order_workbench.py").read_bytes()
            ).hexdigest()
        if knowledge:
            identity["knowledge_snapshot_sha256"] = knowledge.snapshot.identity
            identity["knowledge_implementation_sha256"] = {
                p.name: sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "knowledge").glob("*.py")) if p.name != "metrics.py"
            }
        if metric:
            identity["metric_registry_sha256"] = metric.catalog.identity
            identity["metric_implementation_sha256"] = {
                path: sha256((root / path).read_bytes()).hexdigest()
                for path in (
                    "knowledge/metrics.py", "agents/order_sql.py", "tools/sqlite.py",
                    "runtime/journal.py",
                )
            }
        if analysis:
            identity["analysis_implementation_sha256"] = {
                path: sha256((root / path).read_bytes()).hexdigest()
                for path in ("analytics/service.py", "analytics/artifacts.py", "storage/metric.py")
            }

    async def run(
        self,
        task: TaskInput,
        *,
        task_id: str | None = None,
        deadline_at: float | None = None,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
        step_store: StepRepository | None = None,
        version: int | None = None,
        step_hook=None,
    ) -> RunReport:
        if task.database_id not in self.agent.settings.databases:
            raise AgentError("UNKNOWN_DATABASE", "database_id is not configured")
        report = RunReport(task_id=task_id or uuid4().hex, task=task.model_dump(exclude_none=True))
        if step_store is not None:
            counters = await step_store.counters(report.task_id)
            memory_counters = await step_store.counters(report.task_id, memory=True)
            report.memory_calls = memory_counters["model_calls"]
            report.model_calls = counters["model_calls"] + report.memory_calls
            report.model_retries = counters["model_retries"] + memory_counters["model_retries"]
        started = time.monotonic()
        remaining = (
            self.agent.settings.task_timeout_seconds
            if deadline_at is None
            else deadline_at - time.time()
        )
        deadline = started + remaining

        async def emit(kind: str, payload: dict, *, persist=True) -> None:
            event = {
                "event_id": len(report.events) + 1,
                "type": kind,
                "task_id": report.task_id,
                "wall_time": datetime.now(UTC).isoformat(),
                "offset_seconds": time.monotonic() - started,
                "payload": payload,
            }
            report.events.append(event)
            # Terminal event and task result are committed together by the repository.
            if persist and on_event is not None and kind not in {"TASK_COMPLETED", "TASK_FAILED"}:
                await on_event(event)

        await emit("TASK_STARTED", {"config": self.execution_config})
        stage = "sql"
        draft_token = (
            phase_draft_scope.set(partial(step_store.phase_draft_history, report.task_id))
            if self.agent.settings.session_sam_all_phases and step_store and task.session_id else None
        )
        try:
            async with asyncio.timeout_at(deadline):
                if remaining <= 0:
                    raise TimeoutError
                journal = (
                    StepJournal(step_store, report.task_id, version, emit, hook=step_hook)
                    if step_store is not None
                    else None
                )
                pack = None
                mentions = []
                if self.workbench:
                    dataset = self.agent.settings.databases[task.database_id].workbench
                    if dataset and task.mode in {None, "sql"} and task.question:
                        mentions = calendar_mentions(task.question)
                        if mentions and mentions[0]["year"]:
                            try:
                                start, end = calendar_bounds(mentions, dataset.first_date.year)
                            except ValueError:
                                start, end = "", ""
                            if start and (end <= dataset.first_date.isoformat() or start > dataset.last_date.isoformat()):
                                report.answer_source = "workbench"
                                report.answer = (
                                    f"{COVERAGE_NOTICE}当前数据覆盖 "
                                    f"{dataset.first_date} 至 {dataset.last_date}。"
                                )
                    if report.answer_source != "workbench" and task.database_id == ORDER_DATABASE:
                        task, pack = await self.workbench.prepare(task, report, journal, deadline)
                if task.session_id and report.answer_source != "workbench":
                    stage = "memory_prepare"
                    if self.memory is None:
                        raise AgentError("MEMORY_DISABLED", "Session memory is not configured")
                    if pack is None:
                        pack = await self.memory.prepare(task, report, emit, journal, deadline)
                    if self.workbench and task.database_id == ORDER_DATABASE:
                        pack = self.workbench.ground_period(task, pack)
                        if (pack.plan or {}).get("render_calendar_count") and not task.answer_mode:
                            task = task.model_copy(update={"answer_mode": "explain"})
                    report.memory_context = {
                        k: v
                        for k, v in pack.__dict__.items()
                        if k not in {"planning_messages", "selected"}
                    }
                if report.answer_source == "workbench":
                    pass
                elif task.mode == "knowledge":
                    stage = "knowledge"
                    if self.knowledge is None:
                        raise AgentError("KNOWLEDGE_DISABLED", "Knowledge is not configured")
                    await self.knowledge.run(task, report, deadline=deadline, emit=emit, journal=journal)
                elif task.mode == "metric":
                    stage = "metric"
                    if self.metric is None:
                        raise AgentError("METRIC_DISABLED", "Metrics are not configured")
                    await self.metric.run(task, report, deadline=deadline, emit=emit, journal=journal)
                elif task.mode == "analysis":
                    stage = "analysis"
                    if self.analysis is None:
                        raise AgentError("ANALYSIS_DISABLED", "Analysis is not configured")
                    await self.analysis.run(task, report, deadline=deadline, emit=emit, journal=journal)
                else:
                    stage = "sql"
                    workbench_context = None
                    if (self.workbench and task.database_id == ORDER_DATABASE and pack
                        and not ((pack.plan or {}).get("state") or {}).get("workbench_answer")):
                        workbench_context = await self.workbench.model_context(deadline, journal)
                    memory_text = pack.text if pack else None
                    if pack and (pack.plan or {}).get("route") == "recent_window":
                        recent_turns = json.loads(memory_text)["recent_turns"]
                        recent = recent_turns[-1] if recent_turns else None
                        after_coverage = (
                            recent and recent["status"] == "COMPLETED" and not recent["sql"]
                            and (recent["answer"] or "").startswith(COVERAGE_NOTICE)
                        )
                        independent = (
                            after_coverage and mentions and mentions[0]["year"]
                            and re.search(r"多少|哪些|几个|几笔", task.question)
                            and not re.search(r"^\s*(?:那|再|改|换|只看|继续)|沿用|刚才|上次|不变", task.question)
                        )
                        if independent:
                            # A coverage notice supplies no query state. The current request
                            # specifies a new dated analysis, so use the normal named SQL tool.
                            report.memory_context["plan"]["can_execute"] = True
                            memory_text = None
                        else:
                            last_query = next(
                                (turn["sql"] for turn in reversed(recent_turns)
                                 if turn["status"] == "COMPLETED" and turn["sql"]),
                                None,
                            )
                            if last_query:
                                memory_text += (
                                    "\nLatest successful query: " + last_query
                                    + "\nFor a follow-up that changes one condition, inherit all other "
                                    "conditions from this query, even if an older question differs."
                                )
                    await self.agent.run(
                        task, report, deadline=deadline, emit=emit, journal=journal,
                        memory_text=memory_text,
                        workbench_context=workbench_context,
                    )
                if self.workbench and task.database_id == ORDER_DATABASE and report.answer_source != "workbench":
                    self.workbench.finish(task, report, pack)
                if pack:
                    stage = "memory_write"
                    report.memory_draft = await self.memory.encode(
                        task, report, pack, emit, journal, deadline
                    )
            report.status = "COMPLETED"
        except TimeoutError:
            report.status = "FAILED"
            report.error = {"code": "TASK_TIMEOUT", "detail": "Total task deadline exceeded"}
            report.error["stage"] = stage
        except AgentError as error:
            report.status = "FAILED"
            report.error = {"code": error.code, "detail": str(error)}
            if stage != "sql":
                report.error["stage"] = stage
            elif report.query_result is not None and report.query_result.status == "ok":
                report.error["stage"] = "summary"
        finally:
            if draft_token is not None:
                phase_draft_scope.reset(draft_token)
        report.elapsed_seconds = time.monotonic() - started
        await emit(
            "TASK_COMPLETED" if report.status == "COMPLETED" else "TASK_FAILED",
            {"error": report.error},
        )
        return report
