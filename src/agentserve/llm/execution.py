"""Shared model capacity and bounded transport retries, independent of SQL repair policy."""

import asyncio
import time
from collections.abc import Awaitable, Callable

from agentserve.config import Settings
from agentserve.domain.models import AgentError, ModelReply, ModelRequestError, RunReport
from agentserve.llm.client import SQL_TOOLS, LLMBackend


class ModelExecutor:
    def __init__(self, backend: LLMBackend, settings: Settings):
        self.backend, self.settings = backend, settings
        self.slots = asyncio.Semaphore(settings.llm_concurrency)
        self.resources = None

    async def generate(
        self,
        messages: list[dict],
        *,
        query: bool,
        max_tokens: int,
        deadline: float,
        report: RunReport,
        emit: Callable[[str, dict], Awaitable[None]],
        journal=None,
        purpose: str | None = None,
        json_schema: dict | None = None,
        draft_answer: bool = False,
    ) -> ModelReply:
        limit = self.settings.memory.max_model_calls if purpose else self.settings.max_model_calls

        def used():
            return report.memory_calls if purpose else report.model_calls - report.memory_calls

        inputs = {"messages": messages, "query": query, "max_tokens": max_tokens}
        if purpose:
            inputs["purpose"] = purpose
        structured = purpose is not None or json_schema is not None
        if structured:
            inputs["json_schema"] = json_schema
        phase = purpose or ("knowledge_answer" if structured else "sql" if query else "answer")
        # Missing conversational evidence can be clarified before SQL. Once SQLite has
        # reported a schema/syntax error, use the existing bounded SQL repair path.
        allow_clarify = bool(query and report.task.get("session_id") and report.sql_repairs == 0)
        # S0 resolves/clarifies intent before SQL. Once exact state lookup succeeds,
        # execute that resolved request; a second confirmation stage caused observed regressions.
        if (
            allow_clarify
            and self.settings.memory
            and self.settings.memory.policy in {"state", "state_on_demand"}
            and report.memory_context
            and report.memory_context.get("plan", {}).get("can_execute")
        ):
            allow_clarify = False
        if allow_clarify:
            inputs["allow_clarify"] = True
        step = await journal.open("model", inputs) if journal else None
        if step and step["status"] == "COMPLETED":
            return await journal.reuse_model(step)
        draft_history = None
        if self.settings.session_sam and draft_answer and phase == "answer":
            draft_history = (
                await journal.store.visible_answer_history(journal.task_id)
                if journal and report.task.get("session_id")
                else []
            )
        if self.resources is not None:
            await self.resources.guard(
                messages,
                None if structured else SQL_TOOLS,
                max_tokens,
                self.settings.memory.model_context_tokens,
                deadline,
            )
        previous = len(step["attempts"]) if step else 0
        if previous:
            if used() >= limit:
                raise AgentError(
                    "MODEL_BUDGET_EXHAUSTED", "Interrupted attempts consumed the task model budget"
                )
            if previous > self.settings.max_model_retries:
                raise AgentError(
                    "MODEL_RETRY_EXHAUSTED", "Interrupted model step exhausted its attempt budget"
                )
            last = step["attempts"][-1]
            retry_at = last["retry_at"] or last[
                "ended_at"
            ] + self.settings.model_retry_backoff_seconds * (2 ** (previous - 1))
            delay = max(0, retry_at - time.time())
            report.model_retries += 1
            await emit(
                "MODEL_RETRY_SCHEDULED",
                {
                    "retry": report.model_retries,
                    "delay_seconds": delay,
                    "code": "RECOVERED_ATTEMPT",
                    "outcome_unknown": last["outcome_unknown"],
                },
            )
            async with asyncio.timeout_at(deadline):
                await asyncio.sleep(delay)
        for attempt in range(previous, self.settings.max_model_retries + 1):
            if used() >= limit:
                raise AgentError("MODEL_BUDGET_EXHAUSTED", "Maximum model requests reached")
            await emit(
                "MODEL_QUEUED",
                {"phase": phase, "attempt": attempt + 1},
            )
            queued_at = time.monotonic()
            try:
                async with asyncio.timeout_at(deadline):
                    await self.slots.acquire()
            except TimeoutError as error:
                raise AgentError(
                    "MODEL_TIMEOUT", "Deadline expired waiting for model capacity"
                ) from error
            failure = None
            try:
                handle = await journal.begin(step, self.settings) if journal else None
                if journal:
                    counters = await journal.store.counters(report.task_id)
                    memory_counters = await journal.store.counters(report.task_id, memory=True)
                    report.memory_calls = memory_counters["model_calls"]
                    report.model_calls, report.model_retries = (
                        counters["model_calls"] + report.memory_calls,
                        counters["model_retries"] + memory_counters["model_retries"],
                    )
                else:
                    report.model_calls += 1
                    if purpose:
                        report.memory_calls += 1
                await emit(
                    "MODEL_STARTED",
                    {
                        "call": report.model_calls,
                        "phase": phase,
                        "attempt": attempt + 1,
                        "capacity_wait_seconds": time.monotonic() - queued_at,
                    },
                )
                started = time.monotonic()
                if journal:
                    await journal.boundary("before_execute", step)
                try:
                    kwargs = {"json_mode": True, "json_schema": json_schema} if structured else {}
                    if allow_clarify:
                        kwargs["allow_clarify"] = True
                    if draft_history is not None:
                        kwargs["draft_history"] = draft_history
                    reply = await self.backend.generate(
                        messages,
                        query=query,
                        max_tokens=max_tokens,
                        deadline=min(
                            deadline, started + self.settings.model_request_timeout_seconds
                        ),
                        **kwargs,
                    )
                except AgentError as error:
                    failure = error
                    payload = {
                        "code": error.code,
                        "detail": str(error),
                        "usage": None,
                        "retryable": isinstance(error, ModelRequestError) and error.retryable,
                        "outcome_unknown": error.outcome_unknown
                        if isinstance(error, ModelRequestError)
                        else None,
                        "client_seconds": time.monotonic() - started,
                    }
                    can_retry = (
                        isinstance(error, ModelRequestError)
                        and error.retryable
                        and attempt < self.settings.max_model_retries
                        and used() < limit
                    )
                    terminal_error = error
                    if (
                        isinstance(error, ModelRequestError)
                        and error.retryable
                        and attempt < self.settings.max_model_retries
                        and used() >= limit
                    ):
                        terminal_error = AgentError(
                            "MODEL_BUDGET_EXHAUSTED",
                            "Retry would exceed total model request budget",
                        )
                        failure = terminal_error
                    if journal:
                        retry_at = (
                            time.time() + self.settings.model_retry_backoff_seconds * (2**attempt)
                            if can_retry
                            else None
                        )
                        await journal.model_failure(
                            step, handle, terminal_error, payload, retry_at=retry_at
                        )
                    else:
                        await emit("MODEL_FAILED", payload)
                else:
                    payload = {
                        "request_id": reply.request_id,
                        "usage": reply.usage,
                        "finish_reason": reply.finish_reason,
                        "message": reply.message,
                        "client_seconds": time.monotonic() - started,
                    }
                    if journal:
                        await journal.model_success(step, handle, reply, payload)
                    else:
                        await emit("MODEL_COMPLETED", payload)
                    return reply
            finally:
                self.slots.release()
            if not isinstance(failure, ModelRequestError) or not failure.retryable:
                raise failure
            if attempt == self.settings.max_model_retries:
                raise failure
            if used() >= limit:
                raise AgentError(
                    "MODEL_BUDGET_EXHAUSTED", "Retry would exceed total model request budget"
                )
            report.model_retries += 1
            delay = self.settings.model_retry_backoff_seconds * (2**attempt)
            await emit(
                "MODEL_RETRY_SCHEDULED",
                {
                    "retry": report.model_retries,
                    "delay_seconds": delay,
                    "code": failure.code,
                    "outcome_unknown": failure.outcome_unknown,
                },
            )
            # Model capacity is released during backoff. The task's absolute deadline still applies.
            try:
                async with asyncio.timeout_at(deadline):
                    await asyncio.sleep(delay)
            except TimeoutError as error:
                raise AgentError(
                    "MODEL_TIMEOUT", "Task deadline expired during retry backoff"
                ) from error
        raise AssertionError("Unreachable model retry state")
