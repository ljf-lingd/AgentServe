"""Replay committed external-step outputs; keep persistence out of model and SQL adapters."""

from dataclasses import asdict

from agentserve.domain.models import AgentError, ModelReply, ModelRequestError, QueryResult
from agentserve.storage.steps import StepRepository


class StepJournal:
    def __init__(self, store: StepRepository, task_id: str, version: int, emit, *, hook=None):
        self.store, self.task_id, self.version = store, task_id, version
        self.emit, self.hook = emit, hook
        self.cursor = 0

    async def open(self, kind, inputs):
        step = await self.store.open_step(self.task_id, self.version, self.cursor, kind, inputs)
        self.cursor += 1
        return step

    async def boundary(self, stage, step):
        if self.hook is not None:
            await self.hook(stage, self.task_id, step["step_no"], step["kind"])

    async def reuse_model(self, step):
        await self.emit("STEP_REUSED", {"step_no": step["step_no"], "kind": "model"})
        output = step["output"]
        if "error" in output:
            error = output["error"]
            if "retryable" in error:
                raise ModelRequestError(
                    error["code"],
                    error["detail"],
                    retryable=error["retryable"],
                    outcome_unknown=error["outcome_unknown"],
                )
            raise AgentError(error["code"], error["detail"])
        return ModelReply(**output["reply"])

    async def begin(self, step, settings=None):
        limit = (
            (settings.memory.max_model_calls if step.get("purpose") else settings.max_model_calls)
            if settings
            else None
        )
        return await self.store.begin_attempt(
            self.task_id,
            self.version,
            step,
            model_limit=limit,
            retry_limit=settings.max_model_retries if settings else None,
        )

    async def model_success(self, step, attempt, reply, payload):
        await self.boundary("after_execute", step)
        await self.store.complete(
            self.task_id,
            self.version,
            attempt,
            {"reply": asdict(reply)},
            "MODEL_COMPLETED",
            payload,
        )
        await self.emit("MODEL_COMPLETED", payload, persist=False)
        await self.boundary("after_commit", step)

    async def model_failure(self, step, attempt, error, payload, *, retry_at=None):
        encoded = {"code": error.code, "detail": str(error)}
        if isinstance(error, ModelRequestError):
            encoded.update(retryable=error.retryable, outcome_unknown=error.outcome_unknown)
        await self.store.complete(
            self.task_id,
            self.version,
            attempt,
            {"error": encoded},
            "MODEL_FAILED",
            payload,
            failed=True,
            complete_step=retry_at is None,
            retry_at=retry_at,
        )
        await self.emit("MODEL_FAILED", payload, persist=False)
        await self.boundary("after_commit", step)

    async def execute_sql(self, tool, database_id, sql, *, deadline, parameters=()):
        inputs = {"database_id": database_id, "sql": sql}
        bindings = {"parameters": parameters} if parameters else {}
        if parameters:
            inputs["parameters"] = list(parameters)
        step = await self.open("sql", inputs)
        if step["status"] == "COMPLETED":
            await self.emit("STEP_REUSED", {"step_no": step["step_no"], "kind": "sql"})
            return QueryResult(**step["output"]["result"])
        attempt = await self.begin(step)
        await self.emit(
            "SQL_STARTED",
            {**inputs, "step_no": step["step_no"], "attempt_id": attempt["attempt_id"]},
        )
        await self.boundary("before_execute", step)
        result = await tool.execute(database_id, sql, deadline=deadline, **bindings)
        await self.boundary("after_execute", step)
        payload = {"result": result.to_dict()}
        await self.store.complete(
            self.task_id, self.version, attempt, payload, "SQL_COMPLETED", payload
        )
        await self.emit("SQL_COMPLETED", payload, persist=False)
        await self.boundary("after_commit", step)
        return result
