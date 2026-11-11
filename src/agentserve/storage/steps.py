"""Atomic step outputs, attempt reservations and startup recovery using the task store's transactions."""

import json
import time

from agentserve.domain.models import AgentError, TaskStateError
from agentserve.storage.repository import TaskRepository


class StepRepository:
    def __init__(self, tasks: TaskRepository):
        self.tasks = tasks

    async def visible_answer_history(self, task_id: str) -> list[str]:
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                """SELECT json_extract(prior.report_json,'$.answer') AS answer
                FROM tasks AS current JOIN tasks AS prior ON prior.session_id=current.session_id
                WHERE current.task_id=? AND prior.turn_no<current.turn_no
                    AND prior.config_json=current.config_json AND prior.status='COMPLETED'
                    AND json_extract(prior.report_json,'$.answer_source')='model'
                ORDER BY prior.turn_no DESC LIMIT 4""",
                (task_id,),
            )
        return [row["answer"] for row in rows]

    async def phase_draft_history(self, task_id: str, query: bool, schema_title: str | None) -> list[str]:
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                """SELECT json_extract(s.output_json, '$.reply.sql') AS sql,
                          json_extract(s.output_json, '$.reply.message.content') AS answer
                   FROM tasks current JOIN tasks prior ON prior.session_id=current.session_id
                   JOIN steps s ON s.task_id=prior.task_id
                   WHERE current.task_id=? AND prior.turn_no<current.turn_no
                     AND prior.status='COMPLETED' AND prior.config_json=current.config_json
                     AND s.kind='model' AND s.status='COMPLETED'
                     AND json_extract(s.input_json,'$.query')=?
                     AND json_extract(s.input_json,'$.json_schema.title') IS ?
                     AND ((?=1 AND json_extract(s.output_json,'$.reply.sql')=
                                   json_extract(prior.report_json,'$.sql'))
                          OR (?=0 AND json_type(s.output_json,'$.reply.message.content')='text'))
                   ORDER BY prior.turn_no DESC, s.step_no DESC LIMIT 4""",
                (task_id, query, schema_title, query, query),
            )
        if query:
            return [json.dumps(row["sql"], ensure_ascii=False)[1:-1] for row in rows]
        return [row["answer"] for row in rows]

    async def _owned(self, conn, task_id, version):
        rows = await conn.execute_fetchall(
            "SELECT status,version FROM tasks WHERE task_id=?", (task_id,)
        )
        if not rows or rows[0]["status"] != "RUNNING" or rows[0]["version"] != version:
            raise TaskStateError("STALE_COMMIT", "Task changed before step operation")

    async def open_step(
        self, task_id: str, version: int, step_no: int, kind: str, inputs: dict
    ) -> dict:
        encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True)
        async with self.tasks.transaction() as conn:
            await self._owned(conn, task_id, version)
            rows = await conn.execute_fetchall(
                "SELECT * FROM steps WHERE task_id=? AND step_no=?", (task_id, step_no)
            )
            if not rows:
                await conn.execute(
                    "INSERT INTO steps(task_id,step_no,kind,status,input_json,created_at) VALUES(?,?,?,'PENDING',?,?)",
                    (task_id, step_no, kind, encoded, time.time()),
                )
            elif rows[0]["kind"] != kind or rows[0]["input_json"] != encoded:
                raise AgentError(
                    "RECOVERY_INPUT_CHANGED",
                    "Step input or workflow differs from committed execution",
                )
            attempts = await conn.execute_fetchall(
                "SELECT * FROM attempts WHERE task_id=? AND step_no=? ORDER BY attempt_no",
                (task_id, step_no),
            )
        return {
            "step_no": step_no,
            "kind": kind,
            "status": rows[0]["status"] if rows else "PENDING",
            "output": json.loads(rows[0]["output_json"])
            if rows and rows[0]["output_json"]
            else None,
            "attempts": [dict(row) for row in attempts],
            "purpose": inputs.get("purpose"),
        }

    async def counters(self, task_id: str, *, memory: bool = False) -> dict:
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                """SELECT COUNT(*) AS model_calls,
                COALESCE(SUM(a.attempt_no>1),0) AS model_retries FROM attempts a JOIN steps s
                ON a.task_id=s.task_id AND a.step_no=s.step_no WHERE a.task_id=? AND s.kind='model'
                AND (json_extract(s.input_json,'$.purpose') IS NOT NULL)=?""",
                (task_id, memory),
            )
        return dict(rows[0])

    async def begin_attempt(
        self, task_id, version, step, *, model_limit=None, retry_limit=None
    ) -> dict:
        async with self.tasks.transaction() as conn:
            await self._owned(conn, task_id, version)
            rows = await conn.execute_fetchall(
                "SELECT status FROM steps WHERE task_id=? AND step_no=?", (task_id, step["step_no"])
            )
            if rows[0]["status"] != "PENDING":
                raise TaskStateError(
                    "STEP_ALREADY_COMMITTED", "Committed step cannot execute again"
                )
            number = (
                await conn.execute_fetchall(
                    "SELECT COUNT(*) FROM attempts WHERE task_id=? AND step_no=?",
                    (task_id, step["step_no"]),
                )
            )[0][0] + 1
            if step["kind"] == "model":
                total = (
                    await conn.execute_fetchall(
                        """SELECT COUNT(*) FROM attempts a JOIN steps s ON a.task_id=s.task_id
                        AND a.step_no=s.step_no WHERE a.task_id=? AND s.kind='model'
                        AND (json_extract(s.input_json,'$.purpose') IS NOT NULL)=?""",
                        (task_id, bool(step.get("purpose"))),
                    )
                )[0][0]
                if total >= model_limit:
                    raise AgentError(
                        "MODEL_BUDGET_EXHAUSTED",
                        "Maximum model requests reached, including interrupted attempts",
                    )
                if number > retry_limit + 1:
                    raise AgentError(
                        "MODEL_RETRY_EXHAUSTED",
                        "Interrupted model step exhausted its attempt budget",
                    )
            cursor = await conn.execute(
                "INSERT INTO attempts(task_id,step_no,attempt_no,status,started_at,process_epoch) VALUES(?,?,?,'RUNNING',?,?)",
                (task_id, step["step_no"], number, time.time(), self.tasks.process_epoch),
            )
            await self.tasks._event(
                conn,
                task_id,
                "ATTEMPT_STARTED",
                {
                    "step_no": step["step_no"],
                    "kind": step["kind"],
                    "attempt_id": cursor.lastrowid,
                    "attempt_no": number,
                },
            )
        return {"attempt_id": cursor.lastrowid, "attempt_no": number, "step_no": step["step_no"]}

    async def complete(
        self,
        task_id,
        version,
        attempt,
        output,
        event_kind,
        payload,
        *,
        failed=False,
        complete_step=True,
        retry_at=None,
    ):
        async with self.tasks.transaction() as conn:
            await self._owned(conn, task_id, version)
            changed = await conn.execute(
                """UPDATE attempts SET status=?,ended_at=?,retry_at=?,request_id=?,
                usage_json=?,error_json=?,outcome_unknown=? WHERE attempt_id=? AND status='RUNNING'""",
                (
                    "FAILED" if failed else "COMPLETED",
                    time.time(),
                    retry_at,
                    payload.get("request_id"),
                    json.dumps(payload.get("usage")),
                    json.dumps(
                        {"code": payload["code"], "detail": payload["detail"]} if failed else None
                    ),
                    payload.get("outcome_unknown", False),
                    attempt["attempt_id"],
                ),
            )
            if changed.rowcount != 1:
                raise TaskStateError("STALE_ATTEMPT", "Attempt no longer accepts an output")
            if complete_step:
                await conn.execute(
                    "UPDATE steps SET status='COMPLETED',output_json=?,completed_at=? WHERE task_id=? AND step_no=?",
                    (
                        json.dumps(output, ensure_ascii=False),
                        time.time(),
                        task_id,
                        attempt["step_no"],
                    ),
                )
            await self.tasks._event(
                conn,
                task_id,
                event_kind,
                {**payload, "step_no": attempt["step_no"], "attempt_id": attempt["attempt_id"]},
            )
            if complete_step:
                await self.tasks._event(
                    conn, task_id, "STEP_COMMITTED", {"step_no": attempt["step_no"]}
                )

    async def recover_running(self) -> int:
        async with self.tasks.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT task_id,execution_version FROM tasks WHERE status='RUNNING'"
            )
            for row in rows:
                task_id = row["task_id"]
                await self.tasks._interrupt_attempts(conn, task_id)
                if row["execution_version"] == 0:
                    error = {
                        "code": "LEGACY_INTERRUPTED",
                        "detail": "Pre-G4 task has no durable attempt budget; automatic replay is unavailable",
                    }
                    await conn.execute(
                        "UPDATE tasks SET status='FAILED',version=version+1,ended_at=?,error_json=? WHERE task_id=?",
                        (time.time(), json.dumps(error), task_id),
                    )
                    await self.tasks._event(conn, task_id, "TASK_FAILED", error)
                else:
                    await conn.execute(
                        "UPDATE tasks SET status='QUEUED',version=version+1 WHERE task_id=?",
                        (task_id,),
                    )
                    await self.tasks._event(conn, task_id, "TASK_RECOVERED", {})
        return len(rows)

    async def pause(self, task_id, version):
        async with self.tasks.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status,version FROM tasks WHERE task_id=?", (task_id,)
            )
            if rows and rows[0]["status"] == "RUNNING" and rows[0]["version"] == version:
                await self.tasks._interrupt_attempts(conn, task_id)
                await conn.execute(
                    "UPDATE tasks SET status='QUEUED',version=version+1 WHERE task_id=?", (task_id,)
                )
                await self.tasks._event(conn, task_id, "TASK_PAUSED", {})

    async def history(self, task_id):
        async with self.tasks.lock:
            steps = await self.tasks.connection.execute_fetchall(
                "SELECT * FROM steps WHERE task_id=? ORDER BY step_no", (task_id,)
            )
            attempts = await self.tasks.connection.execute_fetchall(
                "SELECT * FROM attempts WHERE task_id=? ORDER BY attempt_id", (task_id,)
            )
        result = []
        for row in steps:
            step = dict(row)
            step["input"] = json.loads(step.pop("input_json"))
            step["output"] = json.loads(step.pop("output_json") or "null")
            result.append(step)
        decoded = []
        for row in attempts:
            attempt = dict(row)
            attempt["usage"] = json.loads(attempt.pop("usage_json") or "null")
            attempt["error"] = json.loads(attempt.pop("error_json") or "null")
            decoded.append(attempt)
        return {
            "steps": result,
            "attempts": decoded,
            "budget": await self.counters(task_id),
            "memory_budget": await self.counters(task_id, memory=True),
        }
