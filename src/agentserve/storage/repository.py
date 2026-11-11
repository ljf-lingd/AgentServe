"""One-process SQLite task repository. No model or query-tool work inside transactions."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import aiosqlite

from agentserve.domain.models import AgentError, RunReport, TaskInput, TaskStateError
from agentserve.storage.schema import (
    MIGRATE_V1_TO_V2,
    MIGRATE_V2_TO_V3,
    MIGRATE_V3_TO_V4,
    SCHEMA,
    WEB_SCHEMA,
)


class TaskRepository:
    def __init__(self, connection: aiosqlite.Connection, *, web: bool = False):
        self.connection = connection
        self.web = web
        self.lock = asyncio.Lock()
        self.process_epoch = uuid4().hex

    @classmethod
    async def open(cls, path: Path, *, web: bool = False) -> "TaskRepository":
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path, isolation_level=None)
        try:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA busy_timeout=5000")
            version = (await conn.execute_fetchall("PRAGMA user_version"))[0][0]
            if version not in (0, 1, 2, 3, 4, 5, 6):
                raise RuntimeError(f"Unsupported state schema version: {version}")
            if version == 1:
                await conn.executescript(MIGRATE_V1_TO_V2)
            if version in (1, 2):
                await conn.executescript(MIGRATE_V2_TO_V3)
            if version in (1, 2, 3):
                await conn.executescript(MIGRATE_V3_TO_V4)
            await conn.executescript(SCHEMA)
            if web:
                await conn.executescript(WEB_SCHEMA)
            conn.row_factory = aiosqlite.Row
            return cls(conn, web=web)
        except BaseException:
            await conn.close()
            raise

    async def close(self) -> None:
        await self.connection.close()

    @asynccontextmanager
    async def transaction(self):
        async with self.lock:
            await self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

    async def _event(
        self, conn, task_id: str, kind: str, payload: dict, *, offset_seconds: float | None = None
    ) -> None:
        await conn.execute(
            "INSERT INTO events(task_id,type,wall_time,offset_seconds,payload_json,process_epoch) VALUES(?,?,?,?,?,?)",
            (
                task_id,
                kind,
                time.time(),
                offset_seconds,
                json.dumps(payload, ensure_ascii=False),
                self.process_epoch,
            ),
        )

    async def submit(
        self, task: TaskInput, *, config: dict, queue_limit: int,
        idempotency_key: str | None = None, owner_id: str | None = None,
    ) -> tuple[str, bool]:
        input_json = task.model_dump_json(exclude_none=True)
        async with self.transaction() as conn:
            if idempotency_key is not None:
                rows = await conn.execute_fetchall(
                    "SELECT task_id,input_json,session_id FROM tasks WHERE idempotency_key=?",
                    (idempotency_key,),
                )
                if rows:
                    existing = rows[0]
                    if self.web and await conn.execute_fetchall(
                        """SELECT 1 FROM web_task_history WHERE task_id=? AND deleted_at IS NOT NULL
                        UNION ALL SELECT 1 FROM web_session_history
                        WHERE session_id=? AND deleted_at IS NOT NULL""",
                        (existing["task_id"], existing["session_id"]),
                    ):
                        raise AgentError("TASK_NOT_FOUND", "The original task was removed from history")
                    if rows[0]["input_json"] != input_json:
                        raise AgentError(
                            "IDEMPOTENCY_CONFLICT", "Same key was used for a different task"
                        )
                    return rows[0]["task_id"], False
            turn_no = None
            if task.session_id is not None:
                visible = (
                    " AND NOT EXISTS (SELECT 1 FROM web_session_history h "
                    "WHERE h.session_id=sessions.session_id AND h.deleted_at IS NOT NULL)"
                    if self.web else ""
                )
                sessions = await conn.execute_fetchall(
                    "SELECT * FROM sessions WHERE session_id=?" + visible, (task.session_id,)
                )
                if not sessions or (owner_id is not None and not await conn.execute_fetchall(
                    "SELECT 1 FROM session_owners WHERE session_id=? AND user_id=?",
                    (task.session_id, owner_id),
                )):
                    raise AgentError("SESSION_NOT_FOUND", "No session with this ID")
                session = sessions[0]
                if session["database_id"] != task.database_id:
                    raise AgentError(
                        "SESSION_DATABASE_MISMATCH", "Session is bound to another database"
                    )
                identity = config["execution_identity"].get("memory_policy_sha256")
                if session["policy_identity"] != identity:
                    raise AgentError(
                        "SESSION_POLICY_CHANGED", "Start a new session for this memory policy"
                    )
                turn_no = session["next_turn"]
                await conn.execute(
                    "UPDATE sessions SET next_turn=next_turn+1 WHERE session_id=?",
                    (task.session_id,),
                )
            queued = (
                await conn.execute_fetchall("SELECT COUNT(*) FROM tasks WHERE status='QUEUED'")
            )[0][0]
            if queued >= queue_limit:
                raise AgentError("QUEUE_FULL", "Task queue capacity reached")
            task_id, now = uuid4().hex, time.time()
            await conn.execute(
                """INSERT INTO tasks(task_id,input_json,status,idempotency_key,
                created_at,deadline_at,config_json,execution_version,session_id,turn_no)
                VALUES(?,?,'QUEUED',?,?,?,?,1,?,?)""",
                (
                    task_id,
                    input_json,
                    idempotency_key,
                    now,
                    now + config["task_timeout_seconds"],
                    json.dumps(config),
                    task.session_id,
                    turn_no,
                ),
            )
            if owner_id is not None:
                await conn.execute(
                    "INSERT INTO task_owners VALUES (?,?)", (task_id, owner_id)
                )
            await self._event(conn, task_id, "TASK_ACCEPTED", {})
        return task_id, True

    async def get(self, task_id: str) -> dict | None:
        async with self.lock:
            rows = await self.connection.execute_fetchall(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            )
        if not rows:
            return None
        return self.decode_task(rows[0])

    @staticmethod
    def decode_task(row) -> dict:
        row = dict(row)
        task = json.loads(row.pop("input_json"))
        row.pop("idempotency_key")
        row["config_snapshot"] = json.loads(row.pop("config_json"))
        row["report"] = json.loads(row.pop("report_json") or "null")
        row["error"] = json.loads(row.pop("error_json") or "null")
        return {**row, **task}

    async def events(
        self, task_id: str, *, after_event_id: int = 0, limit: int = 100
    ) -> list[dict]:
        async with self.lock:
            rows = await self.connection.execute_fetchall(
                "SELECT * FROM events WHERE task_id=? AND event_id>? ORDER BY event_id LIMIT ?",
                (task_id, after_event_id, limit),
            )
        return [
            {
                **{k: value for k, value in dict(row).items() if k != "payload_json"},
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    async def claim_next(self) -> dict | None:
        async with self.transaction() as conn:
            rows = await conn.execute_fetchall(
                """SELECT t.task_id FROM tasks t WHERE t.status='QUEUED'
                AND (t.session_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM tasks p WHERE p.session_id=t.session_id
                    AND p.turn_no<t.turn_no AND p.status IN ('QUEUED','RUNNING')
                )) ORDER BY t.created_at,t.task_id LIMIT 1"""
            )
            if not rows:
                return None
            task_id = rows[0]["task_id"]
            await conn.execute(
                "UPDATE tasks SET status='RUNNING',version=version+1,started_at=COALESCE(started_at,?),execution_version=1 WHERE task_id=?",
                (time.time(), task_id),
            )
            await self._event(conn, task_id, "TASK_CLAIMED", {})
        return await self.get(task_id)

    async def append_event(self, event: dict) -> None:
        async with self.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status FROM tasks WHERE task_id=?", (event["task_id"],)
            )
            if not rows or rows[0]["status"] != "RUNNING":
                raise TaskStateError("TASK_NOT_RUNNING", "Task no longer accepts execution events")
            await self._event(
                conn,
                event["task_id"],
                event["type"],
                event["payload"],
                offset_seconds=event["offset_seconds"],
            )

    async def record_artifact_download(self, task_id: str, name: str, seconds: float) -> None:
        async with self.transaction() as conn:
            await self._event(
                conn, task_id, "ARTIFACT_PREPARED", {"name": name, "client_seconds": seconds}
            )

    async def finish(self, report: RunReport, *, expected_version: int) -> None:
        output = report.to_dict()
        terminal = output.pop("events")[-1]
        async with self.transaction() as conn:
            changed = await conn.execute(
                """UPDATE tasks SET status=?,version=version+1,ended_at=?,
                report_json=?,error_json=? WHERE task_id=? AND status='RUNNING' AND version=?""",
                (
                    report.status,
                    time.time(),
                    json.dumps(output, ensure_ascii=False),
                    json.dumps(report.error),
                    report.task_id,
                    expected_version,
                ),
            )
            if changed.rowcount != 1:
                raise TaskStateError("STALE_COMMIT", "Task changed before result commit")
            if report.status == "COMPLETED" and report.task.get("session_id"):
                session_id = report.task["session_id"]
                row = (
                    await conn.execute_fetchall(
                        "SELECT turn_no FROM tasks WHERE task_id=?", (report.task_id,)
                    )
                )[0]
                for entry in report.memory_draft or []:
                    values = {k: v for k, v in entry.items() if k != "vector"}
                    await conn.execute(
                        "INSERT INTO memory_entries VALUES(?,?,?,?,?,?)",
                        (
                            entry["entry_id"],
                            session_id,
                            report.task_id,
                            row["turn_no"],
                            json.dumps(values, ensure_ascii=False),
                            json.dumps(entry["vector"]),
                        ),
                    )
                await conn.execute(
                    "UPDATE sessions SET revision=revision+1 WHERE session_id=?", (session_id,)
                )
                await self._event(
                    conn,
                    report.task_id,
                    "MEMORY_COMMITTED",
                    {"entry_ids": [e["entry_id"] for e in report.memory_draft or []]},
                )
            await self._event(
                conn,
                report.task_id,
                terminal["type"],
                terminal["payload"],
                offset_seconds=terminal["offset_seconds"],
            )

    async def fail(self, task_id: str, code: str, detail: str) -> None:
        async with self.transaction() as conn:
            changed = await conn.execute(
                """UPDATE tasks SET status='FAILED',version=version+1,
                ended_at=?,error_json=? WHERE task_id=? AND status='RUNNING'""",
                (time.time(), json.dumps({"code": code, "detail": detail}), task_id),
            )
            if changed.rowcount == 1:
                await self._event(conn, task_id, "TASK_FAILED", {"code": code, "detail": detail})

    async def cancel(self, task_id: str) -> tuple[str, bool]:
        async with self.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            )
            if not rows:
                raise AgentError("TASK_NOT_FOUND", "No task with this ID")
            previous = rows[0]["status"]
            if previous not in {"QUEUED", "RUNNING"}:
                return previous, False
            error = {"code": "USER_CANCELLED", "detail": "Task cancelled by caller"}
            await conn.execute(
                """UPDATE tasks SET status='CANCELLED',cancel_requested=1,
                version=version+1,ended_at=?,error_json=? WHERE task_id=?""",
                (time.time(), json.dumps(error), task_id),
            )
            await self._event(
                conn,
                task_id,
                "TASK_CANCELLED",
                {
                    "previous_status": previous,
                    "application_stop": "requested" if previous == "RUNNING" else "not_started",
                    "gpu_generation_stopped": None,
                },
            )
        return "CANCELLED", True

    async def record_execution_stopped(self, task_id: str) -> None:
        async with self.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            )
            if rows and rows[0]["status"] == "CANCELLED":
                await self._interrupt_attempts(conn, task_id)
                await self._event(
                    conn, task_id, "APPLICATION_EXECUTION_STOPPED", {"gpu_generation_stopped": None}
                )

    async def _interrupt_attempts(self, conn, task_id: str) -> None:
        rows = await conn.execute_fetchall(
            "SELECT attempt_id,step_no FROM attempts WHERE task_id=? AND status='RUNNING'",
            (task_id,),
        )
        await conn.execute(
            "UPDATE attempts SET status='INTERRUPTED',ended_at=?,outcome_unknown=1 WHERE task_id=? AND status='RUNNING'",
            (time.time(), task_id),
        )
        for row in rows:
            await self._event(
                conn,
                task_id,
                "ATTEMPT_INTERRUPTED",
                {"attempt_id": row["attempt_id"], "step_no": row["step_no"], "usage": None},
            )
