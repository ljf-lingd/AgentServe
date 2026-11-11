"""Session and memory persistence, using the task repository's sole transaction owner."""

import json
import time
from uuid import uuid4

from agentserve.domain.models import AgentError, TaskStateError
from agentserve.storage.repository import TaskRepository


class MemoryRepository:
    def __init__(self, tasks: TaskRepository):
        self.tasks = tasks

    async def create_session(
        self, database_id: str, policy_identity: str, *, owner_id: str | None = None
    ) -> dict:
        session = {
            "session_id": uuid4().hex,
            "database_id": database_id,
            "policy_identity": policy_identity,
            "created_at": time.time(),
        }
        async with self.tasks.transaction() as conn:
            await conn.execute(
                "INSERT INTO sessions(session_id,database_id,policy_identity,created_at) VALUES(?,?,?,?)",
                tuple(session.values()),
            )
            if owner_id is not None:
                await conn.execute(
                    "INSERT INTO session_owners VALUES (?,?)", (session["session_id"], owner_id)
                )
        return session

    async def history(
        self, session_id: str, *, after_turn: int = 0, limit: int = 50, full: bool = False
    ) -> list[dict]:
        async with self.tasks.lock:
            exists = await self.tasks.connection.execute_fetchall(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            )
            if not exists:
                raise AgentError("SESSION_NOT_FOUND", "No session with this ID")
            rows = await self.tasks.connection.execute_fetchall(
                "SELECT * FROM tasks WHERE session_id=? AND turn_no>? ORDER BY turn_no LIMIT ?",
                (session_id, after_turn, limit),
            )
        decode = self.tasks.decode_task if full else self._turn
        return [decode(row) for row in rows]

    @staticmethod
    def _turn(row) -> dict:
        report = json.loads(row["report_json"] or "null")
        return {
            "task_id": row["task_id"],
            "turn_no": row["turn_no"],
            "status": row["status"],
            "question": json.loads(row["input_json"])["question"],
            "sql": report["sql"] if report else None,
            "answer": report["answer"] if report else None,
            "created_at": row["created_at"],
            "error": json.loads(row["error_json"] or "null"),
        }

    async def inputs(self, task_id: str, recent_turns: int) -> dict:
        async with self.tasks.lock:
            row = (
                await self.tasks.connection.execute_fetchall(
                    "SELECT session_id,turn_no,created_at FROM tasks WHERE task_id=?", (task_id,)
                )
            )[0]
            session = (
                await self.tasks.connection.execute_fetchall(
                    "SELECT revision FROM sessions WHERE session_id=?", (row["session_id"],)
                )
            )[0]
            recent = await self.tasks.connection.execute_fetchall(
                """SELECT * FROM tasks WHERE session_id=? AND turn_no<?
                AND status IN ('COMPLETED','FAILED','CANCELLED') ORDER BY turn_no DESC LIMIT ?""",
                (row["session_id"], row["turn_no"], recent_turns),
            )
            entries = await self.tasks.connection.execute_fetchall(
                "SELECT * FROM memory_entries WHERE session_id=? AND turn_no<? ORDER BY turn_no,entry_id",
                (row["session_id"], row["turn_no"]),
            )
        return {
            "cutoff_turn": row["turn_no"] - 1,
            "revision": session["revision"],
            "accepted_at": row["created_at"],
            "recent": [self._turn(r) for r in reversed(recent)],
            "entries": [
                {**json.loads(e["entry_json"]), "vector": json.loads(e["vector_json"])}
                for e in entries
            ],
        }

    async def context(self, task_id: str) -> dict | None:
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                "SELECT context_json FROM memory_contexts WHERE task_id=?", (task_id,)
            )
        return json.loads(rows[0][0]) if rows else None

    async def save_context(self, task_id: str, version: int, context: dict) -> None:
        async with self.tasks.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status,version FROM tasks WHERE task_id=?", (task_id,)
            )
            if not rows or rows[0]["status"] != "RUNNING" or rows[0]["version"] != version:
                raise TaskStateError("STALE_COMMIT", "Task changed before context commit")
            await conn.execute(
                "INSERT INTO memory_contexts VALUES(?,?)",
                (
                    task_id,
                    json.dumps(context, ensure_ascii=False),
                ),
            )
            await self.tasks._event(
                conn,
                task_id,
                "MEMORY_CONTEXT_COMMITTED",
                {
                    "cutoff_turn": context["cutoff_turn"],
                    "source_ids": context["source_ids"],
                    "token_count": context["token_count"],
                    "policy_version": context["policy_version"],
                },
            )
