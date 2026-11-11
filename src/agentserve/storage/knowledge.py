"""Persist immutable knowledge evidence through the task repository."""
import json

from agentserve.domain.models import TaskStateError


class KnowledgeRepository:
    def __init__(self, tasks):
        self.tasks = tasks

    async def get(self, task_id):
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                "SELECT context_json FROM knowledge_contexts WHERE task_id=?", (task_id,)
            )
        return json.loads(rows[0][0]) if rows else None

    async def save(self, task_id, version, context):
        async with self.tasks.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status,version FROM tasks WHERE task_id=?", (task_id,)
            )
            if not rows or rows[0]["status"] != "RUNNING" or rows[0]["version"] != version:
                raise TaskStateError("STALE_COMMIT", "Task changed before knowledge commit")
            await conn.execute(
                "INSERT INTO knowledge_contexts VALUES (?,?)",
                (task_id, json.dumps(context, ensure_ascii=False)),
            )
            await self.tasks._event(conn, task_id, "KNOWLEDGE_CONTEXT_COMMITTED", {
                "snapshot_sha256": context["snapshot_sha256"],
                "chunk_ids": [c["chunk_id"] for c in context["chunks"]],
                "evidence_tokens": context["evidence_tokens"],
            })
