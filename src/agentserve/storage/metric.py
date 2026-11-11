"""Commit the approved binding before any execution against the order database."""

import json

from agentserve.domain.models import TaskStateError


class MetricRepository:
    def __init__(self, tasks):
        self.tasks = tasks

    async def get(self, task_id):
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                "SELECT binding_json FROM metric_bindings WHERE task_id=?", (task_id,)
            )
        return json.loads(rows[0][0]) if rows else None

    async def save(self, task_id, version, binding, *, analysis=False):
        async with self.tasks.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status,version FROM tasks WHERE task_id=?", (task_id,)
            )
            if not rows or rows[0]["status"] != "RUNNING" or rows[0]["version"] != version:
                raise TaskStateError("STALE_COMMIT", "Task changed before metric binding commit")
            await conn.execute(
                "INSERT INTO metric_bindings VALUES (?,?)",
                (task_id, json.dumps(binding, ensure_ascii=False)),
            )
            if analysis:
                kind = "ANALYSIS_BINDING_COMMITTED"
                payload = {
                    "registry_sha256": binding["registry_sha256"],
                    "rule_sha256": [value["rule_sha256"] for value in binding["metrics"]],
                    "chunk_ids": [value["citation"]["chunk_id"] for value in binding["metrics"]],
                }
            else:
                kind = "METRIC_BINDING_COMMITTED"
                payload = {
                    "registry_sha256": binding["registry_sha256"],
                    "rule_sha256": binding["rule_sha256"],
                    "chunk_id": binding["citation"]["chunk_id"],
                }
            await self.tasks._event(conn, task_id, kind, payload)
