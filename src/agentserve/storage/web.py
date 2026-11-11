"""Browser history, access control, and read-only diagnostics over the task ledger."""

import json
import math
import time
from datetime import date, timedelta

from agentserve.domain.models import AgentError
from agentserve.storage.repository import TaskRepository


def task_mode(task: dict) -> str:
    return task.get("mode") or ("sql_explain" if task.get("answer_mode") else "sql")


def task_title(task: dict) -> str | None:
    if task.get("question"):
        return task["question"]
    window = task.get("analysis") or task.get("metric")
    if not window:
        return None
    last_day = date.fromisoformat(window["end"]) - timedelta(days=1)
    scope = window.get("city") or "全部城市"
    if task.get("analysis"):
        start, end = window["start"][:7], last_day.strftime("%Y-%m")
        return f"{start if start == end else f'{start} 至 {end}'} · {scope}"
    name = {"settled_revenue": "成交营收", "completed_orders": "完成订单量"}.get(window["name"], window["name"])
    return f"{name} · {scope} · {window['start']} 至 {last_day}"


def public_error(error: dict | None) -> dict | None:
    if not error:
        return None
    code = error["code"]
    detail = {
        "MODEL_TIMEOUT": "模型响应超时，可重新提交。",
        "MODEL_CONNECTION_ERROR": "模型服务连接失败，请稍后重试。",
        "MODEL_HTTP_ERROR": "模型服务暂时无法处理请求，请稍后重试。",
        "MODEL_OUTPUT_LIMIT": "回答超出模型输出限制，请缩小问题范围。",
        "QUEUE_FULL": "任务队列已满，请稍后重试。",
        "SQL_TIMEOUT": "查询超时，请缩小查询范围。",
        "SQL_EXECUTION_ERROR": "查询执行失败，可调整问题后重新提交。",
        "TASK_TIMEOUT": "任务超时，可缩小范围后重新提交。",
        "CONFIG_CHANGED": "运行配置已变更，请重新提交任务。",
        "SESSION_NOT_FOUND": "找不到该会话，请从列表选择或新建会话。",
        "TASK_NOT_FOUND": "找不到该任务，请从历史列表重新打开。",
        "MEMORY_PROTOCOL_ERROR": "我没能理解这句追问。你可以直接说要查的城市、日期和订单指标。",
        "MEMORY_CONTEXT_REQUIRED": "这句追问缺少上下文。请补充城市或完整问题。",
        "DATASET_MODE_UNAVAILABLE": "此数据集未开放所选功能。",
        "METRIC_WINDOW_UNCOVERED": "所选日期超出已批准的指标口径范围，请调整日期后再查。",
        "HISTORY_BUSY": "还有任务正在排队或执行。请等它结束，或先取消任务。",
        "HISTORY_NOT_READY": "查询完成后才能保存结果。",
    }.get(code, "任务处理失败。可调整输入后重新提交。")
    return {"code": code, "detail": detail}


def public_task(task: dict) -> dict:
    report = task["report"]
    result = {key: task[key] for key in (
        "task_id", "status", "created_at", "ended_at", "session_id", "turn_no"
    )}
    result["input"] = {key: task[key] for key in (
        "database_id", "question", "session_id", "mode", "answer_mode",
        "as_of", "clause_id", "metric", "analysis", "save_to_history",
    ) if task.get(key) is not None}
    result["error"] = public_error(task["error"])
    result["report"] = None if report is None else {
        **{key: report[key] for key in (
            "answer", "answer_source", "query_result", "metric", "analysis"
        ) if key in report},
        "citations": (report.get("knowledge") or {}).get("citations", []),
    }
    return result


class WebRepository:
    def __init__(self, tasks: TaskRepository):
        self.tasks = tasks

    @staticmethod
    async def _require_visible(conn, resource, resource_id, owner, *, standalone=False):
        if resource == "session":
            sql = """
                SELECT s.session_id FROM sessions s LEFT JOIN session_owners o USING(session_id)
                LEFT JOIN web_session_history h USING(session_id)
                WHERE s.session_id=? AND (? IS NULL OR o.user_id=?) AND h.deleted_at IS NULL
            """
        else:
            sql = """
                SELECT t.session_id FROM tasks t LEFT JOIN task_owners o USING(task_id)
                LEFT JOIN web_task_history h USING(task_id)
                LEFT JOIN web_session_history sh ON sh.session_id=t.session_id
                WHERE t.task_id=? AND (? IS NULL OR o.user_id=?)
                AND h.deleted_at IS NULL AND sh.deleted_at IS NULL
            """
        rows = await conn.execute_fetchall(sql, (resource_id, owner, owner))
        if not rows or (standalone and resource == "task" and rows[0]["session_id"] is not None):
            raise AgentError(f"{resource.upper()}_NOT_FOUND", f"No {resource} with this ID")

    async def require_visible(self, resource, resource_id, owner):
        async with self.tasks.lock:
            await self._require_visible(self.tasks.connection, resource, resource_id, owner)

    async def change_history(self, resource, resource_ids, owner, title):
        column = "session_id" if resource == "session" else "task_id"
        table = "web_session_history" if resource == "session" else "web_task_history"
        async with self.tasks.transaction() as conn:
            ids = tuple(dict.fromkeys(resource_ids))
            for resource_id in ids:
                await self._require_visible(conn, resource, resource_id, owner, standalone=True)
            if title is None and await conn.execute_fetchall(
                f"SELECT 1 FROM tasks WHERE {column} IN ({','.join('?' for _ in ids)}) "
                "AND status IN ('QUEUED','RUNNING') LIMIT 1", ids,
            ):
                raise AgentError("HISTORY_BUSY", "The history contains an active task")
            deleted_at = time.time() if title is None else None
            await conn.executemany(
                f"""INSERT INTO {table}({column},title,deleted_at) VALUES (?,?,?)
                    ON CONFLICT({column}) DO UPDATE SET title=excluded.title,
                    deleted_at=excluded.deleted_at""",
                [(resource_id, title, deleted_at) for resource_id in ids],
            )

    async def save_task(self, task_id, owner):
        async with self.tasks.transaction() as conn:
            await self._require_visible(conn, "task", task_id, owner, standalone=True)
            rows = await conn.execute_fetchall("SELECT status FROM tasks WHERE task_id=?", (task_id,))
            if rows[0]["status"] != "COMPLETED":
                raise AgentError("HISTORY_NOT_READY", "Only completed results can be saved")
            await conn.execute("INSERT OR IGNORE INTO web_task_history(task_id) VALUES(?)", (task_id,))

    async def saved_task(self, task_id):
        async with self.tasks.lock:
            return bool(await self.tasks.connection.execute_fetchall(
                "SELECT 1 FROM web_task_history WHERE task_id=? AND deleted_at IS NULL", (task_id,),
            ))

    async def sessions(self, database_id: str, owner: str | None, limit: int, offset: int):
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                """SELECT s.session_id, s.created_at, EXISTS (
                    SELECT 1 FROM tasks t WHERE t.session_id=s.session_id AND t.status IN ('QUEUED','RUNNING')
                ) AS active, COALESCE(h.title, (
                    SELECT json_extract(t.input_json, '$.question') FROM tasks t
                    WHERE t.session_id=s.session_id ORDER BY t.turn_no LIMIT 1
                )) AS title FROM sessions s LEFT JOIN session_owners o USING(session_id)
                LEFT JOIN web_session_history h USING(session_id)
                WHERE s.database_id=? AND (? IS NULL OR o.user_id=?)
                AND h.deleted_at IS NULL
                ORDER BY s.created_at DESC, s.session_id DESC LIMIT ? OFFSET ?""",
                (database_id, owner, owner, limit, offset),
            )
        return [dict(row) for row in rows]

    async def independent_tasks(self, database_id: str, owner: str | None, limit: int, offset: int, mode=None):
        async with self.tasks.lock:
            rows = await self.tasks.connection.execute_fetchall(
                """SELECT t.task_id,t.status,t.created_at,t.input_json,h.title
                FROM tasks t LEFT JOIN task_owners o USING(task_id)
                LEFT JOIN web_task_history h USING(task_id)
                WHERE t.session_id IS NULL AND json_extract(t.input_json, '$.database_id')=?
                AND (? IS NULL OR o.user_id=?) AND h.deleted_at IS NULL
                AND (? IS NULL OR json_extract(t.input_json, '$.mode')=?)
                AND (json_extract(t.input_json, '$.save_to_history') IS NOT 0 OR h.task_id IS NOT NULL)
                ORDER BY t.created_at DESC,t.task_id DESC LIMIT ? OFFSET ?""",
                (database_id, owner, owner, mode, mode, limit, offset),
            )
        return [{
            "task_id": row["task_id"], "status": row["status"], "created_at": row["created_at"],
            "mode": task_mode(task := json.loads(row["input_json"])),
            "title": row["title"] or task_title(task),
        } for row in rows]

    async def reset_overview(self):
        async with self.tasks.transaction() as conn:
            await conn.execute(
                """INSERT INTO web_overview VALUES (1,?) ON CONFLICT(id)
                DO UPDATE SET cleared_at=excluded.cleared_at""", (time.time(),)
            )

    async def overview(self):
        async with self.tasks.lock:
            conn = self.tasks.connection
            fetched_at = time.time()
            cleared = await conn.execute_fetchall("SELECT cleared_at FROM web_overview WHERE id=1")
            cleared_at = cleared[0]["cleared_at"] if cleared else None
            since = max(fetched_at - 86400, cleared_at or 0)
            active = await conn.execute_fetchall(
                """SELECT status, count(*) AS count FROM tasks
                WHERE status IN ('QUEUED','RUNNING') GROUP BY status"""
            )
            terminal = await conn.execute_fetchall(
                """SELECT status, count(*) AS count FROM tasks
                WHERE ended_at>=? GROUP BY status""", (since,)
            )
            successful = await conn.execute_fetchall(
                """SELECT json_extract(input_json, '$.mode') AS mode,
                    json_extract(input_json, '$.answer_mode') AS answer_mode,
                    ended_at-created_at AS jct_seconds FROM tasks
                WHERE ended_at>=? AND status='COMPLETED' ORDER BY jct_seconds""", (since,)
            )
            errors = await conn.execute_fetchall(
                """SELECT task_id, input_json, error_json, ended_at FROM tasks
                WHERE status='FAILED' AND ended_at>=? ORDER BY ended_at DESC,task_id DESC LIMIT 10""",
                (since,)
            )
            requests = await conn.execute_fetchall(
                """SELECT a.task_id,a.request_id,a.ended_at,a.usage_json,
                    COALESCE(json_extract(s.input_json,'$.purpose'), CASE
                        WHEN json_extract(s.input_json,'$.query')=1 THEN 'sql'
                        WHEN json_extract(s.input_json,'$.json_schema') IS NOT NULL THEN 'knowledge_answer'
                        ELSE 'answer' END) AS phase,
                    s.output_json,t.input_json
                FROM attempts a JOIN steps s USING(task_id,step_no) JOIN tasks t USING(task_id)
                WHERE a.status='COMPLETED' AND a.request_id IS NOT NULL AND a.ended_at>=?
                ORDER BY a.attempt_id DESC LIMIT 30""", (since,)
            )
        counts = dict.fromkeys(("QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"), 0)
        counts.update({row["status"]: row["count"] for row in [*active, *terminal]})
        jct: dict[str, list[float]] = {}
        for row in successful:
            jct.setdefault(task_mode(dict(row)), []).append(row["jct_seconds"])
        return {
            "fetched_at": fetched_at,
            "cleared_at": cleared_at,
            "counts": counts,
            "jct": [{
                "mode": mode, "n": len(values),
                "p50_seconds": values[math.ceil(len(values) * .50) - 1],
                "p95_seconds": values[math.ceil(len(values) * .95) - 1],
            } for mode, values in jct.items()],
            "errors": [{
                "task_id": row["task_id"], "ended_at": row["ended_at"],
                "mode": task_mode(json.loads(row["input_json"])),
                "error": public_error(json.loads(row["error_json"] or "null")),
            } for row in errors],
            "requests": [{
                "task_id": row["task_id"], "request_id": row["request_id"],
                "ended_at": row["ended_at"], "mode": task_mode(json.loads(row["input_json"])),
                "phase": row["phase"],
                "output_tokens": (json.loads(row["usage_json"] or "null") or {}).get("completion_tokens"),
                "metrics": (json.loads(row["output_json"] or "{}").get("reply") or {}).get("metrics"),
            } for row in requests],
        }
