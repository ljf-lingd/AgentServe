import json
import sqlite3
import time

from agentserve.storage.repository import TaskRepository
from agentserve.storage.steps import StepRepository


async def test_v1_migration_preserves_results_events_and_foreign_keys(tmp_path):
    path = tmp_path / "v1.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE tasks(task_id TEXT PRIMARY KEY,input_json TEXT NOT NULL,
                status TEXT CHECK(status IN ('QUEUED','RUNNING','COMPLETED','FAILED')),
                version INTEGER NOT NULL DEFAULT 0,idempotency_key TEXT UNIQUE,
                created_at REAL NOT NULL,started_at REAL,ended_at REAL,deadline_at REAL NOT NULL,
                config_json TEXT NOT NULL,report_json TEXT,error_json TEXT);
            CREATE INDEX tasks_by_status ON tasks(status,created_at);
            CREATE TABLE events(event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT REFERENCES tasks(task_id),type TEXT,wall_time REAL,
                offset_seconds REAL,payload_json TEXT);
            CREATE INDEX events_by_task ON events(task_id,event_id);
            PRAGMA user_version=1;
        """)
        conn.execute(
            "INSERT INTO tasks(task_id,input_json,status,created_at,deadline_at,config_json,report_json) VALUES(?,?,?,?,?,?,?)",
            (
                "saved",
                json.dumps({"database_id": "demo", "question": "old"}),
                "COMPLETED",
                time.time(),
                time.time() + 90,
                "{}",
                json.dumps({"answer": "old result"}),
            ),
        )
        conn.execute(
            "INSERT INTO events(event_id,task_id,type,wall_time,payload_json) VALUES(17,'saved','TASK_COMPLETED',?,'{}')",
            (time.time(),),
        )
    repo = await TaskRepository.open(path)
    try:
        task = await repo.get("saved")
        assert task["report"] == {"answer": "old result"} and task["cancel_requested"] == 0
        assert (await repo.events("saved"))[0]["event_id"] == 17
        assert (await repo.connection.execute_fetchall("PRAGMA user_version"))[0][0] == 6
        assert (await repo.connection.execute_fetchall("PRAGMA foreign_keys"))[0][0] == 1
        assert await repo.connection.execute_fetchall("PRAGMA foreign_key_check") == []
        assert await repo.cancel("saved") == ("COMPLETED", False)
    finally:
        await repo.close()


async def test_v2_running_task_without_attempt_ledger_is_not_replayed(tmp_path):
    path = tmp_path / "v2.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE tasks(task_id TEXT PRIMARY KEY,input_json TEXT NOT NULL,status TEXT,
                version INTEGER NOT NULL DEFAULT 0,idempotency_key TEXT UNIQUE,
                created_at REAL NOT NULL,started_at REAL,ended_at REAL,deadline_at REAL NOT NULL,
                config_json TEXT NOT NULL,report_json TEXT,error_json TEXT,cancel_requested INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX tasks_by_status ON tasks(status,created_at);
            CREATE TABLE events(event_id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT REFERENCES tasks(task_id),
                type TEXT,wall_time REAL,offset_seconds REAL,payload_json TEXT);
            CREATE INDEX events_by_task ON events(task_id,event_id);
            PRAGMA user_version=2;
        """)
        conn.execute(
            "INSERT INTO tasks(task_id,input_json,status,created_at,deadline_at,config_json) VALUES(?,?,'RUNNING',?,?,?)",
            (
                "legacy",
                json.dumps({"database_id": "demo", "question": "old"}),
                time.time(),
                time.time() + 90,
                "{}",
            ),
        )
    repo = await TaskRepository.open(path)
    try:
        assert await StepRepository(repo).recover_running() == 1
        task = await repo.get("legacy")
        assert task["status"] == "FAILED" and task["error"]["code"] == "LEGACY_INTERRUPTED"
        assert await repo.connection.execute_fetchall("PRAGMA foreign_key_check") == []
    finally:
        await repo.close()
