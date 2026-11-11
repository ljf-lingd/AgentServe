"""Versioned state schema; old migration definitions stay stable as execution evolves."""

V2_TASK_COLUMNS = """
    task_id TEXT PRIMARY KEY, input_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('QUEUED','RUNNING','COMPLETED','FAILED','CANCELLED')),
    version INTEGER NOT NULL DEFAULT 0, idempotency_key TEXT UNIQUE,
    created_at REAL NOT NULL, started_at REAL, ended_at REAL, deadline_at REAL NOT NULL,
    config_json TEXT NOT NULL, report_json TEXT, error_json TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1))
"""
STEP_TABLES = """
CREATE TABLE IF NOT EXISTS steps (
    task_id TEXT NOT NULL REFERENCES tasks(task_id), step_no INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('model','sql')),
    status TEXT NOT NULL CHECK(status IN ('PENDING','COMPLETED')),
    input_json TEXT NOT NULL, output_json TEXT, created_at REAL NOT NULL, completed_at REAL,
    PRIMARY KEY(task_id,step_no)
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, step_no INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('RUNNING','COMPLETED','FAILED','INTERRUPTED')),
    started_at REAL NOT NULL, ended_at REAL, retry_at REAL, request_id TEXT, usage_json TEXT,
    error_json TEXT, outcome_unknown INTEGER, process_epoch TEXT NOT NULL,
    FOREIGN KEY(task_id,step_no) REFERENCES steps(task_id,step_no),
    UNIQUE(task_id,step_no,attempt_no)
);
CREATE INDEX IF NOT EXISTS attempts_by_task ON attempts(task_id,attempt_id);
"""
MEMORY_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, database_id TEXT NOT NULL, policy_identity TEXT NOT NULL,
    created_at REAL NOT NULL, revision INTEGER NOT NULL DEFAULT 0, next_turn INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS memory_contexts (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), context_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id), turn_no INTEGER NOT NULL,
    entry_json TEXT NOT NULL, vector_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_by_session ON memory_entries(session_id,turn_no);
CREATE UNIQUE INDEX IF NOT EXISTS tasks_by_session_turn ON tasks(session_id,turn_no)
    WHERE session_id IS NOT NULL;
"""
KNOWLEDGE_TABLES = """
CREATE TABLE IF NOT EXISTS knowledge_contexts (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), context_json TEXT NOT NULL
);
"""
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tasks ({V2_TASK_COLUMNS}, execution_version INTEGER NOT NULL DEFAULT 1,
    session_id TEXT, turn_no INTEGER);
CREATE INDEX IF NOT EXISTS tasks_by_status ON tasks(status, created_at);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id), type TEXT NOT NULL,
    wall_time REAL NOT NULL, offset_seconds REAL, payload_json TEXT NOT NULL,
    process_epoch TEXT NOT NULL DEFAULT 'legacy'
);
CREATE INDEX IF NOT EXISTS events_by_task ON events(task_id,event_id);
{STEP_TABLES}
{MEMORY_TABLES}
{KNOWLEDGE_TABLES}
CREATE TABLE IF NOT EXISTS metric_bindings (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), binding_json TEXT NOT NULL
);
PRAGMA user_version=6;
"""

WEB_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_users (
    user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','admin'))
);
CREATE TABLE IF NOT EXISTS web_logins (
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES web_users(user_id),
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS session_owners (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    user_id TEXT NOT NULL REFERENCES web_users(user_id)
);
CREATE TABLE IF NOT EXISTS task_owners (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    user_id TEXT NOT NULL REFERENCES web_users(user_id)
);
CREATE TABLE IF NOT EXISTS web_session_history (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    title TEXT, deleted_at REAL
);
CREATE TABLE IF NOT EXISTS web_task_history (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    title TEXT, deleted_at REAL
);
CREATE TABLE IF NOT EXISTS web_overview (
    id INTEGER PRIMARY KEY CHECK(id=1), cleared_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS web_sessions_by_user ON session_owners(user_id, session_id);
CREATE INDEX IF NOT EXISTS web_tasks_by_user ON task_owners(user_id, task_id);
CREATE INDEX IF NOT EXISTS web_tasks_by_end ON tasks(ended_at DESC, task_id DESC);
"""
MIGRATE_V1_TO_V2 = f"""
PRAGMA foreign_keys=OFF;
BEGIN IMMEDIATE;
CREATE TABLE tasks_v2 ({V2_TASK_COLUMNS});
INSERT INTO tasks_v2(task_id,input_json,status,version,idempotency_key,created_at,started_at,
    ended_at,deadline_at,config_json,report_json,error_json)
SELECT task_id,input_json,status,version,idempotency_key,created_at,started_at,
    ended_at,deadline_at,config_json,report_json,error_json FROM tasks;
DROP TABLE tasks;
ALTER TABLE tasks_v2 RENAME TO tasks;
CREATE INDEX tasks_by_status ON tasks(status,created_at);
PRAGMA user_version=2;
COMMIT;
PRAGMA foreign_keys=ON;
"""
MIGRATE_V2_TO_V3 = f"""
BEGIN IMMEDIATE;
ALTER TABLE tasks ADD COLUMN execution_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE events ADD COLUMN process_epoch TEXT NOT NULL DEFAULT 'legacy';
{STEP_TABLES}
PRAGMA user_version=3;
COMMIT;
"""

MIGRATE_V3_TO_V4 = """
BEGIN IMMEDIATE;
ALTER TABLE tasks ADD COLUMN session_id TEXT;
ALTER TABLE tasks ADD COLUMN turn_no INTEGER;
PRAGMA user_version=4;
COMMIT;
"""
