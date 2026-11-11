"""Synchronous read-only SQLite tool for preflight, not a production task runtime."""

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

FUNCTIONS = {"count", "sum", "avg", "min", "max", "coalesce", "round", "abs",
             "lower", "upper", "length", "substr", "date", "strftime", "ifnull"}


def execute_query(database: Path, sql: str, *, timeout_seconds: float = 1.0,
                  row_limit: int = 100, byte_limit: int = 16384) -> dict:
    """Return a bounded JSON result; the caller chooses a fixed database path."""
    result = {"status": "ok", "columns": [], "rows": [], "truncated": False,
              "limit_reason": None, "error": None}
    deadline = time.monotonic() + timeout_seconds
    denied, expired = False, False

    def authorize(action, arg1, arg2, database_name, trigger):
        nonlocal denied
        allowed = action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
        if action == sqlite3.SQLITE_FUNCTION:
            allowed = arg2.lower() in FUNCTIONS
        if not allowed:
            denied = True
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

    def progress():
        nonlocal expired
        expired = time.monotonic() >= deadline
        return int(expired)

    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 65536)
            conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16000)
            conn.set_authorizer(authorize)
            conn.set_progress_handler(progress, 1000)
            cursor = conn.execute(sql)
            if cursor.description is None:
                result.update(status="POLICY_REJECTED", error="A single read-only query is required")
                return result
            result["columns"] = [column[0] for column in cursor.description]
            if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) > byte_limit:
                result.update(status="RESULT_TOO_LARGE", columns=[], error="Column metadata exceeds the result byte budget")
                return result
            for row in cursor:
                if len(result["rows"]) == row_limit:
                    result.update(truncated=True, limit_reason="rows")
                    break
                values = [{"hex": value.hex()} if isinstance(value, bytes) else value for value in row]
                result["rows"].append(values)
                # Reserve the largest truncation metadata while checking the complete payload.
                result.update(truncated=True, limit_reason="bytes")
                payload_bytes = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
                if payload_bytes > byte_limit:
                    result["rows"].pop()
                    if not result["rows"]:
                        result.update(status="RESULT_TOO_LARGE", error="First row exceeds the result byte budget")
                    break
                result.update(truncated=False, limit_reason=None)
    except sqlite3.DatabaseError as error:
        status = "SQL_TIMEOUT" if expired else "POLICY_REJECTED" if denied else "SQL_EXECUTION_ERROR"
        result.update(status=status, columns=[], rows=[], truncated=False,
                      limit_reason=None, error=str(error)[:200])
    return result
