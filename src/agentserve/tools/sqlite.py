"""Bounded SQLite execution off the event loop; cancellation stops the SQLite VM."""

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from agentserve.domain.models import QueryResult

FUNCTIONS = {
    "like",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "coalesce",
    "round",
    "abs",
    "lower",
    "upper",
    "length",
    "substr",
    "date",
    "strftime",
    "ifnull",
}
FUNCTION_PROFILES = {
    "baseline": frozenset(FUNCTIONS),
    "with_replace_v1": frozenset(FUNCTIONS | {"replace"}),
}


class SQLTool:
    """Only configured database IDs cross this boundary; a model never supplies a path."""

    def __init__(
        self,
        databases: dict[str, Path],
        *,
        workers: int = 2,
        timeout_seconds: float = 1,
        row_limit: int = 100,
        byte_limit: int = 4096,
        function_profile: str = "baseline",
    ):
        self.databases = databases
        self.timeout_seconds = timeout_seconds
        self.row_limit = row_limit
        self.byte_limit = byte_limit
        self.functions = FUNCTION_PROFILES[function_profile]
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agentserve-sql")
        self.slots = asyncio.Semaphore(workers)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    async def execute(
        self, database_id: str, sql: str, *, deadline: float, parameters: tuple[str, ...] = ()
    ) -> QueryResult:
        # The task deadline includes waiting for SQL capacity. Per-query time starts in the worker.
        try:
            async with asyncio.timeout_at(deadline):
                await self.slots.acquire()
        except TimeoutError:
            return QueryResult(
                status="SQL_TIMEOUT", error="Deadline expired waiting for SQL capacity"
            )
        stop = threading.Event()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self.executor, self._execute, self.databases[database_id], sql, deadline, stop,
                parameters,
            )
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                stop.set()
                # Keep the slot until the worker has stopped and closed its connection.
                await asyncio.shield(future)
                raise
        finally:
            self.slots.release()

    def _execute(
        self, database: Path, sql: str, deadline: float, stop: threading.Event,
        parameters: tuple[str, ...] = (),
    ) -> QueryResult:
        result = QueryResult()
        deadline = min(deadline, time.monotonic() + self.timeout_seconds)
        denied, expired = False, False

        def authorize(action, arg1, arg2, database_name, trigger):
            nonlocal denied
            allowed = action in {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_RECURSIVE,
            }
            if action == sqlite3.SQLITE_FUNCTION:
                allowed = arg2.lower() in self.functions
            denied |= not allowed
            return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

        def progress():
            nonlocal expired
            expired = time.monotonic() >= deadline
            return int(expired or stop.is_set())

        if progress():
            return QueryResult(status="SQL_TIMEOUT", error="SQL deadline expired before execution")
        try:
            with closing(
                sqlite3.connect(
                    database.as_uri() + "?mode=ro",
                    uri=True,
                    timeout=min(0.1, max(0.001, deadline - time.monotonic())),
                )
            ) as conn:
                conn.execute("PRAGMA query_only=ON")
                conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 65536)
                conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16000)
                conn.set_authorizer(authorize)
                conn.set_progress_handler(progress, 1000)
                cursor = conn.execute(sql, parameters)
                if cursor.description is None:
                    return QueryResult(
                        status="POLICY_REJECTED", error="A read-only query is required"
                    )
                result.columns = [column[0] for column in cursor.description]
                if self._payload_size(result) > self.byte_limit:
                    return QueryResult(
                        status="RESULT_TOO_LARGE", error="Column metadata exceeds budget"
                    )
                for row in cursor:
                    if progress():
                        return QueryResult(
                            status="SQL_TIMEOUT", error="SQL deadline expired fetching rows"
                        )
                    if len(result.rows) == self.row_limit:
                        result.truncated, result.limit_reason = True, "rows"
                        break
                    result.rows.append(
                        [{"hex": v.hex()} if isinstance(v, bytes) else v for v in row]
                    )
                    result.truncated, result.limit_reason = True, "bytes"
                    if self._payload_size(result) > self.byte_limit:
                        result.rows.pop()
                        if not result.rows:
                            return QueryResult(
                                status="RESULT_TOO_LARGE", error="First row exceeds budget"
                            )
                        break
                    result.truncated, result.limit_reason = False, None
                if progress():
                    return QueryResult(status="SQL_TIMEOUT", error="SQL deadline expired")
        except sqlite3.DatabaseError as error:
            status = (
                "SQL_TIMEOUT"
                if expired or stop.is_set()
                else ("POLICY_REJECTED" if denied else "SQL_EXECUTION_ERROR")
            )
            return QueryResult(status=status, error=str(error)[:200])
        return result

    @staticmethod
    def _payload_size(result: QueryResult) -> int:
        return len(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")).encode())
