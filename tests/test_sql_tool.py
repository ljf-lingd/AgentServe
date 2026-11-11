import asyncio
import hashlib
import json
import time

import pytest

from agentserve.tools.sqlite import SQLTool

LONG_QUERY = (
    "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n "
    "WHERE x<1000000000) SELECT SUM(x) FROM n"
)


@pytest.fixture
def tool(settings):
    instance = SQLTool(
        {"demo_orders": settings.databases["demo_orders"].path}, workers=1, timeout_seconds=0.03
    )
    yield instance
    instance.close()


@pytest.mark.parametrize(
    "sql,status",
    [
        ("SELECT COUNT(*) FROM customers", "ok"),
        ("WITH c AS (SELECT * FROM customers) SELECT COUNT(*) FROM c", "ok"),
        ("SELEC name FROM customers", "SQL_EXECUTION_ERROR"),
        ("SELECT missing FROM customers", "SQL_EXECUTION_ERROR"),
        ("UPDATE customers SET name='changed'", "POLICY_REJECTED"),
        ("WITH c AS (SELECT * FROM customers) DELETE FROM customers", "POLICY_REJECTED"),
        ("ATTACH DATABASE ':memory:' AS other", "POLICY_REJECTED"),
        ("PRAGMA query_only=OFF", "POLICY_REJECTED"),
        ("SELECT load_extension('unknown')", "POLICY_REJECTED"),
        ("SELECT randomblob(100000000)", "POLICY_REJECTED"),
        ("SELECT 1; SELECT 2", "SQL_EXECUTION_ERROR"),
    ],
)
async def test_execution_policy(tool, settings, sql, status):
    database = settings.databases["demo_orders"].path
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = await tool.execute("demo_orders", sql, deadline=time.monotonic() + 2)
    assert result.status == status
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


async def test_timeout_stops_worker_and_next_query_runs(tool):
    started = time.monotonic()
    result = await tool.execute("demo_orders", LONG_QUERY, deadline=started + 2)
    assert result.status == "SQL_TIMEOUT"
    assert time.monotonic() - started < 1
    result = await tool.execute(
        "demo_orders", "SELECT COUNT(*) FROM customers", deadline=time.monotonic() + 1
    )
    assert result.rows == [[2]]


async def test_cancel_joins_worker_and_releases_slot(tool):
    tool.timeout_seconds = 5
    pending = asyncio.create_task(
        tool.execute("demo_orders", LONG_QUERY, deadline=time.monotonic() + 5)
    )
    await asyncio.sleep(0.02)
    started = time.monotonic()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert time.monotonic() - started < 1
    result = await tool.execute("demo_orders", "SELECT 1", deadline=time.monotonic() + 1)
    assert result.rows == [[1]]


async def test_expired_deadline_never_executes(tool):
    result = await tool.execute("demo_orders", "SELECT 1", deadline=time.monotonic() - 1)
    assert result.status == "SQL_TIMEOUT"


async def test_deadline_includes_capacity_wait(tool):
    await tool.slots.acquire()
    try:
        result = await tool.execute("demo_orders", "SELECT 1", deadline=time.monotonic() + 0.01)
        assert result.status == "SQL_TIMEOUT"
    finally:
        tool.slots.release()
    assert (await tool.execute("demo_orders", "SELECT 1", deadline=time.monotonic() + 1)).rows == [
        [1]
    ]


async def test_row_limit_and_json_values(tool):
    tool.row_limit = 1
    result = await tool.execute(
        "demo_orders", "SELECT name, NULL, X'ABCD' FROM customers", deadline=time.monotonic() + 1
    )
    assert result.rows == [["Alice", None, {"hex": "abcd"}]]
    assert result.truncated and result.limit_reason == "rows"


async def test_byte_limit_and_oversize_metadata(tool):
    tool.byte_limit = 240
    sql = "SELECT '" + "x" * 80 + "' AS value FROM customers"
    result = await tool.execute("demo_orders", sql, deadline=time.monotonic() + 1)
    assert result.status == "ok" and result.truncated
    assert (
        len(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")).encode()) <= 240
    )
    for sql in ["SELECT '" + "x" * 1000 + "'", 'SELECT 1 AS "' + "x" * 400 + '" WHERE 0']:
        result = await tool.execute("demo_orders", sql, deadline=time.monotonic() + 1)
        assert result.status == "RESULT_TOO_LARGE"
