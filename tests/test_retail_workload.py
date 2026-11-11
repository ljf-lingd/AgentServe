"""Small test suite that distinguishes common incorrect retail SQL."""

import hashlib
import json
import runpy
import sqlite3
from pathlib import Path

import httpx
import pytest
from conftest import completion
from test_memory import resources
from test_task_api import terminal

from agentserve.agents.retail_sql import RETAIL_DATABASE, use_retail_date_index
from agentserve.api.app import create_app
from agentserve.config import MemorySettings
from agentserve.storage.memory import MemoryRepository

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = runpy.run_path(str(ROOT / "scripts/setup/prepare_online_retail_ii.py"))


def test_retail_view_distinguishes_invoice_grain_and_adjustments():
    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA["SCHEMA"] + SCHEMA["VIEW"])
        db.executemany(
            """INSERT INTO retail_lines(source_sheet,source_row,invoice,stock_code,
               quantity,invoice_date,unit_price_millionths,customer_id,country)
               VALUES('test',?,?,'sku',?,'2011-11-15',?,?,'Germany')""",
            [(1, "1", 2, 1_250_000, None), (2, "1", 1, 2_000_000, "A"),
             (3, "C2", -2, 3_000_000, "A"), (4, "C2", -1, 500_000, "A"),
             (5, "c3", 1, 7_000_000, "B"), (6, "4", -1, 9_000_000, None),
             (7, "5", 1, 0, "B")],
        )
        assert db.execute(
            "SELECT sum(sales_gbp),sum(cancellation_gbp),count(DISTINCT customer_id) FROM retail_transactions"
        ).fetchone() == (4.5, 6.5, 2)
        assert db.execute(
            "SELECT count(DISTINCT invoice),count(*) FROM retail_transactions WHERE is_cancellation = 1"
        ).fetchone() == (2, 3)


@pytest.mark.parametrize(("predicate", "expected"), [
    ("strftime('%Y-%m', r.invoice_date) = '2010-12'", "(r.invoice_date >= '2010-12-01' AND r.invoice_date < '2011-01-01')"),
    ("strftime('%Y-%m-%d', invoice_date) >= '2010-12-31' AND strftime('%Y-%m-%d', r.invoice_date) < '2011-01-01'",
     "invoice_date >= '2010-12-31' AND r.invoice_date < '2011-01-01'"),
])
def test_retail_rewrite_preserves_date_boundaries(predicate, expected):
    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA["SCHEMA"] + SCHEMA["VIEW"])
        db.executemany(
            """INSERT INTO retail_lines(source_sheet,source_row,invoice,stock_code,
               quantity,invoice_date,unit_price_millionths,customer_id,country)
               VALUES('test',?,?,'sku',1,?,1000000,'A','Germany')""",
            [(row, str(row), value) for row, value in enumerate([
                "2010-11-30 23:59:59", "2010-12-31 00:00:00", "2010-12-31 23:59:59", "2011-01-01 00:00:00",
            ], 1)],
        )
        prefix = "SELECT r.invoice FROM retail_transactions AS r WHERE "
        suffix = " AND country = 'Germany' ORDER BY invoice;"
        original = prefix + predicate + suffix
        rewritten = use_retail_date_index(original)
        assert rewritten == prefix + expected + suffix
        assert db.execute(original).fetchall() == db.execute(rewritten).fetchall() == [("2",), ("3",)]


@pytest.mark.parametrize("statement", [
    "SELECT * FROM retail_transactions WHERE invoice_date >= '2010-12-01' AND invoice_date < '2011-01-01'",
    "SELECT strftime('%Y-%m', invoice_date) FROM retail_transactions WHERE country = 'GB'",
    "SELECT count(*) FROM retail_transactions WHERE country = 'a AND strftime(''%Y-%m'', invoice_date) = ''2010-12'''",
    "SELECT * FROM retail_transactions WHERE strftime('%Y-%m', invoice_date) = '2010-12' -- comment",
    "SELECT * FROM retail_transactions WHERE strftime('%Y-%m', invoice_date) = '2010-12' OR country = 'GB'",
    "SELECT * FROM retail_transactions WHERE strftime('%Y-%m', invoice_date) = '2010-13'",
    "SELECT * FROM retail_transactions WHERE strftime('%Y-%m-%d', invoice_date) < '2011-01-01'",
    "SELECT * FROM retail_transactions WHERE strftime('%Y-%m-%d', invoice_date, 'localtime') >= '2010-12-01'",
    "SELECT * FROM retail_transactions JOIN other USING (invoice) WHERE strftime('%Y-%m', invoice_date) = '2010-12'",
    "SELECT * FROM retail_lines WHERE strftime('%Y-%m', invoice_date) = '2010-12'",
])
def test_retail_rewrite_leaves_other_queries_intact(statement):
    assert use_retail_date_index(statement) == statement


async def test_retail_task_journals_the_executed_statement_in_a_v7_session(settings):
    database = settings.databases.pop("demo_orders")
    settings.databases[RETAIL_DATABASE] = database
    settings.answer_style = "evidence_direct_v5"
    settings.memory = MemorySettings(resources_identity="test-pinned")
    source = ROOT / "src/agentserve"
    v7_memory_policy = {
        "settings": settings.memory.model_dump(mode="json"),
        "source": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((source / "memory").glob("*.py"))},
        "sql_agent_sha256": "31dc7f292c934a2ebb3ceeb8f7fee8a0a4d8dd02c8c12da114c7f8d6a1d4f0af",
        "model_execution_sha256": hashlib.sha256((source / "llm/execution.py").read_bytes()).hexdigest(),
    }
    v7_session_identity = hashlib.sha256(json.dumps(v7_memory_policy, sort_keys=True).encode()).hexdigest()
    schema = SCHEMA["SCHEMA"] + SCHEMA["VIEW"]
    database.schema_path.write_text(schema, encoding="utf-8")
    with sqlite3.connect(database.path) as conn:
        conn.executescript(schema)
        conn.execute(
            """INSERT INTO retail_lines(source_sheet,source_row,invoice,stock_code,
               quantity,invoice_date,unit_price_millionths,customer_id,country)
               VALUES('test',1,'1','sku',1,'2010-12-31 23:59:59',1000000,'A','Germany')"""
        )
    prefix = "SELECT count(*) FROM retail_transactions WHERE "
    proposed = prefix + "strftime('%Y-%m', invoice_date) = '2010-12'"
    executed = prefix + "(invoice_date >= '2010-12-01' AND invoice_date < '2011-01-01')"
    app = create_app(
        settings,
        model_transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion(sql=proposed))),
        resource_transport=httpx.MockTransport(resources),
    )
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test",
    ) as client:
        sessions = MemoryRepository(app.state.tasks.repository)
        legacy_session = await sessions.create_session(RETAIL_DATABASE, v7_session_identity)
        created = await client.post("/tasks", json={
            "database_id": RETAIL_DATABASE, "question": "Count December lines",
            "session_id": legacy_session["session_id"],
        })
        task_id = created.raise_for_status().json()["task_id"]
        task = await terminal(client, task_id)
        assert task["status"] == "COMPLETED", task["error"]
        assert task["report"]["sql"] == executed
        assert task["report"]["query_result"]["rows"] == [[1]]
        steps = (await client.get(f"/tasks/{task_id}/steps")).raise_for_status().json()
        assert [step["input"]["sql"] for step in steps["steps"] if step["kind"] == "sql"] == [executed]

