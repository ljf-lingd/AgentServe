import sqlite3

import httpx
import pytest
from conftest import completion

from agentserve.agents.sql import SQLAgent
from agentserve.agents.sql_policy import covers_order_period, intent_violations
from agentserve.application.service import TaskService
from agentserve.domain.models import TaskInput
from agentserve.llm.client import VLLMClient
from agentserve.tools.sqlite import SQLTool


def test_status_coverage_and_all_status_override():
    assert intent_violations("SELECT COUNT(*) FROM orders", "已完成订单笔数")
    assert not intent_violations(
        "SELECT COUNT(*) FROM orders WHERE status='completed'", "已完成订单笔数"
    )
    assert not intent_violations(
        "SELECT COUNT(*) FROM orders", "包括所有订单状态，并非只有completed"
    )


def test_settled_existence_accepts_count_and_rejects_amount_only():
    scope = " FROM orders WHERE status='completed'"
    existence = "深圳26年一月有交易吗？"
    amount = "SELECT SUM(order_id)" + scope
    assert intent_violations(amount, existence, settled=True)
    assert not intent_violations("SELECT COUNT(*)" + scope, existence, settled=True)
    assert not intent_violations(amount, "深圳有交易吗，交易额是多少？", settled=True)
    items = " FROM orders JOIN order_items USING(order_id) WHERE status='completed'"
    assert intent_violations("SELECT COUNT(*)" + items, existence, settled=True)
    assert not intent_violations("SELECT COUNT(DISTINCT orders.order_id)" + items, existence, settled=True)


@pytest.mark.parametrize("predicate,accepted", [
    ("o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01'", True),
    ("o.order_date BETWEEN '2026-01-01' AND '2026-03-31'", True),
    ("strftime('%Y-%m',o.order_date) BETWEEN '2026-01' AND '2026-03'", True),
    ("strftime('%Y-%m',o.order_date) >= '2026-01' AND strftime('%Y-%m',o.order_date) <= '2026-03'", True),
    ("o.order_date < '2026-04-01'", False),
    ("o.order_date >= '2026-01-01' AND o.order_date < '2026-03-01'", False),
    ("o.order_date >= '2026-01-01' AND o.order_date < '2026-03-31'", False),
    ("o.order_date >= '2026-01-01' AND o.order_date <= '2026-04-01'", False),
    ("strftime('%Y-%m',o.order_date) = '2026-03'", False),
    ("o.order_date BETWEEN '2025-01-01' AND '2025-03-31'", False),
])
def test_month_span_sql_requires_both_verified_boundaries(predicate, accepted):
    sql = "SELECT SUM(amount) FROM orders o WHERE " + predicate
    assert covers_order_period(sql, ["2026-01-01", "2026-04-01"]) is accepted


def test_first_product_assignment_is_not_category_revenue_grain():
    wrong = "SELECT p.category,SUM(i.quantity*i.unit_price_cents) FROM products p JOIN orders o ON p.product_id=(SELECT product_id FROM order_items WHERE order_id=o.order_id LIMIT 1) JOIN order_items i ON i.order_id=o.order_id WHERE o.status='completed' GROUP BY p.category"
    assert any(
        "first product" in reason
        for reason in intent_violations(wrong, "completed category revenue")
    )
    correct = "SELECT p.category,SUM(i.quantity*i.unit_price_cents) FROM products p JOIN order_items i ON p.product_id=i.product_id JOIN orders o ON o.order_id=i.order_id WHERE o.status='completed' GROUP BY p.category"
    assert not intent_violations(correct, "completed category revenue")


def test_grounded_city_is_required_but_check_does_not_claim_sql_equivalence():
    state = {"fields": {"city": {"operator": "eq", "value": "Shanghai"}}}
    assert intent_violations("SELECT COUNT(*) FROM customers", "继续", state)
    assert not intent_violations(
        "SELECT COUNT(*) FROM customers WHERE city='Shanghai'", "继续", state
    )


async def test_missing_status_is_repaired_before_database_execution(settings):
    db = settings.databases["demo_orders"]
    with sqlite3.connect(db.path) as conn:
        conn.executescript(
            "CREATE TABLE orders(order_id INTEGER,status TEXT); INSERT INTO orders VALUES(1,'completed'),(2,'pending');"
        )
    db.schema_path.write_text(
        db.schema_path.read_text() + "\nCREATE TABLE orders(order_id INTEGER,status TEXT);"
    )
    settings.sql_prompt_version = "evidence_v21"
    replies = [
        completion(sql="SELECT COUNT(*) FROM orders"),
        completion(sql="SELECT COUNT(*) FROM orders WHERE status='completed'"),
        completion(answer="1笔"),
    ]
    calls = []

    def model(request):
        calls.append(request)
        return httpx.Response(200, json=replies[len(calls) - 1])

    tool = SQLTool({"demo_orders": db.path})
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(model)
        ) as client:
            report = await TaskService(
                SQLAgent(VLLMClient(client, settings.model), tool, settings)
            ).run(TaskInput(database_id="demo_orders", question="已完成订单有多少笔"))
        assert report.status == "COMPLETED" and report.query_result.rows == [[1]]
        assert report.sql_repairs == 1 and report.model_calls == 3
        assert sum(e["type"] == "SQL_STARTED" for e in report.events) == 1
        assert (
            next(e for e in report.events if e["type"] == "SQL_INTENT_CHECK_FAILED")["payload"][
                "executed"
            ]
            is False
        )
    finally:
        tool.close()
