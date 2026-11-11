"""Historical model responses are replayed here, never labeled new inference."""

import json

import pytest
from conftest import ROOT, completion
from test_agent import run_script

from agentserve.domain.models import TaskInput


@pytest.mark.skipif(
    not (ROOT / "results/preflight/model-probes.json").exists(),
    reason="Private historical probe fixture is not part of a clean source checkout",
)
async def test_historical_wrong_table_column_is_repaired(settings):
    # Reuse the exact first failed completion, then script a valid repair. This tests policy.
    probes = json.loads((ROOT / "results/preflight/model-probes.json").read_text(encoding="utf-8"))
    record = next(row for row in probes["probes"] if row["id"] == "completed_revenue")
    settings.databases["demo_orders"].path = ROOT / "data/databases/demo_orders/demo_orders.sqlite"
    settings.databases["demo_orders"].schema_path = ROOT / "data/processed/demo_orders/schema.sql"
    correct = (
        "SELECT SUM(oi.quantity * oi.unit_price_cents) FROM orders o "
        "JOIN order_items oi ON o.order_id=oi.order_id WHERE o.status='completed'"
    )
    report, requests = await run_script(
        settings,
        [
            record["generation_response"],
            completion(correct),
            completion(answer="总金额为189300分。"),
        ],
        task=TaskInput(database_id="demo_orders", question=record["question"]),
    )
    assert report.query_result.rows == [[189300]] and report.sql_repairs == 1
    assert "no such column: o.quantity" in requests[1]["messages"][-1]["content"]
