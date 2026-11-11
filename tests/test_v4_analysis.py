import csv
import io
from xml.etree import ElementTree

import httpx
import pytest

from agentserve.api.app import create_app
from agentserve.config import MetricSettings, Settings
from test_task_api import terminal
from test_v32_metrics import DATABASE, ROOT, RULES

BODY = {
    "mode": "analysis",
    "database_id": DATABASE,
    "analysis": {"start": "2026-01-01", "end": "2026-08-01"},
}


@pytest.fixture
def analysis_settings(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v221.C1.json")
    settings.state_database = tmp_path / "state.sqlite"
    settings.memory = None
    settings.metric = MetricSettings(registry_path=RULES)
    settings.analysis_enabled = True
    return settings


async def test_analysis_one_sql_and_saved_artifacts(analysis_settings):
    transport = httpx.MockTransport(lambda _: pytest.fail("Analysis used the model"))
    app = create_app(analysis_settings, model_transport=transport)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        submitted = (await client.post("/tasks", json=BODY)).raise_for_status().json()
        path = f"/tasks/{submitted['task_id']}"
        row = await terminal(client, submitted["task_id"])
        assert row["status"] == "COMPLETED", row["error"]
        report = row["report"]["analysis"]
        assert report["months"][-1] == {"month": "2026-07", "revenue_cents": None, "order_count": 0}
        assert report["totals"]["order_count"] == sum(row["order_count"] for row in report["months"])
        assert [r["citation"]["chunk_id"] for r in report["metrics"]] == [
            "settled-revenue@1#1", "completed-orders@1#1",
        ]
        chart = await client.get(path + "/artifacts/chart.svg")
        assert ElementTree.fromstring(chart.content).tag.endswith("svg")
        csv_text = (await client.get(path + "/artifacts/monthly.csv")).raise_for_status().text
        assert list(csv.DictReader(io.StringIO(csv_text)))[-1]["revenue_cents"] == ""
        assert (await client.get(path + "/artifacts/secret")).status_code == 404
        events = (await client.get(path + "/events")).json()["events"]
        kinds = [event["type"] for event in events]
        assert kinds.count("SQL_STARTED") == 1 and kinds.count("ARTIFACT_PREPARED") == 2
        assert kinds.index("ANALYSIS_BINDING_COMMITTED") < kinds.index("SQL_STARTED")


async def test_analysis_refuses_incomplete_result_and_unsafe_input(analysis_settings):
    analysis_settings.row_limit = 1
    app = create_app(analysis_settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        bad = {**BODY, "analysis": {**BODY["analysis"], "start": "2026-01-02"}}
        assert (await client.post("/tasks", json=bad)).status_code == 422
        row = (await client.post("/tasks", json=BODY)).raise_for_status().json()
        task = await terminal(client, row["task_id"])
        assert task["error"]["code"] == "ANALYSIS_RESULT_TRUNCATED"
        assert task["report"]["query_result"]["truncated"] and "analysis" not in task["report"]
        assert (await client.get(f"/tasks/{row['task_id']}/artifacts/report.md")).status_code == 404


async def test_workbench_distinguishes_unknown_city_and_empty_period(analysis_settings):
    app = create_app(analysis_settings, web_mode="demo")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1",
                          headers={"Origin": "http://127.0.0.1"}) as client,
    ):
        for city, start, end, expected in [
            ("Chongqin", "2026-01-01", "2026-04-01", "当前数据里没有「Chongqin」"),
            ("上海", "2026-01-01", "2026-04-01", "已生成"),
            ("上海", "2026-07-01", "2026-08-01", "所选时段没有已完成订单"),
        ]:
            body = {**BODY, "analysis": {"start": start, "end": end, "city": city}}
            submitted = (await client.post("/tasks", json=body)).raise_for_status().json()
            row = await terminal(client, submitted["task_id"])
            assert row["status"] == "COMPLETED", row["error"]
            assert expected in row["report"]["answer"]
            if city == "Chongqin":
                assert "analysis" not in row["report"]
            else:
                assert row["report"]["analysis"]["window"]["city"] == "Shanghai"
