"""Independent order-state and execution regressions for the V2.2 candidate."""

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from conftest import completion
from test_memory import resources
from test_task_api import terminal

from agentserve.agents.answer_policy import order_amount_note
from agentserve.agents.order_sql import compile_order_sql
from agentserve.api.app import create_app
from agentserve.config import MemorySettings
from agentserve.domain.models import QueryResult
from agentserve.memory.state import build_state, window_order_plan
from agentserve.memory.state_rules import complete_order_question, order_intent
from agentserve.workloads.memory_v21 import build as historical
from agentserve.workloads.memory_v22 import build, create_instances, validate_references

SCHEMA = Path(__file__).resolve().parents[1] / "data/processed/business_orders_v1/schema.sql"


def test_all_72_historical_latest_restores_keep_every_source_field():
    cases, labels = historical("acceptance")
    restored = 0
    for case in cases:
        name = case["turns"][0]["question"].split("“")[1].split("”")[0]
        fields = {
            "metric": {"value": "订单笔数", "operator": "describe", "source_task_id": "v1"},
            "outputs": {"value": ["metric"], "operator": "describe", "source_task_id": "v2"},
            "sort": {"value": ["+metric"], "operator": "describe", "source_task_id": "v2"},
        }
        entries = [
            {
                "entry_id": "v2",
                "state": {
                    "name": name,
                    "version_id": "turn:2",
                    "fields": fields,
                    "original_question": "订单笔数",
                    "modification_questions": [],
                    "complete_order_intent": True,
                },
            }
        ]
        for turn in case["turns"]:
            if not (labels[turn["id"]]["source_turn"] and "最新版本" in turn["question"]):
                continue
            q = turn["question"]
            intent = order_intent(q, entries, 0)
            state, target = build_state(intent, entries, None, q, 3, 0, complete_order_question(q))
            assert intent.changes == []
            assert target["entry_id"] == "v2" and state["fields"] == fields
            assert state["complete_order_intent"]
            restored += 1
    assert restored == 72


@pytest.mark.parametrize(
    "selector,version,complete",
    [
        ("修改后的最新版", None, True),
        ("的更新后的最新版本", None, True),
        ("当前版本", None, True),
        ("turn:1", "turn:1", True),
        ("的最初版本", "turn:1", True),
        ("修改后的最新版以外的版本", None, False),
    ],
)
def test_whole_version_selector_is_separate_from_the_patch(selector, version, complete):
    entries = [
        {
            "entry_id": f"e{i}",
            "state": {"name": "最初的营收表", "version_id": f"turn:{i}", "fields": {}},
        }
        for i in (1, 2)
    ]
    question = f"恢复“最初的营收表”{selector}，保留输出和排序。"
    assert complete_order_question(question) is complete
    intent = order_intent(question, entries, 0)
    if complete:
        assert intent.version_id == version and intent.changes == []
    else:
        assert intent is None


@pytest.mark.parametrize(
    "sql,columns,position",
    [
        (
            "SELECT SUM(o.quantity * o.unit_price_cents) AS metric FROM order_items o",
            ["metric"],
            "第1列",
        ),
        (
            'SELECT p.category, AVG(p."price_cents") AS "FROM" FROM products p',
            ["category", "FROM"],
            "第2列",
        ),
        ("SELECT COUNT(price_cents) AS metric FROM products", ["metric"], None),
        ("SELECT AVG(price_cents)/100.0 AS price_cents FROM products", ["price_cents"], None),
        ("SELECT 'price_cents' AS metric FROM products", ["metric"], None),
        (
            "SELECT price_cents FROM (SELECT COUNT(*) AS price_cents FROM products)",
            ["price_cents"],
            None,
        ),
    ],
)
def test_amount_units_follow_projected_expressions(sql, columns, position):
    note = order_amount_note(sql, QueryResult(columns=columns))
    assert note == (
        f"金额列（{position}）的单位为分；本回答原样展示查询返回值。" if position else ""
    )


def test_development_references_distinguish_wrong_queries_and_all_candidate_sql_matches(tmp_path):
    schema = SCHEMA.read_text(encoding="utf-8")
    cases, gold = build("development")
    databases = create_instances(tmp_path, schema, "development")
    assert all(validate_references(gold, databases)["coverage"].values())
    for case in cases:
        entries = []
        for index, turn in enumerate(case["turns"], 1):
            q = turn["question"]
            label = gold[turn["id"]]
            intent = order_intent(q, entries, 0)
            state, target = build_state(
                intent, entries, None, q, index, 0, complete_order_question(q)
            )
            assert state["complete_order_intent"], q
            assert {k: v["value"] for k, v in state["fields"].items()} == label["state"]
            assert {k: v["operator"] for k, v in state["fields"].items()} == label["operators"]
            assert state["removed_fields"] == label["removed"]
            assert (int(target["entry_id"]) if target else None) == label["source_turn"]
            sql = compile_order_sql(state)
            for database in databases.values():
                with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
                    assert db.execute(sql).fetchall() == db.execute(label["sql"]).fetchall(), q
            entries.append({"state": state, "entry_id": str(index)})


def test_reference_validation_rejects_restored_ne_changed_to_eq(tmp_path):
    _, gold = build("acceptance")
    validate_references(
        gold, create_instances(tmp_path, SCHEMA.read_text(encoding="utf-8"), "acceptance")
    )
    restore = next(
        label for label in gold.values() if label["family"] == 11 and label["bucket"] == "distant"
    )
    assert restore["operators"]["category"] == "ne"
    restore["operators"]["category"] = "eq"
    with pytest.raises(ValueError, match="Category reference disagrees"):
        validate_references(gold, {})


def test_window_cannot_recover_an_anchor_outside_two_visible_turns():
    q = "恢复“专题甲”原版，保留输出和排序。"
    visible = [
        {
            "question": "目录商品种数",
            "created_at": 0,
            "turn_no": n,
            "task_id": str(n),
            "status": "COMPLETED",
        }
        for n in (6, 7)
    ]
    assert window_order_plan(q, {"recent": visible, "accepted_at": 0, "cutoff_turn": 7}) is None


@pytest.mark.parametrize("answer_style", ["evidence_v221", "evidence_direct_v5"])
async def test_v22_api_compiles_known_query_and_preserves_unknown_qualifier(
    settings, tmp_path, answer_style
):
    database = settings.databases["demo_orders"]
    database.schema_path = SCHEMA
    database.path = create_instances(tmp_path, SCHEMA.read_text(encoding="utf-8"), "development")[
        "primary"
    ]
    settings.memory = MemorySettings(policy="state_on_demand")
    settings.sql_prompt_version = "evidence_v22"
    database.context = "All monetary values are integer cents."
    settings.answer_style = answer_style
    calls = []

    def model(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["tool_choice"] == "none":
            return httpx.Response(200, json=completion(answer="results"))
        return httpx.Response(
            200,
            json=completion(
                sql="SELECT AVG(price_cents) AS metric FROM products WHERE name LIKE '%钢笔%'"
            ),
        )

    app = create_app(
        settings,
        model_transport=httpx.MockTransport(model),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        sid = (await client.post("/sessions", json={"database_id": "demo_orders"})).json()[
            "session_id"
        ]
        for q in (
            "新建命名分析“甲”：目录商品种数，输出种数。",
            "新建命名分析“乙”：商品名称包含钢笔的目录商品当前平均标价，输出平均值。",
        ):
            tid = (
                await client.post(
                    "/tasks", json={"database_id": "demo_orders", "session_id": sid, "question": q}
                )
            ).json()["task_id"]
            row = await terminal(client, tid)
            assert row["status"] == "COMPLETED", row
        assert row["report"]["query_result"]["rows"] == [[None]]
        assert "金额列（第1列）的单位为分" in row["report"]["answer"]
    expected = [True, False, True] if answer_style == "evidence_v221" else [False]
    assert [body["tool_choice"] == "none" for body in calls] == expected
    query = next(body for body in calls if body["tool_choice"] != "none")
    assert "商品名称包含钢笔" in query["messages"][-1]["content"]
