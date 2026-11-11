import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
from conftest import completion
from test_memory import resources
from test_task_api import terminal

from agentserve.api.app import create_app
from agentserve.config import MemorySettings
from agentserve.domain.models import AgentError
from agentserve.memory.state import (
    StateChange,
    StateIntent,
    independent_request,
    normalize_date,
    resolve_state,
)
from agentserve.memory.state_rules import order_intent


def test_order_rules_resolve_literal_name_and_restore_original_without_model_choice():
    first, last = entry("turn:1", "Beijing"), entry("turn:2", "Shanghai")
    anchor = datetime(2026, 11, 4, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    restored = order_intent("恢复“营收甲”的原始第一个版本。", [first, last], anchor)
    assert restored.name == "营收甲" and restored.version_id == "turn:1"
    with pytest.raises(AgentError):
        order_intent("恢复“另一组不存在于目录的名称”。", [first, last], anchor)


def test_order_rules_change_only_current_fields_and_delete_price():
    anchor = datetime(2026, 11, 4, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    plan = order_intent(
        "建立命名分析“营收甲”：2026年2月Beijing的completed订单营收合计，单位分。", [], anchor
    )
    name, fields, _ = resolve_state(plan, [], None, "source")
    assert fields["date"]["value"] == ["2026-02-01", "2026-03-01"]
    prior = {"entry_id": "x", "state": {"name": name, "version_id": "turn:1", "fields": fields}}
    change = order_intent("继续“营收甲”，城市改为Shanghai，其他不变。", [prior], anchor)
    _, updated, _ = resolve_state(change, [prior], name, "current")
    assert updated["city"]["value"] == "Shanghai" and updated["date"] == fields["date"]
    short_change = order_intent("改成Shanghai", [prior], anchor)
    assert short_change.action == "refine"
    assert resolve_state(short_change, [prior], name, "current")[1] == updated
    deletion = order_intent("继续“营收甲”，取消价格门槛，其余不变。", [prior], anchor)
    assert [(c.field, c.operation) for c in deletion.changes] == [("price", "delete")]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("另做一个独立检查，不修改此前专题：所有商品有多少种？", True),
        ("An independent question: how many products are there?", True),
        ("不要另做独立统计，继续刚才的分析。", False),
        ("新建命名独立分析“营收甲”：订单总数。", False),
        ("This is not independent; use the previous analysis.", False),
    ],
)
def test_on_demand_requires_explicit_independence(question, expected):
    assert independent_request(question) is expected


def test_calendar_ranges_and_relative_anchor():
    anchor = datetime(2026, 11, 4, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    month = StateChange(field="date", operation="set", value="2026-12-01", evidence="2026年12月")
    normalize_date(month, anchor)
    assert month.operator == "range" and month.value == ["2026-12-01", "2027-01-01"]
    prior = StateChange(field="date", operation="set", value="incorrect", evidence="上个月")
    normalize_date(prior, anchor)
    assert prior.value == ["2026-10-01", "2026-11-01"]


def test_unrepresented_city_alias_keeps_verbatim_model_path():
    assert order_intent("建立分析“检查”：上海completed订单笔数，只返回数量。", [], 0) is None


async def test_unknown_schema_uses_recent_window_without_unverified_state(settings):
    settings.memory = MemorySettings(policy="state_on_demand")
    settings.databases["other"] = settings.databases["demo_orders"].model_copy()
    calls = []

    def model(request):
        body = json.loads(request.content)
        calls.append(body)
        assert "response_format" not in body
        return httpx.Response(
            200, json=completion(answer="2") if body["tool_choice"] == "none" else completion()
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
        sid = (await client.post("/sessions", json={"database_id": "other"})).json()["session_id"]
        tids = []
        for question in ["Count customers", "Show their count again"]:
            tid = (
                await client.post(
                    "/tasks", json={"database_id": "other", "session_id": sid, "question": question}
                )
            ).json()["task_id"]
            tids.append(tid)
            row = await terminal(client, tid)
            assert row["status"] == "COMPLETED" and row["report"]["memory_calls"] == 0
        context = (await client.get(f"/tasks/{tids[1]}/context")).json()["context"]
        assert context["plan"]["route"] == "recent_window" and tids[0] in context["source_ids"]
        assert (
            await app.state.tasks.repository.connection.execute_fetchall(
                "SELECT * FROM memory_entries"
            )
            == []
        )
    assert len(calls) == 4


def entry(version, city):
    return {
        "entry_id": version,
        "state": {
            "name": "营收甲",
            "version_id": version,
            "fields": {
                "city": {"operator": "eq", "value": city, "source_task_id": "original"},
                "status": {"operator": "eq", "value": "completed", "source_task_id": "original"},
            },
        },
    }


def test_exact_old_version_delete_and_field_provenance():
    old, new = entry("turn:1", "Beijing"), entry("turn:2", "Shanghai")
    plan = StateIntent(
        action="resume",
        name="营收甲",
        version_id="turn:1",
        changes=[
            {"field": "status", "operation": "delete"},
            {
                "field": "date",
                "operation": "set",
                "operator": "range",
                "value": ["2026-02-01", "2026-03-01"],
            },
        ],
    )
    name, fields, source = resolve_state(plan, [old, new], "营收甲", "current")
    assert name == "营收甲" and source is old and fields["city"]["value"] == "Beijing"
    assert "status" not in fields and fields["date"]["source_task_id"] == "current"
    assert old["state"]["fields"]["status"]["value"] == "completed"
    with pytest.raises(AgentError, match="本会话没有"):
        resolve_state(StateIntent(action="resume", name="未知"), [old, new], "营收甲", "current")


async def test_state_commit_restart_snapshot_and_no_writer_or_embeddings(settings):
    calls, resource_calls = [], []

    def model(request):
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            request = json.loads(body["messages"][-1]["content"])
            intent = (
                {"action": "resume", "name": "不存在", "changes": []}
                if "不存在" in request["question"]
                else {
                    "action": "new",
                    "name": "客户",
                    "changes": [
                        {
                            "field": "metric",
                            "operation": "set",
                            "operator": "describe",
                            "value": "客户数量",
                            "evidence": "客户数量",
                        }
                    ],
                }
            )
            result = completion(answer=json.dumps(intent))
        else:
            result = (
                completion(answer="模型错误数值123")
                if body["tool_choice"] == "none"
                else completion()
            )
        return httpx.Response(200, json=result)

    def cpu(request):
        resource_calls.append(request.url.path)
        return resources(request)

    settings.memory = MemorySettings(policy="state")
    settings.answer_style = "evidence_v21"

    def app_factory():
        return create_app(
            settings,
            model_transport=httpx.MockTransport(model),
            resource_transport=httpx.MockTransport(cpu),
        )

    app = app_factory()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        sid = (await client.post("/sessions", json={"database_id": "demo_orders"})).json()[
            "session_id"
        ]
        tid = (
            await client.post(
                "/tasks",
                json={"database_id": "demo_orders", "session_id": sid, "question": "统计客户数量"},
            )
        ).json()["task_id"]
        row = await terminal(client, tid)
        assert row["status"] == "COMPLETED", row
        assert row["report"]["memory_calls"] == 0 and row["report"]["model_calls"] == 2
        assert "123" not in row["report"]["answer"] and "[[2]]" in row["report"]["answer"]
        snapshot = (await client.get(f"/tasks/{tid}/context")).json()["context"]
        unknown = (
            await client.post(
                "/tasks",
                json={
                    "database_id": "demo_orders",
                    "session_id": sid,
                    "question": "恢复不存在的分析",
                },
            )
        ).json()["task_id"]
        missing = await terminal(client, unknown)
        assert (
            missing["error"]["code"] == "MEMORY_CONTEXT_REQUIRED"
            and missing["report"]["sql"] is None
        )
        assert len(calls) == 2  # Unresolved explicit references stop before model/SQL calls.
    app = app_factory()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        assert (await client.get(f"/tasks/{tid}/context")).json()["context"] == snapshot
        stored = await app.state.tasks.repository.connection.execute_fetchall(
            "SELECT entry_json FROM memory_entries"
        )
        state = json.loads(stored[0][0])["state"]
        assert state["fields"]["metric"]["source_task_id"] == tid
    assert "/embed" not in resource_calls
