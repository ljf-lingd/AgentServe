import asyncio
import json
import math
import sqlite3

import httpx
import pytest
from conftest import ROOT, completion
from test_memory import resources
from test_task_api import terminal

from agentserve.api.app import BUSINESS_DATABASE, create_app
from agentserve.api.web import COOKIE, create_account, web_settings as load_web_settings
from agentserve.application.order_workbench import calendar_bounds, calendar_mentions, display_calendar_period
from agentserve.config import MemorySettings, Settings

BASE = "http://127.0.0.1"
DATABASE = {"database_id": BUSINESS_DATABASE}
PASSWORD = "example-tests-only"


def test_calendar_summary_does_not_display_exclusive_end_as_part_of_month():
    assert display_calendar_period("2026年7月1日至8月1日，上海完成的订单数量为0。",
                                   ["2026-07-01", "2026-08-01"]) == "2026年7月，上海完成的订单数量为0。"


@pytest.mark.parametrize("question,year,bounds", [
    ("上海2026年1月到3月的成交额", 2028, ["2026-01-01", "2026-04-01"]),
    ("杭州26年一月至三月份的订单", 2026, ["2026-01-01", "2026-04-01"]),
    ("换成 1–3 月", 2025, ["2025-01-01", "2025-04-01"]),
    ("从2025年11月至2026年2月", 2027, ["2025-11-01", "2026-03-01"]),
    ("1月到2026年3月", 2025, ["2026-01-01", "2026-04-01"]),
    ("只看四月份", 2025, ["2025-04-01", "2025-05-01"]),
    ("2026年12月", 2025, ["2026-12-01", "2027-01-01"]),
    ("2028年2月29日", 2026, ["2028-02-29", "2028-03-01"]),
    ("杭州2026年总成交额", 2025, ["2026-01-01", "2027-01-01"]),
])
def test_calendar_accepts_varied_month_and_day_wording(question, year, bounds):
    assert calendar_bounds(calendar_mentions(question), year) == bounds


@pytest.mark.parametrize("question", ["2026年13月", "2026年3月到1月", "2025年2月29日"])
def test_calendar_rejects_invalid_ranges(question):
    with pytest.raises(ValueError):
        calendar_bounds(calendar_mentions(question), 2026)


@pytest.mark.parametrize("question", ["2026年第一季度", "截至3月", "一月和三月分别是多少", "取消日期限制"])
def test_calendar_does_not_collapse_noncontiguous_or_relative_scopes(question):
    assert not calendar_mentions(question)


def test_web_sam_uses_all_phases_with_session_sam_and_can_disable(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_CONFIG", str(ROOT / "configs/app.v5.json"))
    monkeypatch.delenv("AGENTSERVE_WEB_SAM_ALL_PHASES", raising=False)
    assert load_web_settings("demo").session_sam_all_phases
    assert load_web_settings("auth").session_sam_all_phases
    monkeypatch.setenv("AGENTSERVE_WEB_SAM_ALL_PHASES", "0")
    baseline = load_web_settings("demo")
    assert baseline.session_sam and not baseline.session_sam_all_phases


@pytest.fixture
def web_settings(settings):
    settings.databases = {BUSINESS_DATABASE: settings.databases["demo_orders"]}
    database = settings.databases[BUSINESS_DATABASE]
    schema = ("\nALTER TABLE customers ADD COLUMN city TEXT;"
              "\nCREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER);"
              "\nUPDATE customers SET city = 'Shanghai';")
    with sqlite3.connect(database.path) as connection:
        connection.executescript(schema)
    database.schema_path.write_text(database.schema_path.read_text() + schema, encoding="utf-8")
    settings.answer_style = "evidence_direct_v5"
    settings.session_sam = True
    settings.memory = MemorySettings()
    settings.max_model_retries = 0
    return settings


def client(app, *, origin=BASE):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url=BASE, headers={"Origin": origin}
    )


async def make_session(api):
    return (await api.post("/sessions", json=DATABASE)).raise_for_status().json()["session_id"]


async def ask(api, sid, question="How many customers?", *, key=None, explain=False):
    body = {**DATABASE, "session_id": sid, "question": question}
    if explain:
        body["answer_mode"] = "explain"
    return await api.post("/tasks", json=body, headers={"Idempotency-Key": key} if key else None)


def app_for(settings, model, mode):
    return create_app(
        settings, web_mode=mode, model_transport=httpx.MockTransport(model),
        resource_transport=httpx.MockTransport(resources),
    )


async def test_city_queries_bypass_memory_model_and_preserve_order_intent(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v5.json")
    settings.state_database = tmp_path / "state.sqlite"
    app = app_for(settings, lambda _: pytest.fail("Short order query called the model"), "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        catalog = (await api.get("/catalog/cities")).raise_for_status().json()
        assert [item["value"] for item in catalog] == [
            "Beijing", "Chengdu", "Hangzhou", "Shanghai", "Shenzhen", "Wuhan"
        ]
        sid = await make_session(api)
        async def ask_city(text):
            result = (await ask(api, sid, text)).raise_for_status().json()
            task = await terminal(api, result["task_id"])
            assert task["status"] == "COMPLETED", task["error"]
            return task["report"]

        assert "你是想查北京" in (await ask_city("有Beijin的订单吗"))["answer"]
        assert "武汉" in (await ask_city("有哪些城市的订单"))["answer"]
        assert (await ask_city("那北京呢"))["answer"].startswith("有，北京")
        assert "173 笔" in (await ask_city("北京有多少订单"))["answer"]
        assert "164 笔" in (await ask_city("那成都呢"))["answer"]
        capabilities = await ask_city("？为什么不能提示？有哪些城市可以查询 有什么订单可以查看修改？")
        assert "武汉" in capabilities["answer"] and "订单数据是只读的" in capabilities["answer"]
        await ask_city("深圳的订购单数呢 有吗")
        months = (await ask_city("从哪月到哪月"))["answer"]
        assert "上一轮未限定日期" in months and "深圳（Shenzhen）的订单实际覆盖 2026-01 至 2026-06" in months
        assert "164 笔" in (await ask_city("那深圳呢"))["answer"]
        await ask_city("2026年1月Shenzhen的订单笔数。")
        months = (await ask_city("上一轮从几月到几月？"))["answer"]
        assert "2026-01-01 至 2026-02-01（不含结束日）" in months and "实际覆盖 2026-01 至 2026-01" in months

        sid = await make_session(api)
        previous = (await ask(api, sid, "深圳的订购单数呢 有吗")).raise_for_status().json()["task_id"]
        await terminal(api, previous)
        legacy_sql = (
            "SELECT COUNT(*) AS order_count FROM orders WHERE customer_id IN "
            "(SELECT customer_id FROM customers WHERE city = 'Shenzhen')"
        )
        async with app.state.tasks.repository.transaction() as conn:
            await conn.execute("UPDATE tasks SET report_json=json_set(report_json,'$.sql',?) WHERE task_id=?",
                               (legacy_sql, previous))
        assert "实际覆盖 2026-01 至 2026-06" in (await ask_city("从哪月到哪月"))["answer"]
        assert "实际覆盖 2026-01 至 2026-06" in (await ask_city("月份呢"))["answer"]

        march = (await ask_city("2026年3月有多少订单深圳和上海"))["answer"]
        assert "2026年3月" in march and "深圳（Shenzhen） 29 笔" in march and "上海（Shanghai） 35 笔" in march
        assert "实际覆盖 2026-03 至 2026-03" in (await ask_city("月份呢"))["answer"]
        cities = (await ask_city("总共有多少个城市有订单详情？"))["answer"]
        assert "6 个城市有订单" in cities and "武汉" in cities
        empty = (await ask_city("2026年7月深圳和上海有多少订单"))["answer"]
        assert "2026年7月深圳（Shenzhen）和上海（Shanghai）的订单笔数均为 0" in empty


async def test_model_wording_for_settled_revenue_checks_sql_and_amount(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v5.json")
    settings.state_database = tmp_path / "state.sqlite"
    revenue_sql = (
        "SELECT SUM(i.quantity * i.unit_price_cents) AS revenue_cents "
        "FROM orders o JOIN customers c USING(customer_id) JOIN order_items i USING(order_id) "
    )
    answers = iter(["上海的成交营业额为30,624.50元。", "上海的成交营业额为3062450元。"])

    def model(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        if "response_format" in payload:
            return httpx.Response(200, json=completion(answer=json.dumps({
                "action": "new", "changes": [
                    {"field": "city", "operation": "set", "value": "上海", "evidence": "上海"},
                ],
            }, ensure_ascii=False)))
        if payload["tool_choice"] == "none":
            assert "30,624.50 元" in messages[-1]["content"]
            return httpx.Response(200, json=completion(answer=next(answers)))
        if messages[-1]["role"] == "tool":
            assert "exact stored value 'Shanghai'" in messages[-1]["content"]
            assert "orders.status = 'completed'" in messages[-1]["content"]
            predicate = "WHERE c.city = 'Shanghai' AND o.status = 'completed'"
        else:
            predicate = "WHERE c.city = '上海'"
        return httpx.Response(200, json=completion(revenue_sql + predicate))

    app = app_for(settings, model, "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        for source in ("model", "tool_values"):
            sid = await make_session(api)
            task = (await ask(api, sid, "上海的成交营业额是多少")).raise_for_status().json()
            record = await terminal(api, task["task_id"])
            assert record["status"] == "COMPLETED", record["error"]
            assert record["report"]["query_result"]["rows"] == [[3062450]]
            assert record["report"]["answer_source"] == source
            if source == "model":
                assert record["report"]["answer"] == "上海的成交营业额为30,624.50元。"


async def test_calendar_followups_recover_user_year_and_empty_amount(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v5.json")
    settings.state_database = tmp_path / "state.sqlite"
    cases = [
        ("杭州25年六月的交易额是多少", "Hangzhou", "2025-06"),
        ("那7月呢", "Hangzhou", "2025-07"),
        ("上海呢", "Shanghai", "2025-07"),
        ("2026年7月", "Shanghai", "2026-07"),
    ]

    def model(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        if "response_format" in payload:
            changes = []
            if question in (cases[0][0], cases[2][0]):
                label = "杭州" if city == "Hangzhou" else "上海"
                changes.append({"field": "city", "operation": "set", "value": label, "evidence": label})
            if question in (cases[0][0], cases[1][0], cases[3][0]):
                evidence = {cases[0][0]: "25年六月", cases[1][0]: "7月", cases[3][0]: "2026年7月"}[question]
                changes.append({"field": "date", "operation": "set", "value": "2023-07", "evidence": evidence})
            return httpx.Response(200, json=completion(answer=json.dumps({
                "action": "new" if question in (cases[0][0], cases[3][0]) else "refine", "changes": changes,
            }, ensure_ascii=False)))
        if payload["tool_choice"] == "none":
            return httpx.Response(200, json=completion(answer="所查交易额为NULL元。"))
        repair = messages[-1]["role"] == "tool"
        if repair:
            assert "verified period" in messages[-1]["content"]
        chosen_month = "2023-07" if question == cases[1][0] and not repair else month
        sql = ("SELECT SUM(i.quantity * i.unit_price_cents) AS revenue_cents "
               "FROM orders o JOIN customers c USING(customer_id) JOIN order_items i USING(order_id) "
               f"WHERE c.city='{city}' AND o.status='completed' AND strftime('%Y-%m',o.order_date)='{chosen_month}'")
        return httpx.Response(200, json=completion(sql))

    app = app_for(settings, model, "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        sid = await make_session(api)
        for turn, (question, city, month) in enumerate(cases, 1):
            task = (await ask(api, sid, question)).raise_for_status().json()
            record = await terminal(api, task["task_id"])
            assert record["status"] == "COMPLETED", record["error"]
            repo = app.state.tasks.repository
            report = (await repo.get(task["task_id"]))["report"]
            assert f"'{city}'" in report["sql"] and f"'{month}'" in report["sql"]
            assert report["answer"] == "所查范围内没有可汇总的金额。"
            if turn == 2:
                assert report["sql_repairs"] == 1
                async with repo.transaction() as conn:
                    await conn.execute("UPDATE memory_entries SET entry_json=json_set(entry_json, "
                                       "'$.state.fields.date.value', json('[\"2023-07-01\",\"2023-08-01\"]')) "
                                       "WHERE session_id=? AND turn_no=2", (sid,))


async def test_inclusive_revenue_month_range_survives_city_and_date_followups(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v5.json")
    settings.state_database = tmp_path / "state.sqlite"
    cases = [
        ("上海2026年1月到3月的总成交额是多少", "Shanghai", "上海", "1月到3月", 1720075),
        ("深圳呢", "Shenzhen", "深圳", None, 1650200),
        ("是1月到3月", "Shenzhen", None, "1月到3月", 1650200),
        ("为什么月份增加了成交额变少了？", "Shenzhen", None, "月份增加了", 1650200),
    ]

    def model(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        if "response_format" in payload:
            assert question != cases[3][0], "An amount recheck should retain the saved scope"
            changes = []
            if city_label:
                changes.append({"field": "city", "operation": "set", "value": city_label, "evidence": city_label})
            if date_quote:
                changes.append({"field": "date", "operation": "set", "operator": "range",
                                "value": ["2026-01-01", "2026-03-01"], "evidence": date_quote})
            return httpx.Response(200, json=completion(answer=json.dumps({
                "action": "new" if city == "Shanghai" else "refine", "changes": changes,
            }, ensure_ascii=False)))
        if payload["tool_choice"] == "none":
            amount = 1 if question == cases[3][0] else cents / 100
            return httpx.Response(200, json=completion(answer=f"{city}2026年1月到2月的成交额是{amount:,.2f} 元。"))
        repair = messages[-1]["role"] == "tool"
        if repair:
            assert "verified period" in messages[-1]["content"]
        if question == cases[2][0] and not repair:
            dates = "o.order_date >= '2026-01-01' AND o.order_date < '2026-03-01'"
        elif question == cases[1][0]:
            dates = "strftime('%Y-%m', o.order_date) BETWEEN '2026-01' AND '2026-03'"
        else:
            dates = "o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01'"
        sql = ("SELECT SUM(i.quantity * i.unit_price_cents) AS revenue_cents "
               "FROM orders o JOIN customers c USING(customer_id) JOIN order_items i USING(order_id) "
               f"WHERE c.city='{city}' AND o.status='completed' AND {dates}")
        return httpx.Response(200, json=completion(sql))

    app = app_for(settings, model, "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        sid = await make_session(api)
        for question, city, city_label, date_quote, cents in cases:
            task_id = (await ask(api, sid, question)).raise_for_status().json()["task_id"]
            result = await terminal(api, task_id)
            assert result["status"] == "COMPLETED", result["error"]
            report = (await app.state.tasks.repository.get(task_id))["report"]
            assert report["memory_context"]["plan"]["state"]["fields"]["date"]["value"] == ["2026-01-01", "2026-04-01"]
            assert report["query_result"]["rows"] == [[cents]]
            assert "2026年1月至3月" in report["answer"]
            assert f"{cents / 100:,.2f} 元" in report["answer"]
            if question == cases[2][0]:
                assert report["sql_repairs"] == 1


async def test_cross_year_followup_uses_last_explicit_year_and_handles_empty_range(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v5.json")
    settings.state_database = tmp_path / "state.sqlite"
    cases = [
        ("深圳2025年12月至2026年2月的成交额是多少", "2025-12-01", "2026-03-01", 1098400),
        ("只看3月", "2026-03-01", "2026-04-01", 551800),
        ("改看7–8月", "2026-07-01", "2026-09-01", None),
    ]

    def model(request):
        payload = json.loads(request.content)
        if "response_format" in payload:
            assert question == cases[0][0], "Date-only edits should reuse the saved city and metric"
            return httpx.Response(200, json=completion(answer=json.dumps({"action": "new", "changes": [
                {"field": "city", "operation": "set", "value": "深圳", "evidence": "深圳"},
                {"field": "metric", "operation": "set", "value": "成交额", "evidence": "成交额"},
            ]}, ensure_ascii=False)))
        if payload["tool_choice"] == "none":
            answer = "成交额为NULL元。" if cents is None else f"核对成交额为 {cents / 100:,.2f} 元。"
            return httpx.Response(200, json=completion(answer=answer))
        sql = ("SELECT SUM(i.quantity * i.unit_price_cents) AS revenue_cents "
               "FROM orders o JOIN customers c USING(customer_id) JOIN order_items i USING(order_id) "
               "WHERE c.city='Shenzhen' AND o.status='completed' "
               f"AND o.order_date >= '{start}' AND o.order_date < '{end}'")
        return httpx.Response(200, json=completion(sql))

    app = app_for(settings, model, "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        sid = await make_session(api)
        for question, start, end, cents in cases:
            task_id = (await ask(api, sid, question)).raise_for_status().json()["task_id"]
            result = await terminal(api, task_id)
            assert result["status"] == "COMPLETED", result["error"]
            report = (await app.state.tasks.repository.get(task_id))["report"]
            assert report["memory_context"]["plan"]["state"]["workbench_period"] == [start, end]
            assert report["query_result"]["rows"] == [[cents]]
            if cents is None:
                assert report["answer"] == "所查范围内没有可汇总的金额。"


async def test_demo_history_metrics_and_failure(web_settings):
    async def model(request):
        payload = json.loads(request.content)
        if "Simulate model error" in str(payload["messages"]):
            raise httpx.ConnectError("internal-only-detail", request=request)
        if payload["tool_choice"] != "none":
            assert "上海 -> Shanghai" in payload["messages"][0]["content"]
            return httpx.Response(200, json=completion())
        return httpx.Response(200, json={
            **completion(answer="Two customers."), "id": "explain-model",
            "metrics": {
                "time_to_first_token_ms": 5, "tokens_per_second": 10, "mean_itl_ms": None,
                "speculative_decoding": {"num_draft_tokens": 0, "num_accepted_draft_tokens": 0},
                "unrelated": "internal-only-detail",
            },
        })

    app = app_for(web_settings, model, "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        assert (await api.get("/auth/me")).json() == {"mode": "demo", "user": None}
        sid = await make_session(api)
        direct = (await ask(api, sid, key="one")).json()["task_id"]
        assert (await ask(api, sid, key="one")).json() == {"task_id": direct, "created": False}
        explained = (await ask(api, sid, explain=True)).json()["task_id"]
        failure = (await ask(api, sid, "Simulate model error")).json()["task_id"]
        direct_report = (await terminal(api, direct))["report"]
        assert direct_report["query_result"]["rows"] == [[2]]
        assert direct_report["answer_source"] == "model"
        assert direct_report["answer"] == "Two customers."
        assert (await terminal(api, explained))["report"]["answer"] == "Two customers."
        failed = await terminal(api, failure)
        assert failed["status"] == "FAILED"
        assert failed["error"]["code"] == "MODEL_CONNECTION_ERROR"
        assert "internal-only-detail" not in str(failed)
        assert (await api.get("/sessions")).json()[0]["title"] == "How many customers?"
        history = (await api.get(f"/sessions/{sid}/history?limit=2")).json()
        assert [turn["task_id"] for turn in history["turns"]] == [direct, explained]
        assert history["next_after_turn"] == 2
        events = (await api.get(f"/tasks/{failure}/events?limit=1")).json()
        assert set(events["events"][0]) == {"event_id", "type", "wall_time"}
        assert (await api.get(f"/tasks/{explained}/steps")).status_code == 404
        overview = (await api.get("/admin/overview")).json()
        assert overview["counts"]["COMPLETED"] == 2
        assert overview["counts"]["FAILED"] == 1
        assert overview["errors"][0]["task_id"] == failure
        assert "internal-only-detail" not in str(overview)
        entry = next(row for row in overview["requests"] if row["request_id"] == "explain-model")
        assert entry["metrics"] == {
            "time_to_first_token_ms": 5, "tokens_per_second": 10,
            "num_draft_tokens": 0, "num_accepted_draft_tokens": 0,
        }
        assert any(row["metrics"] is None for row in overview["requests"])
        direct_jct = next(row for row in overview["jct"] if row["mode"] == "sql")
        row = await terminal(api, direct)
        assert direct_jct["n"] == 1
        assert math.isclose(direct_jct["p95_seconds"], row["ended_at"] - row["created_at"])
        assert (await api.post("/admin/overview/reset")).status_code == 204
        reset = (await api.get("/admin/overview")).json()
        assert reset["cleared_at"] and reset["counts"]["COMPLETED"] == 0 and reset["requests"] == []
        assert (await api.get(f"/tasks/{direct}")).status_code == 200

    reopened = app_for(web_settings, model, "demo")
    async with reopened.router.lifespan_context(reopened), client(reopened) as api:
        assert (await api.get(f"/tasks/{explained}")).json()["report"]["answer"] == "Two customers."


async def test_structured_preview_requires_save_and_history_is_grouped(tmp_path):
    settings = Settings.load(ROOT / "configs/app.v5.json")
    settings.state_database = tmp_path / "state.sqlite"
    app = app_for(settings, lambda _: pytest.fail("Structured previews do not use the model"), "demo")
    window = {"start": "2026-04-01", "end": "2026-05-01", "city": "杭州"}
    async with app.router.lifespan_context(app), client(app) as api:
        ids = {}
        for mode, body in (
            ("analysis", window), ("metric", {**window, "name": "settled_revenue"}),
        ):
            response = await api.post("/tasks", json={**DATABASE, "mode": mode, mode: body, "save_to_history": False})
            task_id = ids[mode] = response.raise_for_status().json()["task_id"]
            result = await terminal(api, task_id)
            assert result["status"] == "COMPLETED", result["error"]
            assert result["in_history"] is False
            assert (await api.get("/tasks", params={"mode": mode})).json() == []
            assert (await api.post(f"/tasks/{task_id}/save")).status_code == 204
            assert (await api.post(f"/tasks/{task_id}/save")).status_code == 204
        for mode, task_id in ids.items():
            items = (await api.get("/tasks", params={"mode": mode})).json()
            assert [row["task_id"] for row in items] == [task_id]
            assert "2026-04" in items[0]["title"]
            assert (await api.get(f"/tasks/{task_id}")).json()["in_history"] is True
        assert (await api.post("/tasks/batch-delete", json={"ids": list(ids.values())})).status_code == 204
        assert (await api.get("/tasks")).json() == []
        for task_id in ids.values():
            assert (await api.get(f"/tasks/{task_id}")).status_code == 404


async def test_login_ownership_and_cancellation(web_settings):
    async def model(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=completion())

    for name, role in [("alice", "user"), ("bob", "user"), ("operator", "admin")]:
        await create_account(name, PASSWORD, role, web_settings.state_database)

    app = app_for(web_settings, model, "auth")
    async with (
        app.router.lifespan_context(app), client(app) as alice, client(app) as bob,
        client(app) as admin, client(app) as anonymous, client(app, origin="https://other.test") as other,
    ):
        assert (await anonymous.get("/sessions")).status_code == 401
        assert (await anonymous.post("/sessions/batch-delete", json={"ids": ["unknown"]})).status_code == 401
        assert (await anonymous.get("/admin/overview")).status_code == 401
        assert (await other.post("/auth/login", json={"username": "alice", "password": PASSWORD})).status_code == 403
        for api, username, role in [(alice, "alice", "user"), (bob, "bob", "user"), (admin, "operator", "admin")]:
            login = await api.post("/auth/login", json={"username": username, "password": PASSWORD})
            assert login.json() == {"username": username, "role": role}
            assert "HttpOnly" in login.headers["set-cookie"]
            assert (await api.get("/auth/me")).json()["user"]["username"] == username
        sid = await make_session(alice)
        private = (await ask(alice, sid, key="shared-key")).json()["task_id"]
        assert (await ask(alice, sid, key="shared-key")).json()["task_id"] == private
        sid_b = await make_session(bob)
        bob_task = (await ask(bob, sid_b, key="shared-key")).json()["task_id"]
        assert bob_task != private
        empty = await make_session(alice)
        assert (await alice.post("/sessions/batch-delete", json={"ids": [empty, sid_b]})).status_code == 404
        assert (await alice.post("/sessions/batch-delete", json={"ids": [empty, sid]})).status_code == 409
        assert {row["session_id"] for row in (await alice.get("/sessions")).json()} == {sid, empty}
        assert (await alice.delete(f"/sessions/{empty}")).status_code == 204
        assert (await ask(bob, sid)).status_code == 404
        assert {row["session_id"] for row in (await bob.get("/sessions")).json()} == {sid_b}
        for api in (bob, admin):
            for url in (f"/sessions/{sid}/history", f"/tasks/{private}",
                        f"/tasks/{private}/events", f"/tasks/{private}/artifacts/report.md"):
                assert (await api.get(url)).status_code == 404
            assert (await api.post(f"/tasks/{private}/cancel")).status_code == 404
            assert (await api.patch(f"/sessions/{sid}", json={"title": "他人的会话"})).status_code == 404
            assert (await api.delete(f"/sessions/{sid}")).status_code == 404
        assert (await alice.patch(f"/sessions/{sid}", json={"title": "  查数记录  "})).status_code == 204
        assert (await alice.get("/sessions")).json()[0]["title"] == "查数记录"
        assert (await alice.delete(f"/sessions/{sid}")).status_code == 409
        assert (await alice.get("/admin/overview")).status_code == 403
        assert (await alice.post("/admin/overview/reset")).status_code == 403
        assert (await admin.get("/admin/overview")).status_code == 200
        assert (await alice.post(f"/tasks/{private}/cancel")).json()["status"] == "CANCELLED"
        assert (await terminal(alice, private))["status"] == "CANCELLED"
        assert (await alice.delete(f"/sessions/{sid}")).status_code == 204
        assert (await alice.get("/sessions")).json() == []
        assert (await alice.get(f"/tasks/{private}")).status_code == 404
        assert (await alice.get(f"/sessions/{sid}/history")).status_code == 404
        assert (await ask(alice, sid)).status_code == 404
        assert (await ask(alice, sid, key="shared-key")).status_code == 404

        token = alice.cookies.get(COOKIE)
        assert (await alice.post("/auth/logout")).status_code == 204
        assert (await anonymous.get("/sessions", headers={"Cookie": f"{COOKIE}={token}"})).status_code == 401
        async with app.state.tasks.repository.transaction() as conn:
            await conn.execute("UPDATE web_logins SET expires_at=0")
        assert (await bob.get("/sessions")).status_code == 401
        assert (await bob.get("/auth/me")).json()["user"] is None


async def test_configured_workbench_datasets_keep_history_and_sessions_separate(web_settings):
    from agentserve.api.app import BUSINESS_CATALOG
    from agentserve.config import WorkbenchDataset

    def model(request):
        payload = json.loads(request.content)
        if payload.get("tool_choice") == "none":
            return httpx.Response(200, json=completion(answer="查询完成。"))
        return httpx.Response(200, json=completion(sql="SELECT COUNT(*) FROM customers"))

    legacy = app_for(web_settings, model, "demo")
    async with legacy.router.lifespan_context(legacy), client(legacy) as api:
        original = await make_session(api)

    alternate = web_settings.databases[BUSINESS_DATABASE].model_copy(deep=True)
    second_path = alternate.path.with_name("second-dataset.sqlite")
    with sqlite3.connect(alternate.path) as source, sqlite3.connect(second_path) as target:
        source.backup(target)
        target.execute("INSERT INTO customers(customer_id,name,city) VALUES (3,'Carol','Wuhan')")
    source.close()
    target.close()
    alternate.path = second_path
    alternate.workbench = WorkbenchDataset(
        label="Second dataset", first_date="2010-01-01", last_date="2011-12-09",
        currency="GBP", modes=("sql", "knowledge"), data_description="Retail",
        rule_description="Rules", query_examples=("Count customers",), rule_examples=(),
    )
    web_settings.databases[BUSINESS_DATABASE].workbench = BUSINESS_CATALOG
    web_settings.databases["alternate"] = alternate
    app = app_for(web_settings, model, "demo")
    async with app.router.lifespan_context(app), client(app) as api:
        catalog = (await api.get("/catalog/datasets")).raise_for_status().json()
        assert {item["database_id"] for item in catalog} == {BUSINESS_DATABASE, "alternate"}
        resumed = (await ask(api, original)).raise_for_status().json()
        original_task = await terminal(api, resumed["task_id"])
        assert original_task["status"] == "COMPLETED", original_task["error"]
        retail = (await api.post("/sessions", json={"database_id": "alternate"})).raise_for_status().json()
        original_history = (await api.get("/sessions")).raise_for_status().json()
        other_history = (await api.get("/sessions?database_id=alternate")).raise_for_status().json()
        assert {item["session_id"] for item in original_history} == {original}
        assert {item["session_id"] for item in other_history} == {retail["session_id"]}
        other_query = (await api.post("/tasks", json={
            "database_id": "alternate", "session_id": retail["session_id"], "question": "How many customers?",
        })).raise_for_status().json()
        other = await terminal(api, other_query["task_id"])
        assert other["report"]["query_result"]["rows"] == [[3]]
        mismatch = await api.post("/tasks", json={
            "database_id": "alternate", "session_id": original, "question": "count",
        })
        assert mismatch.json()["error"]["code"] == "SESSION_DATABASE_MISMATCH"
        missing = await api.post("/tasks", json={
            "database_id": "alternate", "mode": "analysis",
            "analysis": {"start": "2010-01-01", "end": "2010-02-01"},
        })
        assert missing.status_code == 422


        for database, session in ((BUSINESS_DATABASE, original), ("alternate", retail["session_id"])):
            request = (await api.post("/tasks", json={
                "database_id": database, "session_id": session, "question": "2030 年 1 月的销售额是多少？",
            })).raise_for_status().json()
            task = await terminal(api, request["task_id"])
            assert task["status"] == "COMPLETED"
            assert "超出数据覆盖范围" in task["report"]["answer"]
