import asyncio
import json
from pathlib import Path

import httpx
import pytest
from conftest import completion
from test_memory import resources
from test_recovery import ProcessCrash
from test_task_api import terminal

from agentserve.api.app import create_app
from agentserve.config import KnowledgeSettings, MemorySettings
from agentserve.knowledge.snapshot import KnowledgeSnapshot, assemble, build_snapshot

MANIFEST = Path(__file__).resolve().parents[1] / "data/synthetic/knowledge_v3/manifest.json"
BODY = {
    "mode": "knowledge",
    "database_id": "business_orders_v1",
    "question": "成交营收使用哪种价格？",
}
SCENARIOS = MANIFEST.with_name("scenarios.json")
PRODUCT = MANIFEST.parents[2] / "knowledge_v31/manifest.json"


@pytest.fixture
async def knowledge_settings(settings, tmp_path):
    settings.databases["business_orders_v1"] = settings.databases["demo_orders"]
    settings.memory = MemorySettings()
    path = tmp_path / "snapshot.json"
    await build_snapshot(MANIFEST, path)
    settings.knowledge = KnowledgeSettings(snapshot_path=path, purpose="test")
    return settings


def model(calls, valid=True):
    def handle(request):
        body = json.loads(request.content)
        assert "tools" not in body and "response_format" in body
        calls.append(body)
        payload = json.loads(body["messages"][-1]["content"])
        if "evidence" not in payload:
            assert payload == {"question": BODY["question"]}
            return completion_response({"questions": ["成交营收使用的价格是什么？"]})
        selected = next(
            key
            for source in payload["evidence"]
            for key, sentence in source["sentences"].items()
            if "历史成交单价字段为" in sentence
        )
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["q1"]
        assert selected in schema["properties"]["q1"]["items"]["enum"]
        return completion_response({"q1": [selected if valid else "s99_99"]})

    return httpx.MockTransport(handle)


def completion_response(value):
    return httpx.Response(200, json=completion(answer=json.dumps(value)))


async def test_property_answer_requires_every_fragment_and_retrieved_witness(
    knowledge_settings, tmp_path
):
    snapshot = tmp_path / "properties.json"
    await build_snapshot(PRODUCT, snapshot)
    knowledge_settings.knowledge = KnowledgeSettings(snapshot_path=snapshot, strategy="bm25")
    replies = iter(
        (
            {"q1": ["unknown"], "q2": ["unknown"]},
            {"q1": ["city_column"], "q2": ["unknown"]},
        )
    )

    def handle(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        assert "tools" not in body and set(payload) == {
            "question",
            "requests",
            "properties",
            "evidence",
        }
        assert all(
            span in payload["evidence"]
            for x in payload["properties"].values()
            for span in x["support_ids"]
        )
        return completion_response(next(replies))

    app = create_app(
        knowledge_settings,
        model_transport=httpx.MockTransport(handle),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        for question, outcome in (
            ("按城市汇总用哪个字段？过滤城市值时是否区分大小写？", "answered"),
            ("城市分组的字段和排序列各自是什么？", "insufficient_evidence"),
        ):
            task_id = (await client.post("/tasks", json={**BODY, "question": question})).json()[
                "task_id"
            ]
            task = await terminal(client, task_id)
            report = task["report"]
            result = report["knowledge"]
            assert report["sql"] is None and report["model_calls"] == 1
            assert result["outcome"] == outcome
            if outcome == "answered":
                assert [c["chunk_id"] for c in result["citations"]] == [
                    "customer@1#1",
                    "product@1#1",
                ]
            else:
                assert not result["citations"] and not result["coverage"][1]["supports"]


def test_fixture_registrar_cannot_supply_a_value():
    from agentserve.knowledge.facts import fixed_field, possible_fields, resolve

    snapshot = assemble(SCENARIOS)
    question = "演练中的票据归档标签是什么"
    options = possible_fields(question, snapshot["properties"], fixture=True)
    key = fixed_field(question, options, fixture=True)
    assert key is not None
    by_id = {c["id"]: c for c in snapshot["chunks"]}
    for condition, expected in (
        ("a", "answered"),
        ("b", "answered"),
        ("topical", "insufficient_evidence"),
    ):
        outcome, _, checks = resolve([(question, [key])], [by_id[f"dev-invoice-{condition}"]])
        assert outcome == expected
        assert bool(checks[0]["supports"]) == (condition != "topical")


@pytest.mark.parametrize(
    ("clause", "question"),
    [
        ("relations", "订单明细的关联键可以为空吗？"),
        ("product", "储位编码区分大小写吗？"),
        ("dates", "出库交接日期记录在哪一列？"),
    ],
)
async def test_unregistered_relation_cannot_bind_available_property(
    knowledge_settings, tmp_path, clause, question
):
    snapshot = tmp_path / "properties.json"
    await build_snapshot(PRODUCT, snapshot)
    knowledge_settings.knowledge = KnowledgeSettings(snapshot_path=snapshot, strategy="bm25")

    def handle(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        assert payload["requests"]["q1"]["options"] == []
        return completion_response({"q1": ["unknown"]})

    app = create_app(
        knowledge_settings,
        model_transport=httpx.MockTransport(handle),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        result = await client.post(
            "/tasks",
            json={**BODY, "clause_id": clause, "question": question},
        )
        task = await terminal(client, result.json()["task_id"])
        report = task["report"]
        assert report["model_calls"] == 1 and report["sql"] is None
        assert report["knowledge"]["outcome"] == "insufficient_evidence"
        assert not report["knowledge"]["citations"]
        assert not report["knowledge"]["coverage"][0]["supports"]


@pytest.mark.parametrize("valid", [True, False])
async def test_knowledge_citations_validated_and_no_sql(knowledge_settings, valid):
    calls = []
    app = create_app(
        knowledge_settings,
        model_transport=model(calls, valid),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        headers = {"Idempotency-Key": "same"}
        response = await client.post("/tasks", json=BODY, headers=headers)
        tid = response.json()["task_id"]
        assert (await client.post("/tasks", json=BODY, headers=headers)).json()["task_id"] == tid
        task = await terminal(client, tid)
        report = task["report"]
        assert report["sql"] is None and len(calls) == 2
        if valid:
            quote = report["knowledge"]["citations"][0]["quote"]
            assert "历史成交单价" in quote and "金额单位为分" in quote
            assert report["knowledge"]["as_of"] == "2026-11-05"
            assert "orders@1#1" in report["knowledge"]["candidate_ids"]
        else:
            assert task["error"]["code"] == "KNOWLEDGE_OUTPUT_INVALID"
            assert report["knowledge"]["citations"] == []
        assert (await client.get(f"/tasks/{tid}/context")).json()["context"] is None


@pytest.mark.parametrize(
    ("crash_stage", "crash_kind"),
    [("before_execute", "knowledge"), ("after_commit", "knowledge"), ("after_commit", "model")],
)
async def test_knowledge_restart_reuses_only_committed_context(
    knowledge_settings, crash_stage, crash_kind
):
    calls = []
    reached = asyncio.Event()

    async def hook(stage, task_id, step_no, kind):
        if stage == crash_stage and kind == crash_kind:
            reached.set()
            raise ProcessCrash()

    for step_hook in (hook, None):
        app = create_app(
            knowledge_settings,
            model_transport=model(calls),
            resource_transport=httpx.MockTransport(resources),
            step_hook=step_hook,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            if step_hook:
                tid = (await client.post("/tasks", json=BODY)).json()["task_id"]
                await asyncio.wait_for(reached.wait(), 3)
            else:
                assert (await terminal(client, tid))["status"] == "COMPLETED"
                events = (await client.get(f"/tasks/{tid}/events")).json()["events"]
                assert sum(e["type"] == "KNOWLEDGE_CONTEXT_COMMITTED" for e in events) == 1
                assert any(e["type"] == "KNOWLEDGE_CONTEXT_REUSED" for e in events) == (
                    crash_stage == "after_commit"
                )
    assert len(calls) == 2


async def test_release_snapshot_cannot_load_test_corpus(tmp_path):
    path = tmp_path / "fixture.json"
    await build_snapshot(SCENARIOS, path)
    with pytest.raises(ValueError, match="purpose"):
        KnowledgeSnapshot(path, "bm25", None)
    snapshot = KnowledgeSnapshot(path, "bm25", None, purpose="test")
    assert list(snapshot.active("business_orders_v1", "2026-11-04", "formal-hold")) == [
        "formal-hold@1#1"
    ]
    assert list(snapshot.active("business_orders_v1", "2026-11-05", "formal-hold")) == [
        "formal-hold@2#1"
    ]
    active = snapshot.active("business_orders_v1", "2030-01-01")
    assert "formal-hold@2#1" in active
    assert "formal-hold-draft@1#1" not in active
    assert "formal-hold-other@1#1" not in active


@pytest.mark.parametrize("error", ["digest", "overlap", "purpose", "anchor"])
def test_manifest_rejects_bad_release(error, tmp_path):
    manifest = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    path = tmp_path / "manifest.json"
    for name in {document["path"] for document in manifest["documents"]}:
        (tmp_path / name).write_bytes(SCENARIOS.with_name(name).read_bytes())
    if error == "digest":
        manifest["documents"][0]["source_sha256"] = "0" * 64
        message = "digest changed"
    elif error == "overlap":
        manifest["documents"][0]["effective_to"] = "2026-11-06"
        message = "Overlapping approved versions"
    elif error == "purpose":
        manifest["purpose"] = "product"
        message = "Ineligible source"
    else:
        manifest["documents"][0]["heading"] = "missing anchor"
        message = "ambiguous Markdown anchor"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        assemble(path)


async def test_knowledge_date_and_clause_are_isolated_from_sql(knowledge_settings):
    calls = []
    app = create_app(
        knowledge_settings,
        model_transport=model(calls),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        for fields in (
            {"as_of": "2026-02-30"},
            {"as_of": "20261105"},
            {"mode": "sql", "as_of": "2026-11-05"},
            {"mode": "sql", "clause_id": "orders"},
        ):
            assert (await client.post("/tasks", json={**BODY, **fields})).status_code == 422
        for fields, outcome in [
            ({"as_of": "2025-12-31"}, "insufficient_evidence"),
            ({"clause_id": "unknown"}, "insufficient_evidence"),
            (
                {"as_of": "2026-11-05", "question": "截至 2026年11月4日，成交营收规则是什么？"},
                "clarify_date",
            ),
            ({"question": "旧版成交营收口径如何定义？"}, "clarify_date"),
        ]:
            response = await client.post("/tasks", json={**BODY, **fields})
            row = await terminal(client, response.json()["task_id"])
            assert row["report"]["knowledge"]["outcome"] == outcome
            assert row["report"]["sql"] is None
        assert not calls


@pytest.mark.parametrize("invalid_property", [False, True])
async def test_registered_conflict_requires_both_sources(
    knowledge_settings, tmp_path, invalid_property
):
    path = tmp_path / "scenarios.json"
    await build_snapshot(SCENARIOS, path)
    knowledge_settings.knowledge.snapshot_path = path

    def handler(request):
        body = json.loads(request.content)
        context = json.loads(body["messages"][-1]["content"])
        (key,) = context["requests"]["q1"]["options"]
        return completion_response({"q1": ["s1_1" if invalid_property else key]})

    app = create_app(
        knowledge_settings,
        model_transport=httpx.MockTransport(handler),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        submitted = await client.post(
            "/tasks", json={**BODY, "question": "演练中补寄复核期限是多少天？"}
        )
        row = await terminal(client, submitted.json()["task_id"])
        if invalid_property:
            assert row["error"]["code"] == "KNOWLEDGE_OUTPUT_INVALID"
        else:
            assert row["report"]["knowledge"]["outcome"] == "conflicting_evidence"
            assert len(row["report"]["knowledge"]["citations"]) == 2


@pytest.mark.parametrize("missing", [True, False])
async def test_answer_requires_coverage_for_each_requested_fact(knowledge_settings, missing):
    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        if "evidence" not in payload:
            return completion_response({"questions": ["营收使用哪种价格？", "折扣率是多少？"]})
        support = next(
            key
            for source in payload["evidence"]
            for key, sentence in source["sentences"].items()
            if "历史成交单价字段为" in sentence
        )
        # Empty support means the fact is missing. Omitting the row violates the contract.
        return completion_response({"q1": [support], **({"q2": []} if missing else {})})

    app = create_app(
        knowledge_settings,
        model_transport=httpx.MockTransport(handler),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        submitted = await client.post(
            "/tasks", json={**BODY, "question": "营收使用哪种价格，折扣率是多少？"}
        )
        row = await terminal(client, submitted.json()["task_id"])
        knowledge = row["report"]["knowledge"]
        assert knowledge["citations"] == []
        if missing:
            assert knowledge["outcome"] == "insufficient_evidence"
            assert knowledge["coverage"][1]["supports"] == []
        else:
            assert row["error"]["code"] == "KNOWLEDGE_OUTPUT_INVALID"
