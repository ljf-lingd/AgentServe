import json
import time

import httpx
import pytest
from conftest import completion
from test_memory import resources
from test_task_api import BODY, terminal

from agentserve.api.app import create_app
from agentserve.config import MemorySettings
from agentserve.llm.client import VLLMClient


async def test_all_phase_sam_uses_only_committed_sql_from_same_session(settings):
    settings.answer_style = "evidence_direct_v5"
    settings.session_sam = settings.session_sam_all_phases = True
    settings.memory = MemorySettings(policy="window")
    drafts = []
    statement = "SELECT COUNT(*) FROM customers"

    def model(request):
        payload = json.loads(request.content)
        drafts.append(json.loads(payload["vllm_xargs"]["agentserve_v5_sam"]))
        return httpx.Response(200, json=completion(sql=statement))

    app = create_app(
        settings, model_transport=httpx.MockTransport(model),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        first, second = [
            (await client.post("/sessions", json={"database_id": "demo_orders"})).json()["session_id"]
            for _ in range(2)
        ]
        for sid in (first, first, second):
            response = (await client.post("/tasks", json={**BODY, "session_id": sid})).raise_for_status()
            assert (await terminal(client, response.json()["task_id"]))["status"] == "COMPLETED"
    assert drafts == [[], [statement], []]


async def test_v5_sends_only_committed_visible_same_session_answers(settings):
    settings.answer_style = "evidence_direct_v5"
    settings.session_sam = True
    settings.memory = MemorySettings()
    history = []

    def model(request):
        payload = json.loads(request.content)
        if payload["tool_choice"] != "none":
            assert "vllm_xargs" not in payload
            return httpx.Response(200, json=completion())
        draft = json.loads(payload["vllm_xargs"]["agentserve_v5_sam"])
        history.append(draft)
        return httpx.Response(200, json=completion(answer=f"查到了 2 位客户（{len(history)}）。"))

    async def submit(client, session_id, *, explain=False):
        body = {**BODY, "session_id": session_id}
        if explain:
            body["answer_mode"] = "explain"
        response = await client.post("/tasks", json=body)
        assert response.status_code == 202
        row = await terminal(client, response.json()["task_id"])
        assert row["status"] == "COMPLETED"
        return row["report"]

    app = create_app(
        settings,
        model_transport=httpx.MockTransport(model),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        sessions = [
            (await client.post("/sessions", json={"database_id": "demo_orders"})).json()[
                "session_id"
            ]
            for _ in range(2)
        ]
        first = await submit(client, sessions[0], explain=True)
        plain = await submit(client, sessions[0])
        other = await submit(client, sessions[1], explain=True)
        assert plain["model_calls"] == 1 and plain["answer_source"] == "tool_values"
        assert first["answer_source"] == other["answer_source"] == "model"

    reopened = create_app(
        settings,
        model_transport=httpx.MockTransport(model),
        resource_transport=httpx.MockTransport(resources),
    )
    async with (
        reopened.router.lifespan_context(reopened),
        httpx.AsyncClient(transport=httpx.ASGITransport(reopened), base_url="http://test") as client,
    ):
        resumed = await submit(client, sessions[0], explain=True)
        assert resumed["query_result"] == plain["query_result"] == first["query_result"]
        assert history == [[], [], [first["answer"]]]


async def test_v5_truncated_explanation_skips_model(settings):
    settings.answer_style = "evidence_direct_v5"
    settings.session_sam = True
    settings.row_limit = 1

    def model(request):
        payload = json.loads(request.content)
        assert payload["tool_choice"] != "none" and "vllm_xargs" not in payload
        return httpx.Response(200, json=completion("SELECT name FROM customers ORDER BY name"))

    app = create_app(settings, model_transport=httpx.MockTransport(model))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post("/tasks", json={**BODY, "answer_mode": "explain"})
        report = (await terminal(client, response.json()["task_id"]))["report"]
        assert report["query_result"]["truncated"]
        assert report["model_calls"] == 1 and report["answer_source"] == "bounded_summary"


@pytest.mark.parametrize("mode", ["auto", "named"])
async def test_model_tool_choice_deployment_profile(mode):
    payloads = []

    def model(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=completion(sql="SELECT 1"))

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1/v1/", transport=httpx.MockTransport(model),
    ) as http:
        reply = await VLLMClient(http, "test-model", model_tool_choice=mode).generate(
            [{"role": "user", "content": "Count transactions"}],
            query=True, allow_clarify=True, max_tokens=128, deadline=time.monotonic() + 5,
        )
    assert reply.sql == "SELECT 1"
    expected = "auto" if mode == "auto" else {"type": "function", "function": {"name": "execute_sql"}}
    assert payloads[0]["tool_choice"] == expected
