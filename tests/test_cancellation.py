import asyncio
import json
import time

import httpx
import pytest
from conftest import completion
from test_sql_tool import LONG_QUERY
from test_task_api import BODY, model, terminal

from agentserve.api.app import create_app
from agentserve.domain.models import AgentError, RunReport, TaskInput
from agentserve.storage.repository import TaskRepository
from agentserve.storage.steps import StepRepository


async def stopped(client, task_id):
    async with asyncio.timeout(2):
        while True:
            events = (await client.get(f"/tasks/{task_id}/events")).json()["events"]
            if any(e["type"] == "APPLICATION_EXECUTION_STOPPED" for e in events):
                return events
            await asyncio.sleep(0.005)


async def test_cancel_running_and_queued_is_persistent_and_idempotent(settings):
    settings.task_concurrency = 1
    gate, entered = asyncio.Event(), asyncio.Event()
    app = create_app(settings, model_transport=model(gate=gate, entered=entered))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        first = (await client.post("/tasks", json=BODY)).json()["task_id"]
        await entered.wait()
        second = (await client.post("/tasks", json=BODY)).json()["task_id"]
        assert (await client.post(f"/tasks/{second}/cancel")).status_code == 202
        assert (await client.post(f"/tasks/{first}/cancel")).status_code == 202
        assert (await client.post(f"/tasks/{first}/cancel")).status_code == 200
        await stopped(client, first)
        gate.set()
        for task_id in (first, second):
            task = (await client.get(f"/tasks/{task_id}")).json()
            assert task["status"] == "CANCELLED" and task["cancel_requested"] == 1
            events = (await client.get(f"/tasks/{task_id}/events")).json()["events"]
            assert sum(e["type"] == "TASK_CANCELLED" for e in events) == 1
            if task_id == second:
                assert not any(e["type"] == "MODEL_STARTED" for e in events)
        assert (await client.post("/tasks/missing/cancel")).status_code == 404
    repo = await TaskRepository.open(settings.state_database)
    try:
        assert await StepRepository(repo).recover_running() == 0
        assert (await repo.get(first))["status"] == "CANCELLED"
    finally:
        await repo.close()


async def test_cancel_fences_late_events_and_completion(settings):
    repo = await TaskRepository.open(settings.state_database)
    try:
        task_id, _ = await repo.submit(
            TaskInput(**BODY), config=settings.model_dump(mode="json"), queue_limit=1
        )
        row = await repo.claim_next()
        await repo.cancel(task_id)
        report = RunReport(task_id=task_id, task=BODY, status="COMPLETED", answer="late")
        report.events.append({"type": "TASK_COMPLETED", "payload": {}, "offset_seconds": 1})
        with pytest.raises(AgentError, match="changed"):
            await repo.finish(report, expected_version=row["version"])
        with pytest.raises(AgentError, match="no longer"):
            await repo.append_event(
                {"task_id": task_id, "type": "MODEL_COMPLETED", "payload": {}, "offset_seconds": 1}
            )
        assert (await repo.get(task_id))["report"] is None
    finally:
        await repo.close()


async def test_cancel_completed_is_noop(settings):
    app = create_app(settings, model_transport=model())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        before = await terminal(client, task_id)
        response = await client.post(f"/tasks/{task_id}/cancel")
        assert response.status_code == 200 and not response.json()["changed"]
        assert (await client.get(f"/tasks/{task_id}")).json() == before


async def test_cancel_during_real_sql_releases_capacity(settings):
    settings.sql_timeout_seconds = 5
    settings.sql_workers = 1
    settings.task_concurrency = 1

    def handle(request):
        body = json.loads(request.content)
        question = next(m["content"] for m in body["messages"] if m["role"] == "user")
        sql = LONG_QUERY if question == "heavy" else "SELECT COUNT(*) FROM customers"
        return httpx.Response(
            200, json=completion(sql) if body["tool_choice"] != "none" else completion(answer="2")
        )

    app = create_app(settings, model_transport=httpx.MockTransport(handle))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json={**BODY, "question": "heavy"})).json()[
            "task_id"
        ]
        async with asyncio.timeout(1):
            while True:
                events = (await client.get(f"/tasks/{task_id}/events")).json()["events"]
                if any(e["type"] == "SQL_STARTED" for e in events):
                    break
                await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        started = time.monotonic()
        await client.post(f"/tasks/{task_id}/cancel")
        await stopped(client, task_id)
        assert time.monotonic() - started < 1
        next_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        assert (await terminal(client, next_id))["report"]["query_result"]["rows"] == [[2]]


async def test_cancel_between_claim_and_worker_registration_never_calls_model(
    settings, monkeypatch
):
    claimed, proceed = asyncio.Event(), asyncio.Event()
    app = create_app(
        settings,
        model_transport=httpx.MockTransport(
            lambda _: pytest.fail("Cancelled handoff called model")
        ),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        repo = app.state.tasks.repository
        original = repo.claim_next

        async def paused_claim():
            row = await original()
            if row is not None:
                claimed.set()
                await proceed.wait()
            return row

        monkeypatch.setattr(repo, "claim_next", paused_claim)
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        await claimed.wait()
        await client.post(f"/tasks/{task_id}/cancel")
        proceed.set()
        events = await stopped(client, task_id)
        assert not any(e["type"] == "MODEL_STARTED" for e in events)


async def test_cancel_during_retry_backoff_prevents_next_attempt(settings):
    settings.model_retry_backoff_seconds = 1
    requests = 0

    def unavailable(request):
        nonlocal requests
        requests += 1
        return httpx.Response(503, json={"error": "injected"})

    app = create_app(settings, model_transport=httpx.MockTransport(unavailable))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        async with asyncio.timeout(1):
            while True:
                events = (await client.get(f"/tasks/{task_id}/events")).json()["events"]
                if any(e["type"] == "MODEL_RETRY_SCHEDULED" for e in events):
                    break
                await asyncio.sleep(0.005)
        await client.post(f"/tasks/{task_id}/cancel")
        await stopped(client, task_id)
        assert requests == 1


async def test_cancel_waiting_for_model_capacity_releases_no_extra_slot(settings):
    settings.task_concurrency, settings.llm_concurrency = 2, 1
    gate, entered = asyncio.Event(), asyncio.Event()
    app = create_app(settings, model_transport=model(gate=gate, entered=entered))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        first = (await client.post("/tasks", json=BODY)).json()["task_id"]
        await entered.wait()
        second = (await client.post("/tasks", json=BODY)).json()["task_id"]
        async with asyncio.timeout(1):
            while True:
                events = (await client.get(f"/tasks/{second}/events")).json()["events"]
                if any(e["type"] == "MODEL_QUEUED" for e in events):
                    break
                await asyncio.sleep(0.005)
        await client.post(f"/tasks/{second}/cancel")
        events = await stopped(client, second)
        assert not any(e["type"] == "MODEL_STARTED" for e in events)
        await client.post(f"/tasks/{first}/cancel")
        await stopped(client, first)
        gate.set()
        third = (await client.post("/tasks", json=BODY)).json()["task_id"]
        assert (await terminal(client, third))["status"] == "COMPLETED"
