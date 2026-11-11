import asyncio
import json

import httpx
import pytest
from conftest import completion
from filelock import Timeout

from agentserve.api.app import create_app
from agentserve.application.execution import execution_snapshot
from agentserve.domain.models import TaskInput
from agentserve.storage.repository import TaskRepository

BODY = {"database_id": "demo_orders", "question": "客户数？"}


def model(sql="SELECT COUNT(*) FROM customers", *, gate=None, entered=None):
    async def handle(request):
        body = json.loads(request.content)
        if entered is not None:
            entered.set()
        if gate is not None:
            await gate.wait()
        return httpx.Response(
            200,
            json=completion(sql)
            if body["tool_choice"] != "none"
            else completion(answer="结果来自查询。"),
        )

    return httpx.MockTransport(handle)


async def terminal(client, task_id):
    async with asyncio.timeout(3):
        while True:
            response = await client.get(f"/tasks/{task_id}")
            assert response.status_code == 200
            task = response.json()
            if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return task
            await asyncio.sleep(0.005)


async def test_submit_query_events_and_reopen_without_model(settings):
    app = create_app(settings, model_transport=model())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post("/tasks", json=BODY, headers={"Idempotency-Key": "task-1"})
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        assert response.headers["Location"] == f"/tasks/{task_id}"
        task = await terminal(client, task_id)
        assert task["status"] == "COMPLETED" and task["report"]["query_result"]["rows"] == [[2]]
        assert task["report"]["task_id"] == task_id
        events = (await client.get(f"/tasks/{task_id}/events")).json()["events"]
        assert events[0]["type"] == "TASK_ACCEPTED" and events[-1]["type"] == "TASK_COMPLETED"
        assert "MODEL_COMPLETED" in [e["type"] for e in events]
        page1 = (await client.get(f"/tasks/{task_id}/events?limit=2")).json()
        page2 = (
            await client.get(
                f"/tasks/{task_id}/events?after_event_id={page1['next_after_event_id']}"
            )
        ).json()
        assert page1["events"] + page2["events"] == events
        again = await client.post("/tasks", json=BODY, headers={"Idempotency-Key": "task-1"})
        assert again.status_code == 200 and again.json() == {"task_id": task_id, "created": False}
        conflict = await client.post(
            "/tasks", json={**BODY, "question": "不同问题"}, headers={"Idempotency-Key": "task-1"}
        )
        assert conflict.status_code == 409
    reopened = create_app(
        settings,
        model_transport=httpx.MockTransport(
            lambda _: pytest.fail("No model call on completed task")
        ),
    )
    async with (
        reopened.router.lifespan_context(reopened),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(reopened), base_url="http://test"
        ) as client,
    ):
        saved = (await client.get(f"/tasks/{task_id}")).json()
        assert saved == task


@pytest.mark.parametrize(
    "sql,error",
    [("SELEC broken", "SQL_REPAIR_EXHAUSTED"), ("DELETE FROM customers", "POLICY_REJECTED")],
)
async def test_failures_remain_queryable(settings, sql, error):
    app = create_app(settings, model_transport=model(sql))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        task = await terminal(client, task_id)
        assert task["status"] == "FAILED" and task["error"]["code"] == error
        assert task["report"]["query_result"]["status"] != "ok"


async def test_api_input_errors(settings):
    app = create_app(settings, model_transport=model())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        for bad in [
            {**BODY, "question": "  "},
            {**BODY, "database_id": "../../other"},
            {**BODY, "arbitrary": True},
            {**BODY, "question": 123},
        ]:
            assert (await client.post("/tasks", json=bad)).status_code == 422
        assert (await client.get("/tasks/unknown")).status_code == 404
        assert (await client.get("/tasks/unknown/events")).status_code == 404
        assert (await client.get("/tasks/unknown/events?after_event_id=-1")).status_code == 422


async def test_queue_full_while_model_waits_has_no_database_lock(settings):
    settings.task_concurrency, settings.queue_limit = 1, 1
    gate, entered = asyncio.Event(), asyncio.Event()
    app = create_app(settings, model_transport=model(gate=gate, entered=entered))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        first = (await client.post("/tasks", json=BODY)).json()["task_id"]
        async with asyncio.timeout(1):
            await entered.wait()
        second = (await client.post("/tasks", json=BODY)).json()["task_id"]
        assert (await client.get(f"/tasks/{second}")).json()["status"] == "QUEUED"
        full = await client.post("/tasks", json=BODY)
        assert full.status_code == 429 and full.json()["error"]["code"] == "QUEUE_FULL"
        assert (await client.get(f"/tasks/{first}")).json()["status"] == "RUNNING"
        gate.set()
        assert (await terminal(client, second))["status"] == "COMPLETED"


async def test_task_concurrency_is_bounded_and_results_independent(settings):
    settings.task_concurrency = 2
    active, peak = 0, 0
    overlap = asyncio.Event()

    async def handle(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            overlap.set()
        await asyncio.wait_for(overlap.wait(), timeout=2)
        await asyncio.sleep(0.02)
        body = json.loads(request.content)
        value = next(m["content"] for m in body["messages"] if m["role"] == "user")
        reply = (
            completion(f"SELECT {value}")
            if body["tool_choice"] != "none"
            else completion(answer=value)
        )
        active -= 1
        return httpx.Response(200, json=reply)

    app = create_app(settings, model_transport=httpx.MockTransport(handle))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        accepted = await asyncio.gather(
            *(client.post("/tasks", json={**BODY, "question": str(i)}) for i in range(5))
        )
        tasks = await asyncio.gather(*(terminal(client, r.json()["task_id"]) for r in accepted))
        assert peak == 2
        assert [task["report"]["query_result"]["rows"] for task in tasks] == [
            [[i]] for i in range(5)
        ]


async def test_queued_and_unstarted_running_task_recover_with_original_deadline(settings):
    repo = await TaskRepository.open(settings.state_database)
    config = execution_snapshot(settings)
    running, _ = await repo.submit(TaskInput(**BODY), config=config, queue_limit=10)
    await repo.claim_next()
    queued, _ = await repo.submit(TaskInput(**BODY), config=config, queue_limit=10)
    deadline = (await repo.get(queued))["deadline_at"]
    await repo.close()
    app = create_app(settings, model_transport=model())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task = await terminal(client, queued)
        assert task["status"] == "COMPLETED" and task["deadline_at"] == deadline
        recovered = await terminal(client, running)
        assert recovered["status"] == "COMPLETED"


async def test_expired_queue_does_not_call_model(settings):
    settings.task_timeout_seconds = 0.001
    repo = await TaskRepository.open(settings.state_database)
    task_id, _ = await repo.submit(
        TaskInput(**BODY), config=execution_snapshot(settings), queue_limit=1
    )
    await repo.close()
    await asyncio.sleep(0.01)
    app = create_app(
        settings,
        model_transport=httpx.MockTransport(lambda _: pytest.fail("Expired task called model")),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task = await terminal(client, task_id)
        assert task["error"]["code"] == "TASK_TIMEOUT"


async def test_shutdown_stops_worker_and_keeps_queued_task(settings):
    settings.shutdown_grace_seconds, settings.task_concurrency = 0.01, 1
    gate, entered = asyncio.Event(), asyncio.Event()
    app = create_app(settings, model_transport=model(gate=gate, entered=entered))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        first = (await client.post("/tasks", json=BODY)).json()["task_id"]
        await entered.wait()
        second = (await client.post("/tasks", json=BODY)).json()["task_id"]
    repo = await TaskRepository.open(settings.state_database)
    try:
        assert (await repo.get(first))["status"] == "QUEUED"
        assert (await repo.get(second))["status"] == "QUEUED"
    finally:
        await repo.close()


async def test_one_application_per_state_database(settings):
    first, second = create_app(settings), create_app(settings)
    async with first.router.lifespan_context(first):
        with pytest.raises(Timeout):
            async with second.router.lifespan_context(second):
                pytest.fail("Second dispatcher acquired the same state database")


async def test_state_cannot_be_the_query_database(settings):
    settings.state_database = settings.databases["demo_orders"].path
    app = create_app(settings)
    with pytest.raises(ValueError, match="separate"):
        async with app.router.lifespan_context(app):
            pytest.fail("State database was allowed to overwrite query data")
