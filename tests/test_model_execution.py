import asyncio
import json
import time

import httpx
import pytest
from conftest import completion
from test_task_api import BODY, terminal

from agentserve.api.app import create_app
from agentserve.domain.models import AgentError, ModelReply, ModelRequestError, RunReport
from agentserve.llm.execution import ModelExecutor


async def call(executor, *, name="task", deadline=None, events=None):
    report = RunReport(task_id=name, task={})

    async def emit(kind, payload):
        if events is not None:
            events.append((name, kind, payload))

    reply = await executor.generate(
        [{"role": "user", "content": name}],
        query=True,
        max_tokens=10,
        deadline=deadline or time.monotonic() + 3,
        report=report,
        emit=emit,
    )
    return report, reply


@pytest.mark.parametrize(
    "status,retry",
    [(429, True), (502, True), (503, True), (504, True), (400, False), (401, False), (500, False)],
)
async def test_http_retry_classification_and_total_request_count(settings, status, retry):
    requests = 0

    def handle(request):
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(status, json={"error": "controlled test failure"})
        payload = json.loads(request.content)
        return httpx.Response(
            200, json=completion() if payload["tool_choice"] != "none" else completion(answer="2")
        )

    settings.model_retry_backoff_seconds = 0
    app = create_app(settings, model_transport=httpx.MockTransport(handle))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        task = await terminal(client, task_id)
        assert task["status"] == ("COMPLETED" if retry else "FAILED")
        assert requests == (3 if retry else 1)
        assert task["report"]["model_calls"] == requests
        assert task["report"]["model_retries"] == int(retry)


async def test_retry_budget_exhaustion_is_finite(settings):
    calls = 0

    class Broken:
        async def generate(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise ModelRequestError(
                "MODEL_CONNECTION_ERROR", "test", retryable=True, outcome_unknown=False
            )

    settings.max_model_retries = 2
    settings.model_retry_backoff_seconds = 0
    events = []
    with pytest.raises(ModelRequestError):
        await call(ModelExecutor(Broken(), settings), events=events)
    assert calls == 3
    assert sum(kind == "MODEL_RETRY_SCHEDULED" for _, kind, _ in events) == 2


async def test_backoff_releases_model_slot(settings):
    order, events = [], []
    settings.llm_concurrency = 1
    settings.model_retry_backoff_seconds = 0.05

    class Backend:
        async def generate(self, messages, **kwargs):
            name = messages[0]["content"]
            order.append(name)
            if len(order) == 1:
                raise ModelRequestError(
                    "MODEL_HTTP_ERROR", "503", retryable=True, outcome_unknown=True
                )
            return ModelReply({}, "SELECT 1", name, None, "tool_calls")

    executor = ModelExecutor(Backend(), settings)
    first = asyncio.create_task(call(executor, name="first", events=events))
    async with asyncio.timeout(1):
        while not any(kind == "MODEL_RETRY_SCHEDULED" for _, kind, _ in events):
            await asyncio.sleep(0)
    await call(executor, name="second")
    await first
    assert order == ["first", "second", "first"]


async def test_total_budget_includes_retries(settings):
    settings.max_model_calls = 2
    settings.max_model_retries = 2
    settings.model_retry_backoff_seconds = 0
    calls = 0

    class Backend:
        async def generate(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise ModelRequestError("MODEL_HTTP_ERROR", "503", retryable=True, outcome_unknown=True)

    with pytest.raises(AgentError, match="budget"):
        await call(ModelExecutor(Backend(), settings))
    assert calls == 2


async def test_waiting_for_capacity_can_timeout_and_cancel_without_leaking(settings):
    settings.llm_concurrency = 1
    gate = asyncio.Event()

    class Backend:
        async def generate(self, *args, **kwargs):
            await gate.wait()
            return ModelReply({}, "SELECT 1", "id", None, "tool_calls")

    executor = ModelExecutor(Backend(), settings)
    await executor.slots.acquire()
    with pytest.raises(AgentError, match="capacity"):
        await call(executor, deadline=time.monotonic() + 0.01)
    pending = asyncio.create_task(call(executor))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    executor.slots.release()
    gate.set()
    await call(executor)


async def test_independent_model_capacity(settings):
    settings.llm_concurrency = 1
    active, peak = 0, 0

    class Backend:
        async def generate(self, *args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            await asyncio.sleep(0.01)
            active -= 1
            return ModelReply({}, "SELECT 1", "id", None, "tool_calls")

    executor = ModelExecutor(Backend(), settings)
    await asyncio.gather(*(call(executor, name=str(i)) for i in range(4)))
    assert peak == 1


async def test_request_timeout_has_unknown_usage_and_bounded_retry(settings):
    settings.model_request_timeout_seconds = 0.01
    settings.model_retry_backoff_seconds = 0

    async def slow(request):
        await asyncio.sleep(10)

    app = create_app(settings, model_transport=httpx.MockTransport(slow))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        task = await terminal(client, task_id)
        assert task["error"]["code"] == "MODEL_TIMEOUT"
        assert task["report"]["model_calls"] == 2 and task["report"]["model_retries"] == 1
        events = (await client.get(f"/tasks/{task_id}/events")).json()["events"]
        failures = [e for e in events if e["type"] == "MODEL_FAILED"]
        assert len(failures) == 2
        assert all(
            e["payload"]["usage"] is None and e["payload"]["outcome_unknown"] for e in failures
        )
