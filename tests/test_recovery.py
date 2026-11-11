import asyncio
import json
import sqlite3
import time

import httpx
import pytest
from conftest import completion
from test_task_api import BODY, terminal

from agentserve.api.app import create_app
from agentserve.domain.models import AgentError, TaskInput
from agentserve.storage.repository import TaskRepository
from agentserve.storage.steps import StepRepository
from agentserve.tools.sqlite import SQLTool


class ProcessCrash(BaseException):
    """Test-only loss of a worker without its normal exception-to-failure handling."""


def backend(counter):
    def handle(request):
        counter.append(json.loads(request.content))
        body = counter[-1]
        response = (
            completion() if body["tool_choice"] != "none" else completion(answer="有2位客户。")
        )
        response["id"] = f"model-{len(counter)}"
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handle)


async def crash_once(settings, transport, *, stage, step_no, kind):
    reached = asyncio.Event()

    async def hook(actual_stage, task_id, actual_no, actual_kind):
        if (actual_stage, actual_no, actual_kind) == (stage, step_no, kind):
            reached.set()
            raise ProcessCrash()

    app = create_app(settings, model_transport=transport, step_hook=hook)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        async with asyncio.timeout(2):
            await reached.wait()
        row = (await client.get(f"/tasks/{task_id}")).json()
    return task_id, row


async def resume(settings, transport, task_id):
    app = create_app(settings, model_transport=transport)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task = await terminal(client, task_id)
        history = (await client.get(f"/tasks/{task_id}/steps")).json()
        events = (await client.get(f"/tasks/{task_id}/events?limit=1000")).json()["events"]
    return task, history, events


@pytest.mark.parametrize(
    "stage,sql_runs,sql_attempts",
    [("before_execute", 1, 2), ("after_execute", 2, 2), ("after_commit", 1, 1)],
)
async def test_three_crash_windows_reuse_model_and_bound_sql_replay(
    settings, monkeypatch, stage, sql_runs, sql_attempts
):
    calls = []
    executed = []
    original = SQLTool._execute

    def track(tool, *args):
        executed.append(args[1])
        return original(tool, *args)

    monkeypatch.setattr(SQLTool, "_execute", track)
    task_id, before = await crash_once(settings, backend(calls), stage=stage, step_no=1, kind="sql")
    task, history, events = await resume(settings, backend(calls), task_id)
    assert task["status"] == "COMPLETED" and task["report"]["query_result"]["rows"] == [[2]]
    assert len(calls) == 2 and len(executed) == sql_runs
    assert len([a for a in history["attempts"] if a["step_no"] == 1]) == sql_attempts
    assert history["budget"]["model_calls"] == 2
    assert (
        task["started_at"] == before["started_at"] and task["deadline_at"] == before["deadline_at"]
    )
    assert len({e["process_epoch"] for e in events}) == 2
    assert any(e["type"] == "STEP_REUSED" and e["payload"]["step_no"] == 0 for e in events)


async def test_uncommitted_model_response_uses_next_attempt_and_preserves_unknown_usage(settings):
    calls = []
    task_id, _ = await crash_once(
        settings, backend(calls), stage="after_execute", step_no=0, kind="model"
    )
    task, history, _ = await resume(settings, backend(calls), task_id)
    assert task["status"] == "COMPLETED" and len(calls) == 3
    assert task["report"]["model_calls"] == 3 and task["report"]["model_retries"] == 1
    first = history["attempts"][0]
    assert (
        first["status"] == "INTERRUPTED"
        and first["usage"] is None
        and first["outcome_unknown"] == 1
    )
    assert [a["attempt_no"] for a in history["attempts"] if a["step_no"] == 0] == [1, 2]


async def test_committed_answer_replays_without_any_external_execution(settings, monkeypatch):
    calls = []
    task_id, _ = await crash_once(
        settings, backend(calls), stage="after_commit", step_no=2, kind="model"
    )

    def forbidden(*args):
        pytest.fail("Committed output executed again")

    monkeypatch.setattr(SQLTool, "_execute", forbidden)
    task, history, _ = await resume(settings, httpx.MockTransport(forbidden), task_id)
    assert task["status"] == "COMPLETED" and task["report"]["answer"] == "有2位客户。"
    assert len(history["attempts"]) == 3 and history["budget"]["model_calls"] == 2


async def test_budget_reservations_survive_repeated_crashes(settings):
    settings.max_model_calls = 2
    settings.max_model_retries = 2
    transport = httpx.MockTransport(
        lambda _: pytest.fail("Crash before execution must not call model")
    )
    task_id, _ = await crash_once(
        settings, transport, stage="before_execute", step_no=0, kind="model"
    )
    reached = asyncio.Event()

    async def crash(stage, *args):
        if stage == "before_execute":
            reached.set()
            raise ProcessCrash()

    app = create_app(settings, model_transport=transport, step_hook=crash)
    async with app.router.lifespan_context(app), asyncio.timeout(2):
        await reached.wait()
    task, history, _ = await resume(settings, transport, task_id)
    assert task["status"] == "FAILED" and task["error"]["code"] == "MODEL_BUDGET_EXHAUSTED"
    assert task["report"]["model_calls"] == 2 and len(history["attempts"]) == 2


async def test_expired_absolute_deadline_prevents_replay_calls(settings):
    settings.task_timeout_seconds = 0.5
    calls = []
    task_id, before = await crash_once(
        settings, backend(calls), stage="after_commit", step_no=1, kind="sql"
    )
    await asyncio.sleep(max(0, before["deadline_at"] - time.time()) + 0.02)
    task, _, _ = await resume(
        settings,
        httpx.MockTransport(lambda _: pytest.fail("Expired recovered task called model")),
        task_id,
    )
    assert task["error"]["code"] == "TASK_TIMEOUT" and task["deadline_at"] == before["deadline_at"]


async def test_step_output_event_and_attempt_commit_rollback_together(settings):
    repo = await TaskRepository.open(settings.state_database)
    store = StepRepository(repo)
    try:
        task_id, _ = await repo.submit(
            TaskInput(**BODY), config=settings.model_dump(mode="json"), queue_limit=1
        )
        row = await repo.claim_next()
        step = await store.open_step(task_id, row["version"], 0, "sql", {"sql": "SELECT 1"})
        attempt = await store.begin_attempt(task_id, row["version"], step)
        await repo.connection.executescript("""CREATE TRIGGER fail_step BEFORE INSERT ON events
            WHEN NEW.type='STEP_COMMITTED' BEGIN SELECT RAISE(ABORT,'injected step commit failure'); END;""")
        with pytest.raises(sqlite3.IntegrityError):
            await store.complete(
                task_id, row["version"], attempt, {"result": {}}, "SQL_COMPLETED", {}
            )
        history = await store.history(task_id)
        assert history["steps"][0]["status"] == "PENDING" and history["steps"][0]["output"] is None
        assert history["attempts"][0]["status"] == "RUNNING"
        assert not any(e["type"] == "SQL_COMPLETED" for e in await repo.events(task_id))
    finally:
        await repo.close()


async def test_changed_workflow_input_cannot_reuse_old_output(settings):
    repo = await TaskRepository.open(settings.state_database)
    store = StepRepository(repo)
    try:
        task_id, _ = await repo.submit(
            TaskInput(**BODY), config=settings.model_dump(mode="json"), queue_limit=1
        )
        row = await repo.claim_next()
        await store.open_step(task_id, row["version"], 0, "sql", {"sql": "SELECT 1"})
        with pytest.raises(AgentError, match="differs"):
            await store.open_step(task_id, row["version"], 0, "sql", {"sql": "SELECT 2"})
    finally:
        await repo.close()


async def test_sql_repair_budget_is_reconstructed_from_committed_outputs(settings):
    calls = []

    def broken(request):
        calls.append(request)
        return httpx.Response(200, json=completion("SELECT missing FROM customers"))

    task_id, _ = await crash_once(
        settings, httpx.MockTransport(broken), stage="after_commit", step_no=1, kind="sql"
    )
    task, history, _ = await resume(settings, httpx.MockTransport(broken), task_id)
    assert task["error"]["code"] == "SQL_REPAIR_EXHAUSTED"
    assert task["report"]["sql_repairs"] == 2 and len(calls) == 3
    assert len([s for s in history["steps"] if s["kind"] == "sql"]) == 3


async def test_format_repair_budget_is_reconstructed_from_committed_error(settings):
    calls = []

    def invalid(request):
        calls.append(request)
        return httpx.Response(200, json=completion(calls=2))

    task_id, _ = await crash_once(
        settings, httpx.MockTransport(invalid), stage="after_commit", step_no=0, kind="model"
    )
    task, _, _ = await resume(settings, httpx.MockTransport(invalid), task_id)
    assert task["error"]["code"] == "MODEL_PROTOCOL_ERROR"
    assert task["report"]["format_repairs"] == 1 and len(calls) == 2


async def test_exhausted_per_step_retry_budget_is_not_reset(settings):
    transport = httpx.MockTransport(lambda _: pytest.fail("Prepared crash called model"))
    task_id, _ = await crash_once(
        settings, transport, stage="before_execute", step_no=0, kind="model"
    )
    reached = asyncio.Event()

    async def crash(stage, *args):
        if stage == "before_execute":
            reached.set()
            raise ProcessCrash()

    app = create_app(settings, model_transport=transport, step_hook=crash)
    async with app.router.lifespan_context(app), asyncio.timeout(2):
        await reached.wait()
    task, history, _ = await resume(settings, transport, task_id)
    assert task["error"]["code"] == "MODEL_RETRY_EXHAUSTED"
    assert history["budget"] == {"model_calls": 2, "model_retries": 1}


async def test_committed_budget_failure_has_same_reason_after_crash(settings):
    settings.max_model_calls = 2
    settings.max_model_retries = 2
    commits = 0
    reached = asyncio.Event()

    async def hook(stage, *args):
        nonlocal commits
        if stage == "after_commit":
            commits += 1
            if commits == 2:
                reached.set()
                raise ProcessCrash()

    app = create_app(
        settings,
        model_transport=httpx.MockTransport(
            lambda _: httpx.Response(503, json={"error": "injected"})
        ),
        step_hook=hook,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        async with asyncio.timeout(2):
            await reached.wait()
    task, _, _ = await resume(
        settings, httpx.MockTransport(lambda _: pytest.fail("Budget failure must replay")), task_id
    )
    assert task["error"]["code"] == "MODEL_BUDGET_EXHAUSTED"


async def test_cancelled_checkpoint_is_not_recovered_or_overwritten(settings):
    reached = asyncio.Event()
    calls = []

    async def hook(stage, task_id, step_no, kind):
        if (stage, step_no, kind) == ("after_commit", 1, "sql"):
            reached.set()
            await asyncio.Event().wait()

    app = create_app(settings, model_transport=backend(calls), step_hook=hook)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
        await reached.wait()
        assert (await client.post(f"/tasks/{task_id}/cancel")).status_code == 202
    task, history, _ = await resume(
        settings,
        httpx.MockTransport(lambda _: pytest.fail("Cancelled checkpoint revived")),
        task_id,
    )
    assert task["status"] == "CANCELLED" and len(calls) == 1
    assert len(history["steps"]) == 2 and history["steps"][1]["output"]["result"]["rows"] == [[2]]


async def test_cancel_wins_against_an_uncommitted_step_output(settings):
    repo = await TaskRepository.open(settings.state_database)
    store = StepRepository(repo)
    try:
        task_id, _ = await repo.submit(
            TaskInput(**BODY), config=settings.model_dump(mode="json"), queue_limit=1
        )
        row = await repo.claim_next()
        step = await store.open_step(task_id, row["version"], 0, "sql", {"sql": "SELECT 1"})
        attempt = await store.begin_attempt(task_id, row["version"], step)
        await repo.cancel(task_id)
        with pytest.raises(AgentError, match="changed"):
            await store.complete(
                task_id, row["version"], attempt, {"result": {"rows": [[1]]}}, "SQL_COMPLETED", {}
            )
        await repo.record_execution_stopped(task_id)
        history = await store.history(task_id)
        assert history["steps"][0]["output"] is None
        assert history["attempts"][0]["status"] == "INTERRUPTED"
    finally:
        await repo.close()
