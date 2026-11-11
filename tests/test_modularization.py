import asyncio

import httpx
import pytest
from test_task_api import BODY, model, terminal

from agentserve.agents import sql_prompt
from agentserve.api.app import create_app
from agentserve.application import bootstrap
from agentserve.application.bootstrap import open_task_service
from agentserve.application.execution import execution_snapshot
from agentserve.domain.models import RunReport, TaskInput
from agentserve.storage.repository import TaskRepository


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
async def test_shared_resources_close_on_all_exits(settings, monkeypatch, exit_kind):
    closed = []
    original = bootstrap.SQLTool.close

    def close(tool):
        closed.append(tool)
        original(tool)

    monkeypatch.setattr(bootstrap.SQLTool, "close", close)
    client = None
    try:
        async with open_task_service(settings, model_transport=model()) as service:
            client = service.agent.backend.client
            assert not client.is_closed
            if exit_kind == "error":
                raise RuntimeError("test exit")
            if exit_kind == "cancel":
                raise asyncio.CancelledError
    except (RuntimeError, asyncio.CancelledError):
        assert exit_kind != "normal"
    assert len(closed) == 1 and client.is_closed


async def test_api_closes_repository_when_service_startup_fails(settings, monkeypatch):
    closed = []
    original = TaskRepository.close

    async def close(repo):
        closed.append(repo)
        await original(repo)

    def broken(*args, **kwargs):
        raise RuntimeError("tool construction failed")

    monkeypatch.setattr(TaskRepository, "close", close)
    monkeypatch.setattr(bootstrap, "SQLTool", broken)
    app = create_app(settings)
    with pytest.raises(RuntimeError, match="tool construction failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup should fail")
    assert len(closed) == 1


@pytest.mark.parametrize("change", ["missing", "prompt", "workflow"])
async def test_execution_identity_change_rejects_pending_without_model(settings, change):
    config = execution_snapshot(settings)
    if change == "missing":
        del config["execution_identity"]
    elif change == "prompt":
        config["execution_identity"]["prompt_sha256"] = "old-prompt"
    else:
        config["execution_identity"]["workflow_version"] = "old-workflow"
    repo = await TaskRepository.open(settings.state_database)
    try:
        task_id, _ = await repo.submit(TaskInput(**BODY), config=config, queue_limit=1)
        before = await repo.get(task_id)
    finally:
        await repo.close()
    app = create_app(
        settings,
        model_transport=httpx.MockTransport(
            lambda _: pytest.fail("incompatible execution identity called model")
        ),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task = await terminal(client, task_id)
        assert task["error"]["code"] == "CONFIG_CHANGED"
        assert task["deadline_at"] == before["deadline_at"]
        assert task["config_snapshot"] == config
        assert (await client.get(f"/tasks/{task_id}/steps")).json()["attempts"] == []


async def test_completed_legacy_record_stays_readable(settings):
    repo = await TaskRepository.open(settings.state_database)
    try:
        task_id, _ = await repo.submit(
            TaskInput(**BODY), config=settings.model_dump(mode="json"), queue_limit=1
        )
        row = await repo.claim_next()
        report = RunReport(task_id, BODY, status="COMPLETED", answer="historical")
        report.events.append({"type": "TASK_COMPLETED", "payload": {}, "offset_seconds": 0})
        await repo.finish(report, expected_version=row["version"])
        before = await repo.get(task_id)
    finally:
        await repo.close()
    app = create_app(
        settings,
        model_transport=httpx.MockTransport(
            lambda _: pytest.fail("completed historical task was executed")
        ),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        assert (await client.get(f"/tasks/{task_id}")).json() == before


def test_identity_tracks_actual_prompt_and_schema(settings, monkeypatch):
    before = execution_snapshot(settings)
    monkeypatch.setattr(sql_prompt, "SYSTEM_PROMPT", sql_prompt.SYSTEM_PROMPT + "Changed.")
    changed = execution_snapshot(settings)
    assert (
        changed["execution_identity"]["prompt_sha256"]
        != before["execution_identity"]["prompt_sha256"]
    )
    settings.databases["demo_orders"].schema_path.write_text("CREATE TABLE other(x);")
    assert (
        execution_snapshot(settings)["execution_identity"]["schema_sha256"]
        != before["execution_identity"]["schema_sha256"]
    )
