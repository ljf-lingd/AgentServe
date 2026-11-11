import json

import httpx
from test_task_api import BODY, terminal

from agentserve.api.app import create_app
from agentserve.application.execution import execution_snapshot
from agentserve.domain.models import TaskInput
from agentserve.storage.repository import TaskRepository


async def test_unexpected_model_error_is_saved_and_next_task_can_run(settings):
    calls = 0

    def broken(request):
        nonlocal calls
        calls += 1
        raise ValueError("injected programming failure")

    app = create_app(settings, model_transport=httpx.MockTransport(broken))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        for _ in range(2):
            task_id = (await client.post("/tasks", json=BODY)).json()["task_id"]
            task = await terminal(client, task_id)
            assert task["status"] == "FAILED" and task["error"]["code"] == "INTERNAL_ERROR"
            assert "injected" not in json.dumps(task)
        assert calls == 2


async def test_changed_configuration_does_not_silently_replace_queued_snapshot(settings):
    repo = await TaskRepository.open(settings.state_database)
    old_config = execution_snapshot(settings)
    old_config["sql_max_tokens"] = 128
    task_id, _ = await repo.submit(TaskInput(**BODY), config=old_config, queue_limit=1)
    await repo.close()

    def no_model(request):
        raise AssertionError("Changed configuration must not execute queued task")

    app = create_app(settings, model_transport=httpx.MockTransport(no_model))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        task = await terminal(client, task_id)
        assert task["error"]["code"] == "CONFIG_CHANGED"
        assert task["config_snapshot"] == old_config
