import asyncio
import json

import httpx
import pytest
from conftest import completion
from test_task_api import terminal

from agentserve.api.app import create_app
from agentserve.config import MemorySettings
from agentserve.domain.models import AgentError, TaskInput
from agentserve.memory.models import RetrievalPlan
from agentserve.memory.policy import SQLEpisodePolicy
from agentserve.storage.memory import MemoryRepository
from agentserve.storage.repository import TaskRepository


def resources(request):
    if request.url.path == "/info":
        return httpx.Response(
            200,
            json={
                "identity": "test-pinned",
                "embedding_model": "test",
                "embedding_revision": "revision",
                "tokenizer_sha256": {"test": "sha"},
            },
        )
    body = json.loads(request.content)
    if request.url.path == "/embed":
        return httpx.Response(200, json={"vectors": [[1.0] + [0.0] * 1023 for _ in body["texts"]]})
    return httpx.Response(
        200, json={"counts": [max(1, len(t) // 5) for t in body.get("texts", [json.dumps(body)])]}
    )


def backend(calls, fail_write=False):
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            if "Plan retrieval" in body["messages"][0]["content"]:
                value = {
                    "action": "new",
                    "search_query": "客户数量",
                    "keywords": ["客户"],
                    "topic": "customers",
                    "depth": 3,
                }
            else:
                value = {
                    "entries": [
                        {
                            "text": "查询所有客户数量",
                            "keywords": ["客户"],
                            "topic": "customers",
                            "frame": None,
                        }
                    ]
                }
            response = completion(
                answer="bad"
                if fail_write and "entries" in value
                else json.dumps(value, ensure_ascii=False)
            )
        else:
            response = (
                completion(answer="有2位客户。") if body["tool_choice"] == "none" else completion()
            )
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handle)


def memory_app(settings, calls, **kwargs):
    settings.memory = MemorySettings(policy="simplemem")
    return create_app(
        settings,
        model_transport=backend(calls, kwargs.pop("fail_write", False)),
        resource_transport=httpx.MockTransport(resources),
        **kwargs,
    )


async def test_session_commit_history_sources_and_single_turn_compatibility(settings):
    calls = []
    app = memory_app(settings, calls)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        session = (await client.post("/sessions", json={"database_id": "demo_orders"})).json()
        sid = session["session_id"]
        body = {"database_id": "demo_orders", "question": "客户数量", "session_id": sid}
        first = (await client.post("/tasks", json=body, headers={"Idempotency-Key": "one"})).json()
        assert (await client.post("/tasks", json=body, headers={"Idempotency-Key": "one"})).json()[
            "created"
        ] is False
        row = await terminal(client, first["task_id"])
        assert row["status"] == "COMPLETED", row
        assert row["report"]["memory_calls"] == 2 and row["report"]["model_calls"] == 4
        formats = [c["response_format"] for c in calls if "response_format" in c]
        assert all(f["type"] == "json_schema" and f["json_schema"]["strict"] for f in formats)
        second = (await client.post("/tasks", json=body)).json()["task_id"]
        assert (await terminal(client, second))["status"] == "COMPLETED"
        snapshot = (await client.get(f"/tasks/{second}/context")).json()["context"]
        assert snapshot["cutoff_turn"] == 1 and len(snapshot["selected"]) == 1
        assert snapshot["selected"][0]["task_id"] == first["task_id"]
        assert "vector" not in snapshot["selected"][0]
        history = (await client.get(f"/sessions/{sid}/history?limit=1")).json()
        assert len(history["turns"]) == 1 and history["next_after_turn"] == 1
        history = (await client.get(f"/sessions/{sid}/history?after_turn=1")).json()
        assert len(history["turns"]) == 1 and history["turns"][0]["turn_no"] == 2
        single = (
            await client.post("/tasks", json={"database_id": "demo_orders", "question": "客户数量"})
        ).json()["task_id"]
        assert (await terminal(client, single))["report"]["memory_calls"] == 0


async def test_failed_memory_write_retains_sql_and_activates_nothing(settings):
    app = memory_app(settings, [], fail_write=True)
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
                json={"database_id": "demo_orders", "question": "客户数量", "session_id": sid},
            )
        ).json()["task_id"]
        row = await terminal(client, tid)
        assert row["status"] == "FAILED" and row["error"]["stage"] == "memory_write"
        assert row["report"]["query_result"]["rows"] == [[2]]
        assert (
            await app.state.tasks.repository.connection.execute_fetchall(
                "SELECT * FROM memory_entries"
            )
            == []
        )


async def test_ordering_isolation_database_policy_and_queue_rollback(settings, tmp_path):
    repo = await TaskRepository.open(tmp_path / "sessions.sqlite")
    try:
        store = MemoryRepository(repo)
        a = (await store.create_session("demo_orders", "p"))["session_id"]
        b = (await store.create_session("demo_orders", "p"))["session_id"]
        config = {"task_timeout_seconds": 90, "execution_identity": {"memory_policy_sha256": "p"}}

        async def submit(sid, **kwargs):
            return (
                await repo.submit(
                    TaskInput(database_id="demo_orders", question="q", session_id=sid),
                    config=config,
                    queue_limit=kwargs.get("limit", 10),
                )
            )[0]

        a1, a2, b1 = await submit(a), await submit(a), await submit(b)
        assert (await repo.claim_next())["task_id"] == a1
        assert (await repo.claim_next())["task_id"] == b1
        assert await repo.claim_next() is None
        await repo.cancel(a1)
        assert (await repo.claim_next())["task_id"] == a2
        with pytest.raises(AgentError, match="another database"):
            await repo.submit(
                TaskInput(database_id="different", question="q", session_id=a),
                config=config,
                queue_limit=10,
            )
        config["execution_identity"]["memory_policy_sha256"] = "changed"
        with pytest.raises(AgentError, match="new session"):
            await submit(a)
        config["execution_identity"]["memory_policy_sha256"] = "p"
        await submit(a)
        with pytest.raises(AgentError, match="capacity"):
            await submit(a, limit=1)
        rows = await repo.connection.execute_fetchall(
            "SELECT next_turn FROM sessions WHERE session_id=?", (a,)
        )
        assert rows[0][0] == 4
    finally:
        await repo.close()


@pytest.mark.parametrize(
    "stage,step_no", [("before_execute", 1), ("after_commit", 2), ("after_commit", 4)]
)
async def test_recovery_reuses_context_and_committed_memory_model(settings, stage, step_no):
    from test_recovery import ProcessCrash

    reached = asyncio.Event()
    calls = []

    async def crash(actual, task_id, number, kind):
        if actual == stage and number == step_no:
            reached.set()
            raise ProcessCrash()

    app = memory_app(settings, calls, step_hook=crash)
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
                json={"database_id": "demo_orders", "question": "客户数量", "session_id": sid},
            )
        ).json()["task_id"]
        async with asyncio.timeout(3):
            await reached.wait()
        frozen = (await client.get(f"/tasks/{tid}/context")).json()
    app = memory_app(settings, calls)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        row = await terminal(client, tid)
        assert row["status"] == "COMPLETED", row
        assert (await client.get(f"/tasks/{tid}/context")).json() == frozen
        assert sum("Plan retrieval" in c["messages"][0]["content"] for c in calls) == 1
        assert (
            len(
                await app.state.tasks.repository.connection.execute_fetchall(
                    "SELECT * FROM memory_entries"
                )
            )
            == 1
        )


def test_episode_update_binding_and_ablation_do_not_mix_other_analyses():
    entries = [
        {"entry_id": "a", "turn_no": 1},
        {"entry_id": "b", "turn_no": 2, "update_of": "a"},
        {"entry_id": "c", "turn_no": 3, "update_of": None},
    ]
    plan = RetrievalPlan(action="refine", search_query="q", target_id="a")
    assert SQLEpisodePolicy().bind([entries[2]], entries, plan)[0]["entry_id"] == "b"
    plan.target_id = "a"
    assert SQLEpisodePolicy(updates=False).bind([entries[2]], entries, plan)[0]["entry_id"] == "a"
    assert SQLEpisodePolicy(binding=False).bind([entries[2]], entries, plan)[0]["entry_id"] == "c"


async def test_required_context_overflow_is_explicit_not_truncated():
    from agentserve.memory.policy import pack_context

    class Counter:
        async def count(self, text, deadline):
            return len(text)

    inputs = {"recent": [], "cutoff_turn": 2, "revision": 2}
    target = {"entry_id": "required", "text": "x" * 1000}
    with pytest.raises(AgentError, match="Required referenced episode"):
        await pack_context(
            inputs,
            [target],
            RetrievalPlan(action="resume", search_query="q", target_id="required"),
            "sql-episode-v1",
            Counter(),
            MemorySettings(context_tokens=256),
            999,
        )


async def test_cancel_during_memory_write_never_activates_draft(settings):
    reached = asyncio.Event()
    transport = backend([])

    async def handle(request):
        body = json.loads(request.content)
        if "response_format" in body and "Encode this completed" in body["messages"][0]["content"]:
            reached.set()
            await asyncio.Event().wait()
        return await transport.handle_async_request(request)

    settings.memory = MemorySettings(policy="simplemem")
    app = create_app(
        settings,
        model_transport=httpx.MockTransport(handle),
        resource_transport=httpx.MockTransport(resources),
    )
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
                json={"database_id": "demo_orders", "question": "客户数量", "session_id": sid},
            )
        ).json()["task_id"]
        async with asyncio.timeout(3):
            await reached.wait()
        await client.post(f"/tasks/{tid}/cancel")
        assert (await terminal(client, tid))["status"] == "CANCELLED"
        assert (
            await app.state.tasks.repository.connection.execute_fetchall(
                "SELECT * FROM memory_entries"
            )
            == []
        )


async def test_session_sql_can_clarify_without_executing(settings):
    settings.memory = MemorySettings(policy="window")

    def handle(request):
        body = json.loads(request.content)
        assert body["tool_choice"] == "auto"
        return httpx.Response(200, json=completion(answer="CLARIFY: 请提供要恢复的分析条件。"))

    app = create_app(
        settings,
        model_transport=httpx.MockTransport(handle),
        resource_transport=httpx.MockTransport(resources),
    )
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
                json={"database_id": "demo_orders", "question": "继续之前那组", "session_id": sid},
            )
        ).json()["task_id"]
        row = await terminal(client, tid)
        assert row["error"]["code"] == "MEMORY_CONTEXT_REQUIRED" and row["report"]["sql"] is None


async def test_session_schema_error_uses_sql_repair_instead_of_schema_clarification(settings):
    settings.memory = MemorySettings(policy="window")
    choices = []

    def handle(request):
        body = json.loads(request.content)
        choices.append(body["tool_choice"])
        if body["tool_choice"] == "none":
            reply = completion(answer="有2位客户。")
        elif body["tool_choice"] == "auto":
            reply = completion(sql="SELECT nonexistent_column FROM customers")
        else:
            reply = completion()
        return httpx.Response(200, json=reply)

    app = create_app(
        settings,
        model_transport=httpx.MockTransport(handle),
        resource_transport=httpx.MockTransport(resources),
    )
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
                json={"database_id": "demo_orders", "question": "客户数量", "session_id": sid},
            )
        ).json()["task_id"]
        row = await terminal(client, tid)
        assert row["status"] == "COMPLETED", row
        assert row["report"]["sql_repairs"] == 1
        assert row["report"]["query_result"]["rows"] == [[2]]
        assert choices[0] == "auto" and isinstance(choices[1], dict) and choices[2] == "none"
