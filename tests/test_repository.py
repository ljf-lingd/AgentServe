import asyncio
import sqlite3

import pytest

from agentserve.domain.models import AgentError, RunReport, TaskInput
from agentserve.storage.repository import TaskRepository
from agentserve.storage.steps import StepRepository

TASK = TaskInput(database_id="demo_orders", question="客户数？")


@pytest.fixture
async def repository(settings):
    repo = await TaskRepository.open(settings.state_database)
    yield repo
    await repo.close()


async def submit(repo, settings, task=TASK, key=None):
    return await repo.submit(
        task,
        config=settings.model_dump(mode="json"),
        queue_limit=settings.queue_limit,
        idempotency_key=key,
    )


async def test_acceptance_survives_new_connection_with_event(repository, settings):
    task_id, created = await submit(repository, settings)
    assert created
    second = await TaskRepository.open(settings.state_database)
    try:
        task = await second.get(task_id)
        assert task["status"] == "QUEUED" and task["question"] == TASK.question
        assert [e["type"] for e in await second.events(task_id)] == ["TASK_ACCEPTED"]
        assert task["deadline_at"] > task["created_at"]
    finally:
        await second.close()


async def test_acceptance_and_event_rollback_together(repository, settings):
    await repository.connection.executescript("""CREATE TRIGGER fail_accept BEFORE INSERT ON events
        BEGIN SELECT RAISE(ABORT, 'injected event failure'); END;""")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        await submit(repository, settings)
    rows = await repository.connection.execute_fetchall("SELECT COUNT(*) FROM tasks")
    assert rows[0][0] == 0


async def test_simultaneous_idempotent_submissions_make_one_task(repository, settings):
    results = await asyncio.gather(*(submit(repository, settings, key="same") for _ in range(8)))
    assert len({task_id for task_id, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    assert len(await repository.events(results[0][0])) == 1
    with pytest.raises(AgentError, match="different task"):
        await submit(
            repository, settings, TaskInput(database_id="demo_orders", question="another"), "same"
        )


async def test_capacity_is_atomic_and_idempotency_still_works_when_full(repository, settings):
    settings.queue_limit = 1
    task_id, _ = await submit(repository, settings, key="existing")
    same_id, created = await submit(repository, settings, key="existing")
    assert same_id == task_id and not created
    with pytest.raises(AgentError, match="capacity"):
        await submit(repository, settings)


async def test_claim_is_unique_and_events_use_global_cursor(repository, settings):
    task1, _ = await submit(repository, settings)
    task2, _ = await submit(repository, settings)
    rows = await asyncio.gather(*(repository.claim_next() for _ in range(3)))
    assert {row["task_id"] for row in rows if row} == {task1, task2}
    assert sum(row is None for row in rows) == 1
    events1 = await repository.events(task1, limit=1)
    events2 = await repository.events(task1, after_event_id=events1[0]["event_id"])
    assert [e["type"] for e in events2] == ["TASK_CLAIMED"]
    assert events2[0]["event_id"] > events1[0]["event_id"]


async def test_result_terminal_event_atomicity_and_stale_commit(repository, settings):
    task_id, _ = await submit(repository, settings)
    row = await repository.claim_next()
    report = RunReport(task_id=task_id, task=TASK.model_dump(), status="COMPLETED", answer="2")
    report.events.append(
        {"type": "TASK_COMPLETED", "payload": {"error": None}, "offset_seconds": 1}
    )
    await repository.connection.executescript("""CREATE TRIGGER fail_terminal BEFORE INSERT ON events
        WHEN NEW.type='TASK_COMPLETED' BEGIN SELECT RAISE(ABORT, 'injected terminal failure'); END;""")
    with pytest.raises(sqlite3.IntegrityError):
        await repository.finish(report, expected_version=row["version"])
    task = await repository.get(task_id)
    assert task["status"] == "RUNNING" and task["report"] is None
    await repository.connection.execute("DROP TRIGGER fail_terminal")
    await repository.finish(report, expected_version=row["version"])
    with pytest.raises(AgentError, match="changed"):
        await repository.finish(report, expected_version=row["version"])
    task = await repository.get(task_id)
    assert task["status"] == "COMPLETED" and task["report"]["answer"] == "2"
    assert [e["type"] for e in await repository.events(task_id)].count("TASK_COMPLETED") == 1
    await repository.fail(task_id, "LATE_ERROR", "cannot overwrite completed")
    assert (await repository.get(task_id))["status"] == "COMPLETED"


async def test_startup_recovers_only_running_tasks(repository, settings):
    active_id, _ = await submit(repository, settings)
    await repository.claim_next()
    queued_id, _ = await submit(repository, settings)
    assert await StepRepository(repository).recover_running() == 1
    assert (await repository.get(active_id))["status"] == "QUEUED"
    assert (await repository.get(queued_id))["status"] == "QUEUED"
    assert await StepRepository(repository).recover_running() == 0


async def test_unknown_schema_version_is_rejected(tmp_path):
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="Unsupported"):
        await TaskRepository.open(path)
