"""Bound task concurrency; the database is the queue and Event only wakes the consumer."""

import asyncio
import logging

from agentserve.application.service import TaskService
from agentserve.domain.models import AnalysisInput, MetricInput, TaskInput, TaskStateError
from agentserve.storage.repository import TaskRepository
from agentserve.storage.steps import StepRepository

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, repository: TaskRepository, service: TaskService, *, step_hook=None):
        self.repository, self.service = repository, service
        self.wake = asyncio.Event()
        self.active: set[asyncio.Task] = set()
        self.workers: dict[str, asyncio.Task] = {}
        self.stopping = False
        self.loop_task: asyncio.Task | None = None
        self.steps = StepRepository(repository)
        self.step_hook = step_hook

    async def start(self) -> None:
        await self.steps.recover_running()
        self.loop_task = asyncio.create_task(self._loop(), name="agentserve-dispatcher")

    async def _loop(self) -> None:
        while not self.stopping:
            self.wake.clear()
            while (
                not self.stopping
                and len(self.active) < self.service.agent.settings.task_concurrency
            ):
                task = await self.repository.claim_next()
                if task is None:
                    break
                worker = asyncio.create_task(
                    self._run(task), name=f"agentserve-task-{task['task_id']}"
                )
                self.active.add(worker)
                self.workers[task["task_id"]] = worker
                worker.add_done_callback(self._done)
            if not self.stopping:
                await self.wake.wait()

    def _done(self, worker: asyncio.Task) -> None:
        self.active.discard(worker)
        self.wake.set()
        if not worker.cancelled() and worker.exception() is not None:
            logger.error("Worker failed to save terminal state", exc_info=worker.exception())

    async def _run(self, row: dict) -> None:
        try:
            if row["config_snapshot"] != self.service.execution_config:
                await self.repository.fail(
                    row["task_id"],
                    "CONFIG_CHANGED",
                    "Queued task configuration differs from this application",
                )
                return
            task = TaskInput.model_construct(
                database_id=row["database_id"],
                question=row.get("question"),
                session_id=row.get("session_id"),
                mode=row.get("mode"),
                answer_mode=row.get("answer_mode"),
                as_of=row.get("as_of"),
                clause_id=row.get("clause_id"),
                metric=MetricInput.model_construct(**row["metric"]) if "metric" in row else None,
                analysis=AnalysisInput.model_construct(**row["analysis"]) if "analysis" in row else None,
            )
            report = await self.service.run(
                task,
                task_id=row["task_id"],
                deadline_at=row["deadline_at"],
                on_event=self.repository.append_event,
                step_store=self.steps,
                version=row["version"],
                step_hook=self.step_hook,
            )
            await self.repository.finish(report, expected_version=row["version"])
        except asyncio.CancelledError:
            await self.repository.record_execution_stopped(row["task_id"])
            await self.steps.pause(row["task_id"], row["version"])
            raise
        except TaskStateError:
            # A cancellation may win after claim or immediately before final commit.
            await self.repository.record_execution_stopped(row["task_id"])
        except Exception:
            # Task top-level is the only catch-all; details stay in application logs.
            logger.exception("Task %s failed", row["task_id"])
            await self.repository.fail(
                row["task_id"],
                "INTERNAL_ERROR",
                "Unexpected task execution error; see application log",
            )
        finally:
            self.workers.pop(row["task_id"], None)

    def cancel(self, task_id: str) -> None:
        worker = self.workers.get(task_id)
        if worker is not None and not worker.done() and not worker.cancelling():
            worker.cancel()

    async def stop(self) -> None:
        self.stopping = True
        self.wake.set()
        if self.loop_task is not None:
            await self.loop_task
        if self.active:
            _, pending = await asyncio.wait(
                self.active, timeout=self.service.agent.settings.shutdown_grace_seconds
            )
            for worker in pending:
                if not worker.cancelling():
                    worker.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
