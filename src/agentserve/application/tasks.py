"""Persistent submission and lookup contracts shared by HTTP and API clients."""

from agentserve.domain.models import AgentError, TaskInput
from agentserve.runtime.dispatcher import Dispatcher
from agentserve.storage.memory import MemoryRepository
from agentserve.storage.repository import TaskRepository


class TaskApplication:
    def __init__(self, repository: TaskRepository, dispatcher: Dispatcher):
        self.repository, self.dispatcher = repository, dispatcher
        self.accepting = True

    async def create_session(self, database_id, *, owner_id: str | None = None):
        settings = self.dispatcher.service.agent.settings
        if not self.accepting:
            raise AgentError("SERVICE_STOPPING", "Application is shutting down")
        if database_id not in settings.databases:
            raise AgentError("UNKNOWN_DATABASE", "database_id is not configured")
        if settings.memory is None:
            raise AgentError("MEMORY_DISABLED", "Use the separate V2 configuration")
        identity = self.dispatcher.service.execution_config["execution_identity"][
            "memory_policy_sha256"
        ]
        return await MemoryRepository(self.repository).create_session(
            database_id, identity, owner_id=owner_id
        )

    async def submit(
        self, task: TaskInput, idempotency_key: str | None = None, *, owner_id: str | None = None
    ) -> tuple[str, bool]:
        if not self.accepting:
            raise AgentError("SERVICE_STOPPING", "Application is shutting down")
        settings = self.dispatcher.service.agent.settings
        if task.database_id not in settings.databases:
            raise AgentError("UNKNOWN_DATABASE", "database_id is not configured")
        if owner_id is not None and idempotency_key is not None:
            idempotency_key = f"{owner_id}:{idempotency_key}"
        result = await self.repository.submit(
            task,
            config=self.dispatcher.service.execution_config,
            queue_limit=settings.queue_limit,
            idempotency_key=idempotency_key,
            owner_id=owner_id,
        )
        self.dispatcher.wake.set()
        return result

    async def get(self, task_id: str) -> dict:
        task = await self.repository.get(task_id)
        if task is None:
            raise AgentError("TASK_NOT_FOUND", "No task with this ID")
        return task

    async def cancel(self, task_id: str) -> dict:
        status, changed = await self.repository.cancel(task_id)
        if changed:
            self.dispatcher.cancel(task_id)
            self.dispatcher.wake.set()
        return {"task_id": task_id, "status": status, "changed": changed}
