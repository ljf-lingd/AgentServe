"""Shared resource lifetime for API and direct CLI execution."""

from contextlib import asynccontextmanager

import httpx

from agentserve.agents.sql import SQLAgent
from agentserve.application.order_workbench import OrderWorkbench
from agentserve.application.service import TaskService
from agentserve.config import Settings
from agentserve.llm.client import VLLMClient
from agentserve.memory.resources import MemoryResources
from agentserve.memory.runtime import MemoryRuntime
from agentserve.tools.sqlite import SQLTool


@asynccontextmanager
async def open_task_service(
    settings: Settings, *, model_transport=None, resource_transport=None, workbench=False,
):
    if settings.session_sam_all_phases and not settings.session_sam:
        raise ValueError("All-phase SAM requires the session-sam serving profile")
    tool = SQLTool(
        {key: db.path for key, db in settings.databases.items()},
        workers=settings.sql_workers,
        timeout_seconds=settings.sql_timeout_seconds,
        row_limit=settings.row_limit,
        byte_limit=settings.result_byte_limit,
        function_profile=settings.sql_function_profile,
    )
    try:
        async with (
            httpx.AsyncClient(
                base_url=settings.base_url + "/", trust_env=False, transport=model_transport
            ) as client,
            httpx.AsyncClient(
                base_url=(
                    settings.memory.resources_url if settings.memory else "http://127.0.0.1:8002"
                )
                + "/",
                trust_env=False,
                transport=resource_transport,
            ) as resource_client,
        ):
            resources = None
            if settings.memory:
                resources = MemoryResources(resource_client)
                await resources.connect(settings.memory.resources_identity)
                settings = settings.model_copy(deep=True)
                settings.memory.resources_identity = resources.info.identity
            agent = SQLAgent(
                VLLMClient(
                    client, settings.model, draft_all_phases=settings.session_sam_all_phases,
                    model_tool_choice=settings.model_tool_choice,
                ),
                tool, settings,
            )
            agent.model.resources = resources
            memory = MemoryRuntime(agent.model, resources, settings.memory) if resources else None
            knowledge = None
            if settings.knowledge:
                from agentserve.knowledge.service import KnowledgeService
                from agentserve.knowledge.snapshot import KnowledgeSnapshot

                if resources is None:
                    raise ValueError("Knowledge requires configured CPU tokenizer resources")
                knowledge = KnowledgeService(
                    KnowledgeSnapshot(
                        settings.knowledge.snapshot_path,
                        settings.knowledge.strategy,
                        resources,
                        purpose=settings.knowledge.purpose,
                    ),
                    agent.model,
                )
            metric = None
            if settings.metric:
                from agentserve.knowledge.metrics import MetricCatalog, MetricService

                metric = MetricService(
                    MetricCatalog(
                        settings.metric.registry_path,
                        purpose=settings.metric.purpose,
                        database_ids=settings.databases,
                    ),
                    tool,
                )
            analysis = None
            if settings.analysis_enabled:
                from agentserve.analytics.service import AnalysisService

                if metric is None or "business_orders_v1" not in settings.databases:
                    raise ValueError("Analysis requires a metric registry and business_orders_v1")
                analysis = AnalysisService(metric.catalog, tool)
            yield TaskService(
                agent, memory, knowledge, metric, analysis,
                workbench=OrderWorkbench(tool, memory) if workbench else None,
            )
    finally:
        tool.close()
