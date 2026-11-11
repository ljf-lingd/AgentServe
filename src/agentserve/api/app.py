"""Persistent Task API. Start exactly one worker; FileLock enforces one owner per state DB."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from agentserve.analytics.artifacts import render_artifact
from agentserve.api.web import mount_access, web_settings
from agentserve.application.bootstrap import open_task_service
from agentserve.application.order_workbench import city_label
from agentserve.application.tasks import TaskApplication
from agentserve.config import DEFAULT_APP_CONFIG, Settings, WorkbenchDataset
from agentserve.domain.models import AgentError, SessionInput, TaskInput
from agentserve.runtime.dispatcher import Dispatcher
from agentserve.storage.memory import MemoryRepository
from agentserve.storage.repository import TaskRepository
from agentserve.storage.web import WebRepository, public_error, public_task

BUSINESS_DATABASE = "business_orders_v1"
BUSINESS_CATALOG = WorkbenchDataset(
    label="中文订单",
    first_date="2026-01-01", last_date="2026-06-30", currency="CNY",
    modes=("sql", "knowledge", "metric", "analysis"),
    data_description="订单日期与状态、客户城市、商品目录和订单明细。可按城市和月份查笔数、成交额或商品销量。",
    rule_description="成交营收、订单笔数、客单价、商品销量等计算口径，以及订单字段、空结果和只读查询说明。",
    query_examples=("有哪些城市有订单？", "2026 年 3 月上海有多少笔订单？", "杭州 5 月成交额是多少？"),
    rule_examples=("成交营收算哪些订单？", "完成订单量怎么统计？", "订单金额取成交价还是商品标价？"),
)


class HistoryTitle(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(min_length=1, max_length=80)


class HistorySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(min_length=1, max_length=100)


def create_app(
    settings: Settings | None = None,
    *,
    model_transport=None,
    resource_transport=None,
    step_hook=None,
    web_mode: str | None = None,
) -> FastAPI:
    mode = web_mode if web_mode is not None else os.environ.get("AGENTSERVE_WEB_MODE", "")
    if mode not in {"", "demo", "auth"}:
        raise ValueError("AGENTSERVE_WEB_MODE must be demo or auth")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings or (web_settings(mode) if mode else Settings.load(
            Path(os.environ.get("AGENTSERVE_CONFIG", str(DEFAULT_APP_CONFIG)))
        ))
        if mode:
            app.state.datasets = {
                key: database.workbench
                for key, database in config.databases.items()
                if database.workbench is not None
            }
            if BUSINESS_DATABASE in config.databases:
                app.state.datasets.setdefault(BUSINESS_DATABASE, BUSINESS_CATALOG)
        if config.state_database.resolve() in {
            db.path.resolve() for db in config.databases.values()
        }:
            raise ValueError("State database must be separate from query databases")
        config.state_database.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(config.state_database) + ".lock", timeout=0):
            repository = await TaskRepository.open(config.state_database, web=bool(mode))
            try:
                async with open_task_service(
                    config, model_transport=model_transport, resource_transport=resource_transport,
                    workbench=bool(mode),
                ) as service:
                    dispatcher = Dispatcher(repository, service, step_hook=step_hook)
                    application = TaskApplication(repository, dispatcher)
                    app.state.tasks = application
                    app.state.workbench = service.workbench
                    await dispatcher.start()
                    try:
                        yield
                    finally:
                        application.accepting = False
                        await dispatcher.stop()
            finally:
                await repository.close()

    app = FastAPI(title="AgentServe Task API", version="0.4.0", lifespan=lifespan,
                  openapi_url=None if mode else "/openapi.json")

    def owner(request: Request) -> str | None:
        return request.state.user["user_id"] if mode == "auth" else None

    async def require_visible(request: Request, resource: str, resource_id: str) -> None:
        if mode:
            await WebRepository(request.app.state.tasks.repository).require_visible(
                resource, resource_id, owner(request)
            )

    def require_database(request: Request, database_id: str, task_mode="sql") -> None:
        if mode:
            catalog = request.app.state.datasets.get(database_id)
            if catalog is None:
                raise AgentError("UNKNOWN_DATABASE", "Database unavailable in the workbench")
            if task_mode not in catalog.modes:
                raise AgentError("DATASET_MODE_UNAVAILABLE", "此数据集未开放所选功能。")

    if mode:
        mount_access(app, mode)
        assets = Path(__file__).resolve().parents[3] / "web" / "dist"
        if assets.is_dir():
            app.mount("/app", StaticFiles(directory=assets, html=True), name="workbench")

        @app.exception_handler(RequestValidationError)
        async def invalid_input(request: Request, error: RequestValidationError):
            return JSONResponse(status_code=422, content={"error": {
                "code": "INVALID_INPUT", "detail": "请检查输入格式和日期范围。",
            }})

        @app.get("/sessions")
        async def list_sessions(
            request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 50,
            offset: Annotated[int, Query(ge=0)] = 0,
            database_id: str = BUSINESS_DATABASE,
        ):
            require_database(request, database_id)
            return await WebRepository(request.app.state.tasks.repository).sessions(
                database_id, owner(request), limit, offset
            )

        @app.get("/tasks")
        async def list_tasks(
            request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 50,
            offset: Annotated[int, Query(ge=0)] = 0,
            mode: Literal["knowledge", "metric", "analysis"] | None = None,
            database_id: str = BUSINESS_DATABASE,
        ):
            require_database(request, database_id, mode or "sql")
            return await WebRepository(request.app.state.tasks.repository).independent_tasks(
                database_id, owner(request), limit, offset, mode,
            )

        @app.get("/admin/overview")
        async def overview(request: Request):
            return await WebRepository(request.app.state.tasks.repository).overview()

        @app.post("/admin/overview/reset", status_code=204)
        async def reset_overview(request: Request):
            await WebRepository(request.app.state.tasks.repository).reset_overview()

        @app.get("/catalog/datasets")
        async def list_datasets(request: Request):
            return [
                {"database_id": key, **dataset.model_dump(mode="json")}
                for key, dataset in request.app.state.datasets.items()
            ]

        @app.get("/catalog/cities")
        async def list_cities(request: Request):
            rows = await request.app.state.workbench.cities(time.monotonic() + 3)
            return [{"value": city, "label": city_label(city)} for city, _ in rows]

        @app.patch("/sessions/{session_id}", status_code=204)
        async def rename_session(session_id: str, body: HistoryTitle, request: Request):
            await WebRepository(request.app.state.tasks.repository).change_history(
                "session", [session_id], owner(request), body.title
            )

        @app.delete("/sessions/{session_id}", status_code=204)
        async def remove_session(session_id: str, request: Request):
            await WebRepository(request.app.state.tasks.repository).change_history(
                "session", [session_id], owner(request), None
            )

        @app.patch("/tasks/{task_id}", status_code=204)
        async def rename_task(task_id: str, body: HistoryTitle, request: Request):
            await WebRepository(request.app.state.tasks.repository).change_history(
                "task", [task_id], owner(request), body.title
            )

        @app.delete("/tasks/{task_id}", status_code=204)
        async def remove_task(task_id: str, request: Request):
            await WebRepository(request.app.state.tasks.repository).change_history(
                "task", [task_id], owner(request), None
            )

        @app.post("/sessions/batch-delete", status_code=204)
        async def remove_sessions(body: HistorySelection, request: Request):
            await WebRepository(request.app.state.tasks.repository).change_history(
                "session", body.ids, owner(request), None,
            )

        @app.post("/tasks/batch-delete", status_code=204)
        async def remove_tasks(body: HistorySelection, request: Request):
            await WebRepository(request.app.state.tasks.repository).change_history(
                "task", body.ids, owner(request), None,
            )

        @app.post("/tasks/{task_id}/save", status_code=204)
        async def save_task(task_id: str, request: Request):
            await WebRepository(request.app.state.tasks.repository).save_task(task_id, owner(request))

    @app.exception_handler(AgentError)
    async def agent_error(request: Request, error: AgentError):
        status = {
            "DATASET_MODE_UNAVAILABLE": 422,
            "UNKNOWN_DATABASE": 422,
            "TASK_NOT_FOUND": 404,
            "ARTIFACT_NOT_FOUND": 404,
            "SESSION_NOT_FOUND": 404,
            "SESSION_DATABASE_MISMATCH": 409,
            "SESSION_POLICY_CHANGED": 409,
            "MEMORY_DISABLED": 409,
            "IDEMPOTENCY_CONFLICT": 409,
            "HISTORY_BUSY": 409,
            "HISTORY_NOT_READY": 409,
            "QUEUE_FULL": 429,
            "SERVICE_STOPPING": 503,
        }.get(error.code, 500)
        body = {"code": error.code, "detail": str(error)}
        return JSONResponse(status_code=status, content={"error": public_error(body) if mode else body})

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.post("/sessions", status_code=201)
    async def create_session(body: SessionInput, request: Request):
        require_database(request, body.database_id)
        return await request.app.state.tasks.create_session(body.database_id, owner_id=owner(request))

    @app.get("/sessions/{session_id}/history")
    async def session_history(
        session_id: str,
        request: Request,
        after_turn: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ):
        await require_visible(request, "session", session_id)
        turns = await MemoryRepository(request.app.state.tasks.repository).history(
            session_id, after_turn=after_turn, limit=limit, full=bool(mode)
        )
        return {
            "turns": [public_task(turn) for turn in turns] if mode else turns,
            "next_after_turn": turns[-1]["turn_no"] if turns else after_turn,
        }

    @app.get("/tasks/{task_id}/context")
    async def task_context(task_id: str, request: Request):
        if mode:
            raise AgentError("TASK_NOT_FOUND", "Raw diagnostics are unavailable in web mode")
        await request.app.state.tasks.get(task_id)
        return {
            "context": await MemoryRepository(request.app.state.tasks.repository).context(task_id)
        }

    @app.post("/tasks", status_code=202)
    async def submit(
        task: TaskInput,
        request: Request,
        idempotency_key: Annotated[str | None, Header(min_length=1, max_length=128)] = None,
    ):
        require_database(request, task.database_id, task.mode or "sql")
        if mode and task.mode is None and task.session_id is None:
            return JSONResponse(status_code=422, content={"error": {
                "code": "SESSION_REQUIRED", "detail": "请在查数会话中提交该任务。",
            }})
        task_id, created = await request.app.state.tasks.submit(
            task, idempotency_key, owner_id=owner(request)
        )
        return JSONResponse(
            status_code=202 if created else 200,
            headers={"Location": f"/tasks/{task_id}"},
            content={"task_id": task_id, "created": created},
        )

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str, request: Request):
        await require_visible(request, "task", task_id)
        application = request.app.state.tasks
        task = await application.get(task_id)
        if not mode:
            return task
        result = public_task(task)
        if task.get("save_to_history") is False:
            result["in_history"] = await WebRepository(application.repository).saved_task(task_id)
        return result

    @app.get("/tasks/{task_id}/artifacts/{name}")
    async def task_artifact(task_id: str, name: str, request: Request):
        await require_visible(request, "task", task_id)
        started = time.monotonic()
        application = request.app.state.tasks
        task = await application.get(task_id)
        analysis = (task["report"] or {}).get("analysis")
        if task["status"] != "COMPLETED" or analysis is None:
            raise AgentError("ARTIFACT_NOT_FOUND", "No completed analysis artifact")
        content, media_type = render_artifact(analysis, name)
        await application.repository.record_artifact_download(
            task_id, name, time.monotonic() - started
        )
        return Response(
            content, media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.post("/tasks/{task_id}/cancel")
    async def cancel(task_id: str, request: Request):
        await require_visible(request, "task", task_id)
        result = await request.app.state.tasks.cancel(task_id)
        return JSONResponse(status_code=202 if result["changed"] else 200, content=result)

    @app.get("/tasks/{task_id}/steps")
    async def steps(task_id: str, request: Request):
        if mode:
            raise AgentError("TASK_NOT_FOUND", "Raw diagnostics are unavailable in web mode")
        application = request.app.state.tasks
        await application.get(task_id)
        return await application.dispatcher.steps.history(task_id)

    @app.get("/tasks/{task_id}/events")
    async def events(
        task_id: str,
        request: Request,
        after_event_id: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ):
        await require_visible(request, "task", task_id)
        application = request.app.state.tasks
        await application.get(task_id)
        rows = await application.repository.events(
            task_id, after_event_id=after_event_id, limit=limit
        )
        return {
            "events": [{
                "event_id": row["event_id"], "type": row["type"], "wall_time": row["wall_time"],
            } for row in rows] if mode else rows,
            "next_after_event_id": rows[-1]["event_id"] if rows else after_event_id,
        }

    return app
