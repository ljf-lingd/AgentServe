"""Real Task API fault cases with healthy comparisons and ledger assertions."""

import asyncio
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import aiosqlite

from agentserve.config import Settings
from agentserve.storage.memory import MemoryRepository
from agentserve.storage.repository import TaskRepository
from agentserve.storage.steps import StepRepository
from http_server import http_server
from recovery_app import create_recovery_app
from v6_phase_benchmark import DATABASE, DATE, ROOT, TASKS, correct, dump


def create_v5_recovery_app():
    settings = Settings.load(Path(os.environ["AGENTSERVE_CONFIG"]))
    app = create_recovery_app(settings, web_mode="demo")
    lifespan = app.router.lifespan_context
    read_inputs = MemoryRepository.inputs

    async def fixed_time(repository, task_id, recent_turns):
        inputs = await read_inputs(repository, task_id, recent_turns)
        return {**inputs, "accepted_at": datetime.fromisoformat(DATE).timestamp()}

    @asynccontextmanager
    async def timed_lifespan(app):
        with patch.object(MemoryRepository, "inputs", fixed_time):
            async with lifespan(app):
                yield

    app.router.lifespan_context = timed_lifespan
    return app


def read_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []


def poll(check, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if record := check():
            return record
        time.sleep(0.03)
    raise TimeoutError(f"Timed out waiting for {check.__name__}")


class FaultCase:
    def __init__(self, output, base, model_url, resources):
        self.output = output
        self.settings = base.model_copy(deep=True)
        self.settings.memory.resources_url = resources
        self.settings.state_database = output / "tasks.sqlite"
        self.settings.task_concurrency = self.settings.llm_concurrency = 1
        self.proxy_log = output / "proxy.jsonl"
        self.step_log = output / "steps.jsonl"
        self.marker = output / "marker.json"
        output.mkdir()
        dump(output / "proxy.json", {"target_url": model_url, "audit_path": str(self.proxy_log)})

    @contextmanager
    def proxy(self):
        with http_server("fault_proxy:create_proxy", self.output, "proxy", env={
            "AGENTSERVE_PROXY_CONFIG": str(self.output / "proxy.json"),
        }, app_dir=ROOT / "scripts/experiments") as client:
            self.settings.base_url = str(client.base_url).rstrip("/") + "/v1"
            dump(self.output / "app.json", self.settings.model_dump(mode="json"))
            yield client

    @contextmanager
    def api(self, phase, *, pause_sql=None):
        probe = self.output / f"{phase}-probe.json"
        dump(probe, {"enabled": pause_sql is not None, "stage": "after_execute",
                     "step_no": pause_sql, "kind": "sql", "marker_path": str(self.marker),
                     "audit_path": str(self.step_log)})
        owned = []
        with http_server("v5_fault_cases:create_v5_recovery_app", self.output, phase, env={
            "AGENTSERVE_CONFIG": str(self.output / "app.json"),
            "AGENTSERVE_RECOVERY_PROBE": str(probe),
        }, app_dir=ROOT / "scripts/experiments", on_ready=owned.append) as client:
            client.headers["Origin"] = str(client.base_url).rstrip("/")
            yield client, owned[0]

    def quick_check(self):
        path = self.settings.state_database.as_uri()
        with closing(sqlite3.connect(path + "?mode=ro", uri=True)) as conn:
            return conn.execute("PRAGMA quick_check").fetchone()[0]

    def receipt(self, task_id):
        async def read():
            uri = self.settings.state_database.as_uri() + "?mode=ro"
            async with aiosqlite.connect(uri, uri=True) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("BEGIN"):
                    repo = TaskRepository(db, web=True)
                    return {"task": await repo.get(task_id), "events": await repo.events(task_id, limit=1000),
                            "journal": await StepRepository(repo).history(task_id)}

        return asyncio.run(read())


def post(client, name, session=None):
    body = {"database_id": DATABASE, **TASKS[name]}
    if session:
        body["session_id"] = session
    response = client.post("/tasks", headers={"Idempotency-Key": f"v5-{name}"}, json=body)
    return response.raise_for_status().json()


def open_session(client):
    return client.post("/sessions", json={"database_id": DATABASE}).raise_for_status().json()["session_id"]


def events(client, task_id):
    return client.get(f"/tasks/{task_id}/events?limit=1000").raise_for_status().json()["events"]


def terminal(client, task_id):
    def finished():
        task = client.get(f"/tasks/{task_id}").raise_for_status().json()
        return task if task["status"] not in {"QUEUED", "RUNNING"} else None

    return poll(finished, 95)


def model_ids(run):
    return [step["output"]["reply"]["request_id"] for step in run["journal"]["steps"]
            if step["kind"] == "model" and step["status"] == "COMPLETED"]


def validate_healthy_result(name, run, gold):
    assert correct(name, run["task"], gold), (name, run["task"]["status"], run["task"]["error"])
    assert sum(event["type"] == "TASK_COMPLETED" for event in run["events"]) == 1


def match_health(run, healthy, gold):
    validate_healthy_result("chengdu", run, gold)
    report = run["task"]["report"]
    baseline = healthy["chengdu"]["task"]["report"]
    assert report["query_result"]["rows"] == baseline["query_result"]["rows"]
    assert len(model_ids(run)) == len(model_ids(healthy["chengdu"]))


def healthy(case, gold):
    runs = {}
    with case.proxy(), case.api("healthy") as (client, _):
        for name, session in (("chengdu", open_session(client)), ("revenue_rule", None)):
            task_id = post(client, name, session)["task_id"]
            terminal(client, task_id)
            runs[name] = case.receipt(task_id)
            dump(case.output / f"{name}.json", runs[name])
            validate_healthy_result(name, runs[name], gold)
    records = read_lines(case.proxy_log)
    ids = model_ids(runs["chengdu"]) + model_ids(runs["revenue_rule"])
    assert [row["request_id"] for row in records if row["type"] == "UPSTREAM_COMPLETED"] == ids
    return runs


def restart(case, baseline, gold):
    expected_sql = baseline["chengdu"]["task"]["report"]["sql"]
    sql_step = next(step["step_no"] for step in baseline["chengdu"]["journal"]["steps"]
                    if step["kind"] == "sql" and step["input"]["sql"] == expected_sql)
    with case.proxy():
        with case.api("before-exit", pause_sql=sql_step) as (client, process):
            task_id = post(client, "chengdu", open_session(client))["task_id"]
            poll(case.marker.exists, 50)
            marker = json.loads(case.marker.read_bytes())
            assert marker["pid"] == process.pid and marker["task_id"] == task_id
            before = case.receipt(task_id)
            assert before["task"]["status"] == "RUNNING"
            dump(case.output / "before-exit.json", before)
            process.kill()
            process.wait(timeout=5)
        assert case.quick_check() == "ok"
        with case.api("after-exit") as (client, _):
            terminal(client, task_id)
            run = case.receipt(task_id)
            dump(case.output / "after-exit.json", run)
    match_health(run, baseline, gold)
    for key in ("started_at", "deadline_at", "task_id"):
        assert run["task"][key] == before["task"][key]
    assert any(event["type"] == "TASK_RECOVERED" for event in run["events"])
    assert any(event["type"] == "STEP_REUSED" and event["payload"]["kind"] == "model"
               for event in run["events"])
    interrupted = [a for a in run["journal"]["attempts"] if a["status"] == "INTERRUPTED"]
    assert len(interrupted) == 1 and interrupted[0]["step_no"] == sql_step
    executed = [step for step in read_lines(case.step_log) if step["kind"] == "sql"
                and step["stage"] == "after_execute" and step["task_id"] == task_id
                and step["step_no"] == sql_step]
    assert len(executed) == 2 and executed[0]["pid"] != executed[1]["pid"]
    calls = [row["request_id"] for row in read_lines(case.proxy_log) if row["type"] == "UPSTREAM_COMPLETED"]
    assert calls == model_ids(run) and set(model_ids(before)) <= set(calls)
    return {"task_id": task_id, "sql_step": sql_step, "sql_executions": len(executed), "model_requests": len(calls),
            "reused_model_requests": len(model_ids(before)), "processes": [row["pid"] for row in executed],
            "same_started_at_and_deadline": True}


def timeout(case, baseline, gold):
    with case.proxy() as proxy, case.api("timeout") as (client, _):
        proxy.post("/control/hold_once").raise_for_status()
        task_id = post(client, "chengdu", open_session(client))["task_id"]
        original_deadline = case.receipt(task_id)["task"]["deadline_at"]

        def failure():
            return any(event["type"] == "MODEL_FAILED" for event in events(client, task_id))

        poll(failure, case.settings.model_request_timeout_seconds + 12)
        proxy.post("/control/release").raise_for_status()
        terminal(client, task_id)
        run = case.receipt(task_id)
        dump(case.output / "task.json", run)
    match_health(run, baseline, gold)
    records = read_lines(case.proxy_log)
    holds = [row for row in records if row["type"] == "RESPONSE_HELD"]
    failed = [event for event in run["events"] if event["type"] == "MODEL_FAILED"]
    assert len(failed) == 1 and failed[0]["payload"]["code"] == "MODEL_TIMEOUT"
    failed = failed[0]
    assert len(holds) == 1 and holds[0]["wall_time"] < failed["wall_time"]
    assert [row["sequence"] for row in records if row["type"] == "RESPONSE_RELEASED"] == [holds[0]["sequence"]]
    started = next(row for row in records if row["type"] == "UPSTREAM_STARTED")
    completed = [row for row in records if row["type"] == "UPSTREAM_COMPLETED"]
    upstream_seconds = completed[0]["wall_time"] - started["wall_time"]
    assert upstream_seconds < case.settings.model_request_timeout_seconds
    assert len(completed) == len(model_ids(run)) + 1 and completed[0]["request_id"] not in model_ids(run)
    attempts = [attempt for attempt in run["journal"]["attempts"]
                if (attempt["error"] or {}).get("code") == "MODEL_TIMEOUT"]
    assert len(attempts) == 1 and attempts[0]["outcome_unknown"] and attempts[0]["usage"] is None
    assert sum(event["type"] == "MODEL_RETRY_SCHEDULED" for event in run["events"]) == 1
    assert run["task"]["ended_at"] <= run["task"]["deadline_at"] == original_deadline
    return {"task_id": task_id, "upstream_requests": len(completed), "first_upstream_seconds": upstream_seconds,
            "client_timeout_seconds": failed["payload"]["client_seconds"], "retries": 1,
            "unknown_usage": True, "same_deadline": True, "discarded_request_id": completed[0]["request_id"]}


def cancel(case):
    with case.proxy() as proxy:
        with case.api("cancel") as (client, _):
            proxy.post("/control/hold_once").raise_for_status()
            with ThreadPoolExecutor(max_workers=2) as pool:
                session = open_session(client)
                first = pool.submit(post, client, "chengdu", session)
                second = pool.submit(post, client, "chengdu", session)
                left, right = first.result(), second.result()
            running = left["task_id"]
            assert running == right["task_id"] and left["created"] != right["created"]

            def held():
                return next((row for row in read_lines(case.proxy_log) if row["type"] == "RESPONSE_HELD"), None)

            poll(held)
            assert client.get(f"/tasks/{running}").raise_for_status().json()["status"] == "RUNNING"
            queued = post(client, "revenue_rule")["task_id"]
            assert client.get(f"/tasks/{queued}").raise_for_status().json()["status"] == "QUEUED"
            for task_id in (queued, running):
                assert client.post(f"/tasks/{task_id}/cancel").status_code == 202

            def stopped():
                return next((event for event in events(client, running)
                             if event["type"] == "APPLICATION_EXECUTION_STOPPED"), None)

            poll(stopped)
            proxy.post("/control/release").raise_for_status()
            collected = {task_id: case.receipt(task_id) for task_id in (queued, running)}
            dump(case.output / "before-reopen.json", collected)
        with case.api("cancel-reopened"):
            reopened = {task_id: case.receipt(task_id) for task_id in (queued, running)}
            dump(case.output / "after-reopen.json", reopened)
    for task_id, result in reopened.items():
        assert result["task"]["status"] == "CANCELLED"
        assert result["task"]["ended_at"] == collected[task_id]["task"]["ended_at"]
        assert not any(event["type"] in {"TASK_RECOVERED", "TASK_COMPLETED"} for event in result["events"])
        assert sum(event["type"] == "TASK_CANCELLED" for event in result["events"]) == 1
    assert not reopened[queued]["journal"]["attempts"]
    assert not model_ids(reopened[running])
    proxy_events = read_lines(case.proxy_log)
    assert sum(row["type"] == "UPSTREAM_COMPLETED" for row in proxy_events) == 1
    assert sum(row["type"] == "RESPONSE_RELEASED" for row in proxy_events) == 1
    stop = next(event for event in reopened[running]["events"] if event["type"] == "APPLICATION_EXECUTION_STOPPED")
    return {"running_id": running, "queued_id": queued, "duplicate_single_task": True,
            "running_and_queued_cancelled": True, "gpu_generation_stopped": stop["payload"]["gpu_generation_stopped"]}
