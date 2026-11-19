"""Replay a frozen mixed workload through the workbench with bounded concurrency."""

import asyncio
import json
import random
import time
from datetime import datetime
from unittest.mock import patch

import httpx

from agentserve.api.app import create_app
from agentserve.storage.memory import MemoryRepository
from v5_phase_evidence import ModelTrace
from v6_phase_benchmark import DATABASE, ORIGIN, dump, terminal


def groups(manifest, *, smoke):
    suite = manifest["workload"]
    sql, knowledge, metric, analysis = (suite[k] for k in ("sql", "knowledge", "metric", "analysis"))
    if smoke:
        sql = [sql[0], sql[-1]]
        knowledge = [row for outcome in ("answered", "insufficient_evidence", "clarify_date")
                     for row in [r for r in knowledge if r["label"]["outcome"] == outcome][:2]]
        metric, analysis = metric[:4], analysis[:4]
    mixed = [("sql", session["id"], session["cases"]) for session in sql]
    for kind, records in (("knowledge", knowledge), ("metric", metric), ("analysis", analysis)):
        mixed.extend((kind, case["id"], [case]) for case in records)
    random.Random(501).shuffle(mixed)
    return mixed


def index_cases(workload):
    cases = {f"{kind}:{case['id']}": (kind, group, case)
             for kind, group, records in workload for case in records}
    if len(cases) != sum(len(records) for _, _, records in workload):
        raise ValueError("Case IDs must be unique within each workload kind")
    return cases


async def run(settings, workload, business_time, directory):
    model = ModelTrace()
    app = create_app(settings, web_mode="demo", model_transport=model)
    accepted_at = datetime.fromisoformat(business_time).timestamp()
    get_memory_inputs = MemoryRepository.inputs
    pending = iter(workload)
    rows = {}

    async def fixed_time(repository, task_id, recent_turns):
        result = await get_memory_inputs(repository, task_id, recent_turns)
        return {**result, "accepted_at": accepted_at}

    try:
        with patch.object(MemoryRepository, "inputs", fixed_time), (
            directory / "cases.jsonl"
        ).open("x", encoding="utf-8", buffering=1) as ledger:
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url=ORIGIN, headers={"Origin": ORIGIN},
            ) as client:
                tasks = app.state.tasks

                async def worker():
                    for kind, group, cases in pending:
                        scope = {}
                        if kind == "sql":
                            session = (await client.post("/sessions", json={"database_id": DATABASE})).raise_for_status()
                            scope["session_id"] = session.json()["session_id"]
                        for case in cases:
                            key = f"{kind}:{case['id']}"
                            response = await client.post("/tasks", json={**case["request"], **scope})
                            if response.is_success:
                                task_id = response.json()["task_id"]
                                task = await terminal(client, task_id, settings.task_timeout_seconds)
                                record = {"task": await tasks.repository.get(task_id),
                                          "events": await tasks.repository.events(task_id, limit=1000),
                                          "journal": await tasks.dispatcher.steps.history(task_id),
                                          "jct": task["ended_at"] - task["created_at"]}
                            else:
                                record = {"task": None, "events": [], "journal": {"steps": [], "attempts": []},
                                          "jct": None, "rejection": {"status": response.status_code,
                                                                     "body": response.text}}
                            rows[key] = {**record, "kind": kind, "group": group}
                            ledger.write(json.dumps({"case_id": key, **rows[key]}, ensure_ascii=False) + "\n")
                            if len(rows) % 40 == 0:
                                print(json.dumps({"attempt": str(directory), "recorded": len(rows)}), flush=True)

                started = time.perf_counter()
                async with asyncio.TaskGroup() as workers:
                    for _ in range(settings.task_concurrency):
                        workers.create_task(worker())
                elapsed = time.perf_counter() - started
    finally:
        dump(directory / "model-requests.json", model.calls)
    return rows, elapsed, model.calls
