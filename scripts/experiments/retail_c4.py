"""Independent retail workload and four-session replay with request-ID attribution."""

import asyncio
import hashlib
import time

import httpx
from retail_history import TraceTransport, exact_trace, model_trace, write_json
from retail_replay import DATABASE, MONTHS, conversation, workloads
from retail_scale import query

from agentserve.api.app import create_app

CONCURRENCY = 4
PAIRS = (("德国", "法国"), ("比利时", "荷兰"), ("西班牙", "瑞士"),
         ("法国", "英国"), ("澳大利亚", "德国"), ("荷兰", "英国"))
METRICS = ("销售额", "取消金额", "销售发票数", "取消发票数")
MONTH_RANGE = tuple(month for month in MONTHS if "2010-01" <= month <= "2011-11")


def reference_key(row):
    return row["sql"], tuple(row["params"])


def cases():
    used = {
        reference_key(row) for cohort in ("development", "formal")
        for session in workloads(cohort) for row in session
    }
    sessions = []
    for metric in METRICS:
        for left, right in PAIRS:
            months = sorted(
                ((one, two) for one in MONTH_RANGE for two in MONTH_RANGE if one != two),
                key=lambda pair: hashlib.sha256(f"uci-c4-v1:{left}:{right}:{metric}:{pair}".encode()).digest(),
            )
            for initial, following in months:
                sequence = conversation(left, right, initial, following, metric)
                keys = {reference_key(row) for row in sequence}
                if keys.isdisjoint(used):
                    sessions.append(sequence)
                    used.update(keys)
                    break
            else:
                raise ValueError(f"Cannot allocate an unseen UCI sequence for {metric}, {left}/{right}")
    return sessions


def attach_traces(sessions, records):
    captured = {record["id"]: record for record in records if record.get("id")}
    ids = [record["id"] for record in records if record.get("id")]
    if len(captured) != len(ids):
        raise ValueError("Engine response IDs must be unique")
    committed = 0
    exact = 0
    for session in sessions:
        for turn in session["turns"]:
            steps = turn.pop("journal")
            replies = [step["output"]["reply"]["request_id"] for step in steps
                       if step["kind"] == "model" and step["output"] and "reply" in step["output"]]
            trace = [captured[request_id] for request_id in replies if request_id in captured]
            model_trace(trace, steps)
            turn["trace"] = trace
            committed += len(replies)
            exact += sum(map(exact_trace, trace))
    return committed, exact


async def replay(base, arm, sequences, warmup, output, *, model_transport=None, resource_transport=None):
    if len(sequences) % CONCURRENCY or len({len(seq) for seq in sequences}) != 1:
        raise ValueError("C4 requires full waves of equal-length sessions")
    configuration = base.model_copy(deep=True)
    configuration.task_concurrency = configuration.llm_concurrency = CONCURRENCY
    configuration.session_sam = configuration.session_sam_all_phases = arm == "session"
    configuration.state_database = output / "retail.sqlite"
    if configuration.state_database.exists():
        raise FileExistsError(configuration.state_database)
    model = TraceTransport(arm, model_transport or httpx.AsyncHTTPTransport(trust_env=False))
    # Every evaluation question uses explicit dates. Keep the exposed fixture clock
    # later than all previous turn numbers, independent of active request order.
    model.turn_number = 4
    app = create_app(configuration, web_mode="", model_transport=model,
                     resource_transport=resource_transport)
    gate = asyncio.Semaphore(CONCURRENCY)
    sessions = []
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1",
    ) as client:
        async def new_session():
            result = await client.post("/sessions", json={"database_id": DATABASE})
            return result.raise_for_status().json()["session_id"]

        async def run_turn(session, target):
            row = await query(client, target, gate, session_id=session["id"])
            path = f"/tasks/{row['task_id']}"
            task = (await client.get(path)).raise_for_status().json()
            report = task.get("report") or {}
            journal = (await client.get(path + "/steps")).raise_for_status().json()
            row.update(sql=report.get("sql"), answer=report.get("answer"), journal=journal["steps"])
            session["active_seconds"] += row["jct_seconds"]
            session["turns"].append(row)

        warmup_session = {"id": await new_session(), "active_seconds": 0.0, "turns": []}
        await run_turn(warmup_session, warmup)
        if not warmup_session["turns"][0]["correct"]:
            raise RuntimeError("The common retail warmup did not match independent gold")
        model.records.clear()
        started = time.monotonic()
        for start in range(0, len(sequences), CONCURRENCY):
            wave = sequences[start:start + CONCURRENCY]
            ids = await asyncio.gather(*(new_session() for _ in wave))
            active = [{"id": session_id, "active_seconds": 0.0, "turns": []} for session_id in ids]
            for turns in zip(*wave, strict=True):
                await asyncio.gather(*(run_turn(session, target)
                                       for session, target in zip(active, turns, strict=True)))
            sessions.extend(active)
        elapsed = time.monotonic() - started
    committed, exact = attach_traces(sessions, model.records)
    write_json(output / "retail.json", {"arm": arm, "sessions": sessions})
    if not committed or committed != exact:
        raise RuntimeError(f"Expected exact token capture for all committed requests, got {exact}/{committed}")
    return sessions, elapsed, len(model.records)
