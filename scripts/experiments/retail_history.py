"""Compare SAM corpus sources on a fixed UCI retail workload using the Task API."""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from retail_replay import DATABASE, gold, workloads
from retail_scale import query

from agentserve.api.app import create_app
from agentserve.config import Settings


SAM_KEY = "agentserve_v5_sam"
FIXTURE_EPOCH = 1794960000.0  # The pilot has only explicit historical calendar dates.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_ephemeral_fields(messages, turn_number):
    tool_ids = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            identifier = f"chatcmpl-tool-{len(tool_ids):016x}"
            tool_ids[call["id"]] = identifier
            call["id"] = identifier
        if message.get("tool_call_id") in tool_ids:
            message["tool_call_id"] = tool_ids[message["tool_call_id"]]
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("Visible session evidence ("):
            continue
        start = content.index('\n{"recent_turns":') + 1
        memory, end = json.JSONDecoder().raw_decode(content, start)
        memory["accepted_at_unix"] = FIXTURE_EPOCH + turn_number
        for prior in memory["recent_turns"]:
            prior["task_id"] = f"{prior['turn_no']:032x}"
            prior["created_at"] = FIXTURE_EPOCH + prior["turn_no"]
        message["content"] = content[:start] + json.dumps(memory, ensure_ascii=False) + content[end:]


class TraceTransport(httpx.AsyncBaseTransport):
    def __init__(self, arm, transport):
        self.arm = arm
        self.transport = transport
        self.records = []
        self.turn_number = 0
        self.slot_draft = None

    async def handle_async_request(self, request):
        payload = json.loads(await request.aread())
        freeze_ephemeral_fields(payload["messages"], self.turn_number)
        extra = payload.setdefault("vllm_xargs", {})
        if self.arm != "session":
            draft = self.slot_draft if payload.get("tool_choice", "none") != "none" else None
            extra[SAM_KEY] = json.dumps([draft] if draft else [], ensure_ascii=False)
        history = json.loads(extra[SAM_KEY])
        payload["return_token_ids"] = True
        request_hash = digest({
            **payload,
            "vllm_xargs": {key: value for key, value in extra.items() if key != SAM_KEY},
        })
        headers = request.headers.copy()
        headers.pop("Content-Length", None)
        sent = httpx.Request(request.method, request.url, headers=headers, json=payload,
                             extensions=request.extensions)
        started = time.time()
        row = {"start": started, "request_hash": request_hash, "history": history}
        response = None
        try:
            response = await self.transport.handle_async_request(sent)
            body = await response.aread()
            row["http_status"] = response.status_code
            if response.is_success:
                result = json.loads(body)
                choice = result["choices"][0]
                row.update(
                    id=result["id"], prompt=result.get("prompt_token_ids"),
                    tokens=choice.get("token_ids"), finish=choice.get("finish_reason"),
                    spec=(result.get("metrics") or {}).get("speculative_decoding"),
                )
            return response
        finally:
            if response is not None:
                await response.aclose()
            row["end"] = time.time()
            self.records.append(row)

    async def aclose(self):
        await self.transport.aclose()


def model_trace(record, steps):
    by_id = {
        step["output"]["reply"]["request_id"]: (step, step["output"]["reply"])
        for step in steps if step["kind"] == "model" and step["output"] and "reply" in step["output"]
    }
    for item in record:
        committed = by_id.get(item.get("id"))
        if committed is None:
            continue
        step, reply = committed
        inputs = step["input"]
        schema = inputs.get("json_schema") or {}
        item.update(
            step_no=step["step_no"], phase=[inputs["query"], schema.get("title")],
            purpose=inputs.get("purpose"), max_tokens=inputs["max_tokens"],
            committed_at=step["completed_at"], sql=reply.get("sql"),
            content=reply["message"].get("content"),
        )


def exact_trace(row):
    return (
        row.get("http_status") == 200 and "step_no" in row
        and all(isinstance(row.get(key), list) and row[key] and all(type(v) is int for v in row[key])
                for key in ("prompt", "tokens"))
    )


async def run_arm(base, label, arm, cases, warmup, output, *, draft_builder=None):
    config = base.model_copy(deep=True)
    config.session_sam = config.session_sam_all_phases = arm == "session"
    config.state_database = output / f"{label}.sqlite"
    if config.state_database.exists():
        raise FileExistsError(config.state_database)
    model = TraceTransport(arm, httpx.AsyncHTTPTransport(trust_env=False))
    app = create_app(config, web_mode="", model_transport=model)
    gate = asyncio.Semaphore(1)
    sessions = []

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
    ) as client:
        async def conversation(sequence):
            reply = (await client.post("/sessions", json={"database_id": DATABASE})).raise_for_status()
            session_id = reply.json()["session_id"]
            elapsed = 0.0
            turns = []
            committed_sql = None
            for turn_number, turn in enumerate(sequence, start=1):
                model.turn_number = turn_number
                model.slot_draft = draft_builder(turn["question"], committed_sql) if draft_builder else None
                model.records.clear()
                started = time.monotonic()
                row = await query(client, turn, gate, session_id=session_id)
                elapsed += time.monotonic() - started
                path = f"/tasks/{row['task_id']}"
                task = (await client.get(path)).raise_for_status().json()
                journal = (await client.get(path + "/steps")).raise_for_status().json()
                report = task.get("report") or {}
                trace = model.records.copy()
                model_trace(trace, journal["steps"])
                row.update(
                    created_at=task["created_at"], ended_at=task["ended_at"],
                    sql=report.get("sql"), answer=report.get("answer"), trace=trace,
                )
                if row["status"] == "COMPLETED":
                    committed_sql = row["sql"]
                turns.append(row)
            return {"id": session_id, "active_seconds": elapsed, "turns": turns}

        await conversation([warmup])
        for sequence in cases:
            sessions.append(await conversation(sequence))
            write_json(output / f"{label}.json", {"arm": arm, "sessions": sessions})
    rows = [row for session in sessions for row in session["turns"]]
    requests = [trace for row in rows for trace in row["trace"]]
    captured = sum(map(exact_trace, requests))
    print(f"{label} {sum(row['correct'] for row in rows)}/{len(rows)} correct, "
          f"{captured}/{len(requests)} model traces", flush=True)
    if len(requests) == 0 or captured / len(requests) < .95:
        raise RuntimeError(f"Insufficient committed token traces in {label}")
    return sessions


def validate_development(current, repeated, session):
    request_pairs = []
    comparable_inputs = 0
    route_divergences = 0
    for actual, later, treatment in zip(current, repeated, session, strict=True):
        for index, (a, b, c) in enumerate(zip(actual["turns"], later["turns"], treatment["turns"], strict=True)):
            if not all(turn["correct"] for turn in (a, b, c)):
                raise RuntimeError("Development sessions must match all independent SQL gold rows")
            first_inputs = [(t["trace"][0]["request_hash"], t["trace"][0]["prompt"]) for t in (a, b, c)]
            if index == 0 and not first_inputs[0] == first_inputs[1] == first_inputs[2]:
                raise RuntimeError("The first model request differs across development arms")
            if len(a["trace"]) != len(c["trace"]) or first_inputs[0] != first_inputs[2]:
                route_divergences += 1
            for first, second, third in zip(a["trace"], b["trace"], c["trace"]):
                request_pairs.append((first, second, third))
                comparable_inputs += (
                    first["request_hash"] == second["request_hash"] == third["request_hash"]
                    and first["prompt"] == second["prompt"] == third["prompt"]
                )
    if not any(third["history"] for _, _, third in request_pairs):
        raise RuntimeError("Session arm passed no historical text on development retails")
    if not all(all(exact_trace(row) for row in (a, b, c))
               and not a["history"] and not b["history"]
               for a, b, c in request_pairs):
        raise RuntimeError("Development token capture or empty-history validation failed")
    if not any(c["history"] and a["request_hash"] == c["request_hash"] and a["prompt"] == c["prompt"]
               for a, _, c in request_pairs):
        raise RuntimeError("No matched treatment request with nonempty history")
    return {"matched_inputs": comparable_inputs, "route_divergences": route_divergences}


async def run(args):
    draft_builder = None
    treatment = "slots" if args.slot_draft else "session"
    if args.slot_draft:
        from retail_slot_draft import slot_draft

        draft_builder = slot_draft
    label_suffix = "slots" if args.slot_draft else "s"
    dev_arms = [("dev-c1", "current"), ("dev-c2", "current"), (f"dev-{label_suffix}", treatment)]
    formal_arms = [("b1-c", "current"), (f"b1-{label_suffix}", treatment),
                   (f"b2-{label_suffix}", treatment), ("b2-c", "current")]
    base = Settings.load(args.config)
    base.base_url = args.model_url
    base.memory.resources_url = args.resources_url
    base.task_concurrency = base.llm_concurrency = 1
    database = base.databases[DATABASE].path
    frozen = workloads("formal")
    cases = gold(database, [sequence for sequence in frozen if len(sequence) == 3])
    warmup = gold(database, [workloads("development")[0]])[0][0]
    assert len(cases) == 8
    args.output.mkdir(parents=True)
    with database.open("rb") as file:
        database_sha = hashlib.file_digest(file, "sha256").hexdigest()
    manifest = {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "database_sha256": database_sha, "cases_sha256": digest(cases),
        "evaluation_indices": list(range(6)), "development_indices": [6, 7],
        "concurrency": 1, "minimum_improvement": .05,
        "prompt_control": "Fixed per-turn clock, task IDs and tool-call IDs in model-visible messages",
        "execution_order": [label for label, _ in dev_arms + formal_arms], "treatment": treatment,
    }
    if args.slot_draft:
        prototype = Path(__file__).with_name("retail_slot_draft.py")
        manifest["slot_draft_sha256"] = hashlib.sha256(prototype.read_bytes()).hexdigest()
    write_json(args.output / "manifest.json", manifest)
    dev = [await run_arm(base, label, arm, cases[6:], warmup, args.output,
                         draft_builder=draft_builder if arm == "slots" else None)
           for label, arm in dev_arms]
    development = validate_development(*dev)
    noise = [2 * abs(a["active_seconds"] - b["active_seconds"])
             / (a["active_seconds"] + b["active_seconds"])
             for a, b in zip(dev[0], dev[1], strict=True)]
    threshold = max(.05, statistics.median(noise))
    write_json(args.output / "threshold.json", {"noise_by_session": noise, "threshold": threshold,
                                              "development": development})
    print(f"Frozen practical threshold {threshold:.2%}", flush=True)
    for label, arm in formal_arms:
        await run_arm(base, label, arm, cases[:6], warmup, args.output,
                      draft_builder=draft_builder if arm == "slots" else None)
    with database.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != database_sha:
            raise RuntimeError("Read-only database fingerprint changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--resources-url", required=True)
    parser.add_argument("--slot-draft", action="store_true", help="Compare Current with a typed SQL draft")
    asyncio.run(run(parser.parse_args()))
