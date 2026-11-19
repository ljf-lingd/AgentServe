"""Freeze and run v5 P0-1 on the original V6 business mix."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx

from agentserve.api.app import create_app
from agentserve.config import Settings
from agentserve.storage.memory import MemoryRepository

ROOT = Path(__file__).resolve().parents[2]
DATABASE = "business_orders_v1"
DATE = "2026-11-10T19:00:00+08:00"
ORIGIN = "http://127.0.0.1"
RANGE = {"start": "2026-01-01", "end": "2026-02-01", "city": "Shenzhen"}
GROUPS = [["chengdu", "march", "follow"], ["shanghai", "january", "direct"],
          ["revenue_rule"], ["count_rule"], ["metric"], ["report"]]
TASKS = {
    "chengdu": {"question": "成都的总交易是多少"},
    "march": {"question": "深圳26年3月有交易吗 交易额是多少？"},
    "follow": {"question": "那上海呢？"},
    "shanghai": {"question": "上海的成交营业额是多少"},
    "january": {"question": "深圳26年一月有交易吗"},
    "direct": {"question": "2026年3月上海有多少订单"},
    "revenue_rule": {"mode": "knowledge", "question": "不加额外条件，哪些订单会进入成交营收？"},
    "count_rule": {"mode": "knowledge", "question": "完成订单量只算哪些订单？"},
    "metric": {"mode": "metric", "metric": {"name": "完成订单量", **RANGE}},
    "report": {"mode": "analysis", "analysis": RANGE},
}
ARMS = {
    "ar": {"session_sam": False, "session_sam_all_phases": False},
    "answer": {"session_sam": True, "session_sam_all_phases": False},
    "all": {"session_sam": True, "session_sam_all_phases": True},
}
BLOCKS = [
    {"concurrency": 1, "order": ["ar", "answer", "all"]},
    {"concurrency": 1, "order": ["all", "answer", "ar"]},
    {"concurrency": 4, "order": ["answer", "all", "ar"]},
    {"concurrency": 4, "order": ["ar", "all", "answer"]},
]
MODEL = "agent-qwen"
SPEC = {"method": "custom_class", "model": "session_sam.SamProposer", "num_speculative_tokens": 4}
ENGINE_ENV = {"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_BATCH_INVARIANT": "1",
              "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_NO_USAGE_STATS": "1",
              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
ENGINE_ARGS = ["--served-model-name", MODEL, "--host", "127.0.0.1", "--dtype", "bfloat16",
               "--tensor-parallel-size", "1", "--max-model-len", "8192", "--max-num-seqs", "4",
               "--gpu-memory-utilization", "0.45", "--enforce-eager", "--no-enable-prefix-caching",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--enable-per-request-metrics"]


def file_sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def gold_rows(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        gold = {}
        for name, city, month in (
            ("chengdu", "Chengdu", ""), ("shanghai", "Shanghai", ""),
            ("march", "Shenzhen", "2026-03"), ("follow", "Shanghai", "2026-03"),
            ("january", "Shenzhen", "2026-01"),
        ):
            gold[name] = list(conn.execute(
                "SELECT COUNT(DISTINCT o.order_id), SUM(i.quantity*i.unit_price_cents) "
                "FROM orders o JOIN customers c USING(customer_id) LEFT JOIN order_items i USING(order_id) "
                "WHERE c.city=? AND o.order_date LIKE ? AND o.status='completed'", (city, month + "%"),
            ).fetchone())
        gold["direct"] = conn.execute(
            "SELECT COUNT(*) FROM orders o JOIN customers c USING(customer_id) "
            "WHERE c.city='Shanghai' AND o.order_date LIKE '2026-03%'",
        ).fetchone()[0]
        return gold


def correct(name, task, gold):
    if task["status"] != "COMPLETED":
        return False
    report = task["report"]
    if name in {"count_rule", "revenue_rule"}:
        anchor = "L9-L9" if name == "count_rule" else "L5-L5"
        return (report["knowledge"]["outcome"] == "answered" and anchor in report["answer"]
                and "completed" in report["answer"] and report.get("query_result") is None)
    result = report.get("query_result") or {}
    if result.get("status") != "ok" or result["truncated"]:
        return False
    rows = result["rows"]
    if name in {"direct", "metric"}:
        return rows == [[gold["direct"] if name == "direct" else gold["january"][0]]]
    if name == "report":
        count, revenue = gold["january"]
        return rows == [["2026-01", revenue, count]]
    count, revenue = gold[name]
    expected = count if name == "january" else revenue
    return (len(rows) == 1 and 1 <= len(rows[0]) <= 2 and expected in rows[0]
            and all(value in (revenue, count) for value in rows[0])
            and (name == "january" or report["answer_source"] == "model"))


def plan(config, base):
    files = sorted((ROOT / "src/agentserve").rglob("*.py")) + [
        Path(__file__).resolve(), ROOT / "serving/vllm/session_sam.py", ROOT / "serving/vllm/session_sam_worker.py",
        config.resolve(), base.knowledge.snapshot_path, base.metric.registry_path,
    ]
    for db in base.databases.values():
        files.extend((db.path, db.schema_path))
    return {
        "version": 1, "business_time": DATE, "gpu": "RTX 4090", "vllm": "0.32.0+cu129",
        "model": "Qwen2.5-7B-Instruct", "arms": ARMS, "blocks": BLOCKS,
        "workload": TASKS, "sessions": GROUPS, "warmup": "one full mix per arm in a separate task database",
        "gold": gold_rows(base.databases[DATABASE].path),
        "source_sha256": {path.relative_to(ROOT).as_posix(): file_sha(path) for path in dict.fromkeys(files)},
        "engine_env": ENGINE_ENV, "engine_args": ENGINE_ARGS, "sam_spec": SPEC,
        "application": {key: getattr(base, key) for key in (
            "task_timeout_seconds", "model_request_timeout_seconds", "max_model_calls",
            "max_model_retries", "sql_timeout_seconds", "sql_workers", "row_limit", "result_byte_limit",
        )},
    }


def engine_args(arm):
    if arm == "ar":
        return ENGINE_ARGS.copy()
    return ENGINE_ARGS + ["--worker-cls", "session_sam_worker.SessionSamWorker", "--speculative-config",
                          json.dumps(SPEC, separators=(",", ":")), "--per-request-spec-decode-metrics", "detailed"]


async def terminal(client, task_id, seconds):
    async with asyncio.timeout(seconds + 2):
        while True:
            task = (await client.get(f"/tasks/{task_id}")).raise_for_status().json()
            if task["status"] not in {"QUEUED", "RUNNING"}:
                return task
            await asyncio.sleep(.03)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {"n": len(ordered), "p50": ordered[math.ceil(.50 * len(ordered)) - 1],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1]}


def phase_stats(rows):
    phases = defaultdict(list)
    for row in rows.values():
        for step in row["journal"]["steps"]:
            reply = (step["output"] or {}).get("reply")
            if step["kind"] != "model" or reply is None:
                continue
            inputs = step["input"]
            phase = inputs.get("purpose") or (
                "sql" if inputs["query"] else "knowledge" if inputs.get("json_schema") else "answer"
            )
            phases[phase].append(reply)
    summaries = {}
    for phase, replies in sorted(phases.items()):
        usage = [reply["usage"] or {} for reply in replies]
        metrics = [reply["metrics"] or {} for reply in replies]
        summaries[phase] = {"replies": len(replies), "metrics": {}}
        for key in ("prompt_tokens", "completion_tokens", "num_draft_tokens", "num_accepted_draft_tokens"):
            source = usage if key in ("prompt_tokens", "completion_tokens") else metrics
            values = [item[key] for item in source if key in item and item[key] is not None]
            summaries[phase][key] = {"n": len(values), "sum": sum(values) if values else None}
        for key in ("time_to_first_token_ms", "mean_itl_ms", "tokens_per_second"):
            summaries[phase]["metrics"][key] = distribution(item[key] for item in metrics if key in item)
    return summaries


def summarize(rows, elapsed):
    return {"tasks": len(rows), "correct": sum(row["correct"] for row in rows.values()),
            "status": dict(Counter(row["task"]["status"] for row in rows.values())),
            "jct_seconds": distribution(row["jct"] for row in rows.values()),
            "tasks_per_second": len(rows) / elapsed, "phases": phase_stats(rows)}


async def run_mix(settings, concurrency, gold):
    original = MemoryRepository.inputs

    async def fixed_time(repository, task_id, recent_turns):
        result = await original(repository, task_id, recent_turns)
        return {**result, "accepted_at": datetime.fromisoformat(DATE).timestamp()}

    app = create_app(settings, web_mode="demo")
    rows = {}
    with patch.object(MemoryRepository, "inputs", fixed_time):
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url=ORIGIN, headers={"Origin": ORIGIN},
        ) as client:
            async def group(names):
                scope = {"database_id": DATABASE}
                if TASKS[names[0]].get("mode", "sql") == "sql":
                    session = (await client.post("/sessions", json=scope)).raise_for_status().json()
                    scope["session_id"] = session["session_id"]
                for name in names:
                    task_id = (await client.post("/tasks", json={**scope, **TASKS[name]})).raise_for_status().json()["task_id"]
                    task = await terminal(client, task_id, settings.task_timeout_seconds)
                    # Demo mode exposes redacted events. Read the full audit through the owned repository.
                    repo = app.state.tasks.repository
                    rows[name] = {"task": await repo.get(task_id), "events": await repo.events(task_id, limit=1000),
                                  "journal": await app.state.tasks.dispatcher.steps.history(task_id),
                                  "jct": task["ended_at"] - task["created_at"]}
                    rows[name]["correct"] = correct(name, rows[name]["task"], gold)

            started = time.perf_counter()
            if concurrency == 1:
                for names in GROUPS:
                    await group(names)
            else:
                await asyncio.gather(*(group(names) for names in GROUPS))
            elapsed = time.perf_counter() - started
    return rows, summarize(rows, elapsed)


async def live(args, base, frozen):
    if args.model_url is None or args.resources_url is None:
        raise ValueError("live requires --model-url and --resources-url")
    base.base_url, base.memory.resources_url = args.model_url.rstrip("/"), args.resources_url.rstrip("/")
    base.model = MODEL
    for key, enabled in ARMS[args.arm].items():
        setattr(base, key, enabled)
    concurrency = BLOCKS[args.block - 1]["concurrency"]
    base.task_concurrency = base.llm_concurrency = concurrency
    args.output.mkdir()
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        models = (await client.get(base.base_url + "/models")).raise_for_status().json()["data"]
        if MODEL not in {model["id"] for model in models}:
            raise ValueError("Endpoint does not expose the frozen served model name")
        base.state_database = args.output / "warmup.sqlite"
        warmup, counts = await run_mix(base, concurrency, frozen["gold"])
        dump(args.output / "warmup.json", {"summary": counts, "cases": warmup})
        if any(task["task"]["status"] != "COMPLETED" for task in warmup.values()):
            raise RuntimeError("Warmup has failed tasks. Inspect warmup.json before timing.")
        metrics_url = base.base_url.removesuffix("/v1") + "/metrics"
        (args.output / "vllm-before.prom").write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
        base.state_database = args.output / "tasks.sqlite"
        rows, counts = await run_mix(base, concurrency, frozen["gold"])
        (args.output / "vllm-after.prom").write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
    dump(args.output / "result.json", {"arm": args.arm, "block": args.block, "concurrency": concurrency,
                                        "summary": counts, "cases": rows, "freeze_sha256": file_sha(args.manifest)})
    print(json.dumps(counts), flush=True)


def semantic_signature(row):
    report = row["task"].get("report") or {}
    return [row["task"]["status"], report.get("sql"), report.get("query_result"),
            report.get("answer"), (report.get("knowledge") or {}).get("citations")]


def compare(output, manifest):
    blocks = []
    for index, block in enumerate(BLOCKS, 1):
        results = {}
        for arm in block["order"]:
            result = json.loads((output / f"b{index}-{arm}" / "result.json").read_bytes())
            if (result["freeze_sha256"] != file_sha(manifest) or result["arm"] != arm
                    or result["block"] != index or set(result["cases"]) != set(TASKS)):
                raise ValueError(f"Block {index} {arm} does not match the frozen experiment")
            results[arm] = result
        pairs = {}
        for baseline, treatment in (("ar", "answer"), ("ar", "all"), ("answer", "all")):
            left, right = results[baseline]["cases"], results[treatment]["cases"]
            comparable = [name for name in TASKS if left[name]["correct"] and right[name]["correct"]]
            pairs[f"{baseline}_to_{treatment}"] = {
                "paired_correct": len(comparable), "total": len(TASKS),
                "jct_delta_seconds": distribution(right[name]["jct"] - left[name]["jct"] for name in comparable),
                "changed_results": [name for name in TASKS if semantic_signature(left[name]) != semantic_signature(right[name])],
                "failed_or_wrong": [name for name in TASKS if name not in comparable],
            }
        blocks.append({"block": index, **block, "pairs": pairs,
                       "arms": {arm: result["summary"] for arm, result in results.items()}})
    dump(output / "comparison.json", blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "engine", "live", "compare"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/app.v5.json")
    parser.add_argument("--arm", choices=ARMS, default="ar")
    parser.add_argument("--block", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-url")
    parser.add_argument("--resources-url")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-python", default="python")
    parser.add_argument("--start-engine", action="store_true")
    args = parser.parse_args()
    if args.action == "engine":
        frozen = json.loads(args.manifest.read_bytes())
        if (frozen["engine_env"] != ENGINE_ENV or frozen["engine_args"] != ENGINE_ARGS or frozen["sam_spec"] != SPEC):
            raise ValueError("Serving settings differ from the frozen contract")
        if args.model_dir is None:
            parser.error("engine requires --model-dir")
        command = [args.vllm_python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model_dir),
                   "--port", str(args.port), *engine_args(args.arm)]
        print(json.dumps({"environment": ENGINE_ENV, "argv": command}), flush=True)
        if args.start_engine:
            env = {**os.environ, **ENGINE_ENV}
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "serving/vllm"), env.get("PYTHONPATH"))))
            os.execvpe(command[0], command, env)
        return
    if args.action == "compare":
        if args.output is None:
            parser.error("compare requires --output")
        compare(args.output, args.manifest)
        return
    base = Settings.load(args.config)
    current = plan(args.config, base)
    if args.action == "freeze":
        dump(args.manifest, current)
    elif args.output is None:
        parser.error("live requires --output")
    elif current != json.loads(args.manifest.read_bytes()):
        raise ValueError("Frozen v5 manifest differs from current sources, data, or contract")
    else:
        asyncio.run(live(args, base, current))


if __name__ == "__main__":
    main()
