"""Real GPU session comparisons. Full traces and SQLite remain on the server."""

import argparse
import asyncio
import hashlib
import json
import time
from contextlib import ExitStack
from datetime import UTC, datetime
import os
from pathlib import Path

import httpx
from gpu_monitor import monitor
from http_server import http_server
from model_server import temporary_model

from agentserve.config import Settings
from agentserve.evaluation.sql import SQLEvaluator
from agentserve.workloads.memory import conversation, freeze

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ["window", "simplemem", "sql_episode", "no_binding", "no_updates"]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_sessions(endpoint, conversations, gold, settings, output, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    evaluator = SQLEvaluator({k: v.path for k, v in settings.databases.items()})
    records = []
    async with httpx.AsyncClient(base_url=endpoint, timeout=30, trust_env=False) as client:

        async def run(conversation):
            async with semaphore:
                response = await client.post(
                    "/sessions", json={"database_id": "business_orders_v1"}
                )
                response.raise_for_status()
                session_id = response.json()["session_id"]
                history = []
                for turn in conversation["turns"]:
                    started = time.monotonic()
                    # Payload uses only runtime questions; gold never crosses this boundary.
                    response = await client.post(
                        "/tasks",
                        json={
                            "database_id": "business_orders_v1",
                            "session_id": session_id,
                            "question": turn["question"],
                        },
                        headers={"Idempotency-Key": turn["id"]},
                    )
                    response.raise_for_status()
                    task_id = response.json()["task_id"]
                    async with asyncio.timeout(settings.task_timeout_seconds + 30):
                        while True:
                            response = await client.get(f"/tasks/{task_id}")
                            response.raise_for_status()
                            row = response.json()
                            if row["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                                break
                            await asyncio.sleep(0.1)
                    terminal_received = time.monotonic()
                    report = row["report"] or {}
                    context = (await client.get(f"/tasks/{task_id}/context")).json()["context"]
                    steps = (await client.get(f"/tasks/{task_id}/steps")).json()
                    label = gold[turn["id"]]
                    if label["clarify"]:
                        correct = (
                            report.get("sql") is None
                            and (row["error"] or {}).get("code") == "MEMORY_CONTEXT_REQUIRED"
                        )
                        evaluation = {
                            "status": "correct" if correct else "incorrect",
                            "reason": "clarification",
                        }
                    else:
                        evaluation = await evaluator.evaluate(
                            "business_orders_v1",
                            report.get("sql"),
                            label["sql"],
                            ordered=label["ordered"],
                        )
                    record = {
                        "id": turn["id"],
                        "conversation": conversation["id"],
                        "session_id": session_id,
                        "question": turn["question"],
                        "task": row,
                        "context": context,
                        "steps": steps,
                        "evaluation": evaluation,
                        "tags": label["tags"],
                        "clarify": label["clarify"],
                        "jct_seconds": terminal_received - started,
                        "offline_audit_evaluation_seconds": time.monotonic() - terminal_received,
                    }
                    with (output / "turns.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    history.extend(
                        [
                            {"role": "user", "content": turn["question"]},
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "sql": report.get("sql"),
                                        "answer": report.get("answer"),
                                        "error": row["error"],
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ]
                    )
                async with httpx.AsyncClient(
                    base_url=settings.memory.resources_url, trust_env=False
                ) as resources:
                    response = await resources.post(
                        "/count", json={"messages": history}, timeout=30
                    )
                    response.raise_for_status()
                    count = response.json()["counts"][0]
                    if len(conversation["turns"]) == 48 and count <= 8192:
                        raise AssertionError("Long conversation did not exceed 8192 history tokens")
                    write(output / f"{conversation['id']}-history-tokens.json", {"tokens": count})
                print(
                    json.dumps(
                        {
                            "policy": settings.memory.policy,
                            "session": conversation["id"],
                            "turns_done": len(records),
                        }
                    ),
                    flush=True,
                )

        try:
            await asyncio.gather(*(run(c) for c in conversations))
        finally:
            evaluator.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "acceptance"], default="development")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES[:3])
    parser.add_argument(
        "--sessions", type=int, help="Explicit smoke subset; never an acceptance result"
    )
    parser.add_argument("--concurrency", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--recovery", action="store_true", help="Real owned API SIGKILL checks")
    parser.add_argument(
        "--long-probe", action="store_true", help="Separate 48-turn development probe"
    )
    parser.add_argument(
        "--freeze", type=Path, help="Required development freeze.json for full acceptance"
    )
    parser.add_argument(
        "--regression", action="store_true", help="Also rerun the existing 40 business cases"
    )
    args = parser.parse_args()
    if args.split == "acceptance" and (args.sessions or not args.freeze):
        parser.error("Full acceptance requires --freeze and cannot use --sessions")
    if args.long_probe and (args.split != "development" or args.sessions):
        parser.error("--long-probe is a separate development-only probe")
    freeze(ROOT)
    conversations = json.loads((ROOT / f"data/splits/memory_v2/{args.split}.json").read_text())
    gold = json.loads((ROOT / f"data/gold/memory_v2/{args.split}.json").read_text())
    if args.sessions:
        conversations = conversations[: args.sessions]
    if args.long_probe:
        probe, gold = conversation(90, 48, "probe")
        conversations = [probe]
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for base in [ROOT / "src", ROOT / "scripts"]
        for p in sorted(base.rglob("*.py"))
    }
    for p in [
        ROOT / "configs/app.v2.json",
        ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json",
        ROOT / "models/Qwen3-Embedding-0.6B/agent-manifest.json",
    ]:
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = None
    if args.freeze:
        frozen = json.loads(args.freeze.read_text())
        if frozen["source_sha256"] != sources:
            raise ValueError("Code changed after development; freeze a new development version")
    output = (
        ROOT
        / "results/v2-memory"
        / (args.split + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    info = {
        "state": "prepared",
        "split": args.split,
        "smoke_subset": bool(args.sessions or args.long_probe),
        "policies": args.policies,
        "session_count": len(conversations),
        "source_sha256": sources,
        "manifest": json.loads((ROOT / "data/splits/memory_v2/manifest.json").read_text()),
        "embedding": json.loads(
            (ROOT / "models/Qwen3-Embedding-0.6B/agent-manifest.json").read_text()
        ),
        "generation_model": json.loads(
            (ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json").read_text()
        ),
        "answer_faithfulness": "unreviewed_until_attributed_review",
        "runs": [],
    }
    write(output / "summary.json", info)
    print(json.dumps({"output": str(output)}), flush=True)
    settings = Settings.load(ROOT / "configs/app.v2.json")
    before = {
        k: hashlib.sha256(db.path.read_bytes()).hexdigest() for k, db in settings.databases.items()
    }
    info["base_settings"] = settings.model_dump(mode="json")
    info["database_sha256"] = before
    if frozen and (frozen["manifest"] != info["manifest"] or frozen["database_sha256"] != before):
        raise ValueError("Frozen data or database differs from development")
    try:
        with ExitStack() as stack:
            stack.enter_context(monitor(output))
            resource = stack.enter_context(
                http_server(
                    "memory_resources:create_app",
                    output,
                    "cpu-resources",
                    app_dir=ROOT / "scripts/experiments",
                    env={
                        "AGENT_EMBEDDING_PATH": str(ROOT / "models/Qwen3-Embedding-0.6B"),
                        "AGENT_TOKENIZER_PATH": str(ROOT / "models/Qwen2.5-7B-Instruct"),
                        "CUDA_VISIBLE_DEVICES": "",
                    },
                    python_executable=os.environ.get("VLLM_PYTHON", str(Path.home() / "miniconda3/envs/agent-vllm/bin/python")),
                    readiness_seconds=120,
                )
            )
            settings.memory.resources_url = str(resource.base_url).rstrip("/")
            settings.memory.resources_identity = resource.get("/info").json()["identity"]
            if frozen and resource.get("/info").json() != frozen["resources"]:
                raise ValueError("CPU resources differ from the development freeze")
            settings.base_url = stack.enter_context(
                temporary_model(output, max_model_len=8192, verify_gpu=True)
            )
            info["resources"] = resource.get("/info").json()
            if args.recovery:
                from memory_recovery import run_recovery

                info["recovery"] = run_recovery(settings, output / "recovery")
            for policy in args.policies:
                run = output / policy
                run.mkdir()
                settings.memory.policy = policy
                settings.state_database = run / "tasks.sqlite"
                config = run / "app.json"
                config.write_text(settings.model_dump_json(indent=2))
                with http_server(
                    "agentserve.api.app:create_app",
                    run,
                    "api",
                    env={"AGENTSERVE_CONFIG": str(config)},
                ) as client:
                    records = asyncio.run(
                        run_sessions(
                            str(client.base_url),
                            conversations,
                            gold,
                            settings,
                            run,
                            args.concurrency,
                        )
                    )
                info["runs"].append(
                    {
                        "policy": policy,
                        "turns": len(records),
                        "correct": sum(r["evaluation"]["status"] == "correct" for r in records),
                    }
                )
                write(output / "summary.json", info)
            if args.regression:
                from run_g5 import run_case

                regression = settings.model_copy(deep=True)
                regression.memory = None
                info["single_turn_regression"] = run_case(
                    regression,
                    output / "v1-business",
                    json.loads(
                        (ROOT / "data/processed/business_orders_v1/questions.json").read_text()
                    ),
                    {
                        q["id"]: q
                        for q in json.loads(
                            (ROOT / "data/gold/business_orders_v1/answers.json").read_text()
                        )
                    },
                    concurrency=4,
                    warmup=False,
                )
            info["state"] = "completed"
            if not args.sessions and not args.long_probe and args.split == "development":
                write(
                    output / "freeze.json",
                    {
                        "source_sha256": sources,
                        "manifest": info["manifest"],
                        "database_sha256": before,
                        "resources": info["resources"],
                    },
                )
    except BaseException as error:
        info.update(state="interrupted_or_failed", error=repr(error))
        raise
    finally:
        info["query_databases_unchanged"] = all(
            hashlib.sha256(db.path.read_bytes()).hexdigest() == before[k]
            for k, db in settings.databases.items()
        )
        write(output / "summary.json", info)


if __name__ == "__main__":
    main()
