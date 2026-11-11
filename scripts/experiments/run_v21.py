"""V2.1 controlled session driver with a bounded audit queue and immutable run receipts."""

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

from agentserve.config import DatabaseConfig, Settings
from agentserve.domain.models import QueryResult
from agentserve.evaluation.sql import SQLEvaluator, compare
from agentserve.workloads.memory_v21 import WORKLOAD_NAME, prepare

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint():
    paths = [
        p
        for folder in ["src", "scripts", "tests", "serving", "configs"]
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".sh", ".json", ".toml", ".txt"}
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
    }


async def run_sessions(endpoint, conversations, gold, settings, output):
    evaluator = SQLEvaluator({k: v.path for k, v in settings.databases.items()})
    reference_cache = {}
    records = []
    queue = asyncio.Queue(maxsize=16)
    semaphore = asyncio.Semaphore(4)
    started = time.monotonic()
    db_hash = {
        k: hashlib.sha256(v.path.read_bytes()).hexdigest() for k, v in settings.databases.items()
    }
    async with httpx.AsyncClient(base_url=endpoint, timeout=30, trust_env=False) as client:

        async def audit():
            while True:
                conv, turn, row = await queue.get()
                audit_started = time.monotonic()
                try:
                    tid = row["task_id"]
                    context_response = await client.get(f"/tasks/{tid}/context")
                    context_response.raise_for_status()
                    steps_response = await client.get(f"/tasks/{tid}/steps")
                    steps_response.raise_for_status()
                    label = gold[turn["id"]]
                    report = row["report"] or {}
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
                        db = conv.get("database_id", "business_orders_v1")
                        key = (db_hash[db], label["sql"], label["ordered"], "strict-v1")
                        # Workload references are fixed, deterministic SELECTs on frozen databases.
                        if key not in reference_cache:
                            reference_cache[key] = await evaluator.tool.execute(
                                db, label["sql"], deadline=time.monotonic() + 4
                            )
                        reference = reference_cache[key]
                        # Cache successes only; failures must retain their actual status.
                        if reference.status != "ok" or reference.truncated:
                            reference_cache.pop(key, None)
                        prediction = (
                            await evaluator.tool.execute(
                                db, report["sql"], deadline=time.monotonic() + 4
                            )
                            if report.get("sql")
                            else QueryResult(status="NO_SQL")
                        )
                        evaluation = compare(prediction, reference, ordered=label["ordered"])
                    record = {
                        "id": turn["id"],
                        "conversation": conv["id"],
                        "question": turn["question"],
                        "database_id": conv.get("database_id", "business_orders_v1"),
                        "task": row,
                        "context": context_response.json()["context"],
                        "steps": steps_response.json(),
                        "evaluation": evaluation,
                        "tags": label["tags"],
                        "clarify": label["clarify"],
                        "provenance": {
                            k: label[k]
                            for k in ["source_turn", "distance", "family", "length"]
                            if k in label
                        },
                        "jct_seconds": row["ended_at"] - row["created_at"],
                        "offline_audit_seconds": time.monotonic() - audit_started,
                    }
                    records.append(record)
                    with (output / "turns.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    write(
                        output / "progress.json",
                        {
                            "audited": len(records),
                            "elapsed_seconds": time.monotonic() - started,
                            "last_case": turn["id"],
                        },
                    )
                finally:
                    queue.task_done()

        async def conversation(conv):
            async with semaphore:
                db = conv.get("database_id", "business_orders_v1")
                response = await client.post("/sessions", json={"database_id": db})
                response.raise_for_status()
                sid = response.json()["session_id"]
                history = []
                for turn in conv["turns"]:
                    response = await client.post(
                        "/tasks",
                        json={"database_id": db, "session_id": sid, "question": turn["question"]},
                        headers={"Idempotency-Key": turn["id"]},
                    )
                    response.raise_for_status()
                    tid = response.json()["task_id"]
                    with (output / "task-ledger.jsonl").open("a") as handle:
                        handle.write(
                            json.dumps({"id": turn["id"], "task_id": tid, "session_id": sid}) + "\n"
                        )
                    async with asyncio.timeout(settings.task_timeout_seconds + 30):
                        while True:
                            response = await client.get(f"/tasks/{tid}")
                            response.raise_for_status()
                            row = response.json()
                            if row["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                                break
                            await asyncio.sleep(0.2)
                    # Same-session commit completed. Offline audit does not block next submission.
                    await queue.put((conv, turn, row))
                    report = row["report"] or {}
                    history.append(
                        {
                            "question": turn["question"],
                            "sql": report.get("sql"),
                            "answer": report.get("answer"),
                        }
                    )
                counts = await client.post(
                    settings.memory.resources_url + "/count",
                    json={
                        "texts": [
                            json.dumps(history[:-1], ensure_ascii=False),
                            json.dumps(history, ensure_ascii=False),
                        ]
                    },
                )
                counts.raise_for_status()
                write(
                    output / (conv["id"] + "-history-tokens.json"),
                    {
                        "before_last_turn": counts.json()["counts"][0],
                        "whole_session": counts.json()["counts"][1],
                        "definition": "Canonical JSON of user questions, observed SQL and final answers; not actual model input.",
                    },
                )
                print(
                    json.dumps({"session_finished": conv["id"], "policy": settings.memory.policy}),
                    flush=True,
                )

        workers = [asyncio.create_task(audit()) for _ in range(2)]
        try:
            # TaskGroup propagates audit failures instead of hanging on queue.join.
            async with asyncio.TaskGroup() as group:
                for conv in conversations:
                    group.create_task(conversation(conv))

                async def drain():
                    await asyncio.gather(*workers)

                audit_task = group.create_task(drain())
                while len(records) < sum(len(c["turns"]) for c in conversations):
                    await asyncio.sleep(0.5)
                await queue.join()
                audit_task.cancel()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            evaluator.close()
    expected = {t["id"] for c in conversations for t in c["turns"]}
    if expected != {r["id"] for r in records} or len(records) != len(expected):
        raise RuntimeError("Missing or duplicate audit records")
    return records, time.monotonic() - started


def summarize(rows, wall):
    import numpy as np

    summary = {
        "turns": len(rows),
        "wall_seconds": wall,
        "completed": sum(r["task"]["status"] == "COMPLETED" for r in rows),
        "correct": sum(r["evaluation"]["status"] == "correct" for r in rows),
        "p50_jct": float(np.percentile([r["jct_seconds"] for r in rows], 50)),
        "p95_jct": float(np.percentile([r["jct_seconds"] for r in rows], 95)),
        "subsets": {},
        "errors": {},
    }
    for tag in ["inheritance", "correction", "long_reference", "old_version", "clarification"]:
        subset = [r for r in rows if tag in r["tags"]]
        summary["subsets"][tag] = {
            "n": len(subset),
            "correct": sum(r["evaluation"]["status"] == "correct" for r in subset),
        }
    for r in rows:
        if r["task"]["error"]:
            code = r["task"]["error"]["code"]
            summary["errors"][code] = summary["errors"].get(code, 0) + 1
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["development", "calibration", "acceptance"], default="development"
    )
    parser.add_argument("--registration", type=Path)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["state", "state_on_demand", "window", "sql_episode"],
        default=["window", "state_on_demand"],
    )
    parser.add_argument("--serving", choices=["C0", "C1"], default="C0")
    parser.add_argument("--sessions", type=int)
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--targeted-controls", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--calibration-design", type=Path)
    args = parser.parse_args()
    if args.stage == "acceptance" and (not args.registration or args.sessions):
        parser.error("Acceptance requires a frozen registration and all registered sessions")
    prepare(ROOT)
    split = "acceptance" if args.stage == "acceptance" else "development"
    conversations = json.loads((ROOT / f"data/splits/{WORKLOAD_NAME}/{split}.json").read_text())
    gold = json.loads((ROOT / f"data/gold/{WORKLOAD_NAME}/{split}.json").read_text())
    if args.calibration_design:
        design = json.loads(args.calibration_design.read_text())
        from agentserve.workloads.memory_v21 import build

        candidates, gold = build("development", development_length=design["session_length"])
        conversations = [c for c in candidates if c["family"] in design["families"]]
    if args.sessions:
        conversations = conversations[: args.sessions]
    output = (
        ROOT
        / "results/v2.1-memory"
        / (args.stage + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = fingerprint()
    settings = Settings.load(ROOT / "configs/app.v2.json")
    if args.public:
        public = json.loads((ROOT / "data/splits/sparc_v21/manifest.json").read_text())
        for key, db in public["databases"].items():
            settings.databases[key] = DatabaseConfig(
                path=ROOT / db["path"], schema_path=ROOT / db["schema_path"], context=db["context"]
            )
        conversations += json.loads((ROOT / f"data/splits/sparc_v21/{split}.json").read_text())
        gold.update(json.loads((ROOT / f"data/gold/sparc_v21/{split}.json").read_text()))
    settings.sql_prompt_version = "evidence_v21"
    settings.answer_style = "evidence_v21"
    settings.llm_concurrency = 2 if args.serving == "C0" else 4
    summary = {
        "state": "prepared",
        "stage": args.stage,
        "serving": args.serving,
        "policies": args.policies,
        "sources": sources,
        "session_count": len(conversations),
        "runs": [],
    }
    hashes = {
        k: hashlib.sha256(v.path.read_bytes()).hexdigest() for k, v in settings.databases.items()
    }
    summary["database_sha256"] = hashes
    if args.stage == "acceptance":
        registration = json.loads(args.registration.read_text())
        assert registration["sources"] == sources and registration["database_sha256"] == hashes
        assert registration["policies"] == args.policies and registration["serving"] == args.serving
        assert registration["schema_sha256"] == {
            k: hashlib.sha256(v.schema_path.read_bytes()).hexdigest()
            for k, v in settings.databases.items()
        }
        for name, expected in registration["input_files_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        assert (
            registration["inputs_sha256"]
            == hashlib.sha256(
                (ROOT / f"data/splits/{WORKLOAD_NAME}/acceptance.json").read_bytes()
            ).hexdigest()
        )
        assert (
            registration["gold_sha256"]
            == hashlib.sha256(
                (ROOT / f"data/gold/{WORKLOAD_NAME}/acceptance.json").read_bytes()
            ).hexdigest()
        )
        summary["registration_sha256"] = hashlib.sha256(args.registration.read_bytes()).hexdigest()
    write(output / "summary.json", summary)
    print(json.dumps({"output": str(output)}), flush=True)
    try:
        with ExitStack() as stack:
            stack.enter_context(monitor(output))
            cpu = stack.enter_context(
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
            settings.memory.resources_url = str(cpu.base_url).rstrip("/")
            settings.memory.resources_identity = cpu.get("/info").json()["identity"]
            summary["resources"] = cpu.get("/info").json()
            if args.stage == "acceptance":
                assert summary["resources"] == registration["resources"]
            from audit_v21_controls import verify_resources

            summary["resource_equivalence"] = verify_resources(cpu)
            settings.base_url = stack.enter_context(
                temporary_model(
                    output,
                    max_model_len=8192,
                    verify_gpu=True,
                    max_num_seqs=settings.llm_concurrency,
                )
            )
            if args.calibration_design:
                with httpx.Client(trust_env=False) as warm_client:
                    response = warm_client.post(
                        settings.base_url + "/chat/completions",
                        json={
                            "model": settings.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Return exactly this SQL: SELECT COUNT(*) FROM customers",
                                }
                            ],
                            "temperature": 0,
                            "max_tokens": 32,
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    summary["excluded_model_warmup"] = response.json()
            if args.recovery:
                from memory_recovery import run_recovery

                summary["state_recovery"] = run_recovery(
                    settings, output / "state-recovery", policy="state"
                )
            for policy in args.policies:
                run = output / policy
                run.mkdir()
                settings.memory.policy = policy
                settings.state_database = run / "tasks.sqlite"
                config = run / "app.json"
                write(config, settings.model_dump(mode="json"))
                with http_server(
                    "agentserve.api.app:create_app",
                    run,
                    "api",
                    env={"AGENTSERVE_CONFIG": str(config)},
                ) as api:
                    rows, wall = asyncio.run(
                        run_sessions(str(api.base_url), conversations, gold, settings, run)
                    )
                metrics = summarize(rows, wall)
                write(run / "metrics.json", metrics)
                summary["runs"].append({"policy": policy, **metrics})
                write(output / "summary.json", summary)
            if args.regression:
                from run_g5 import run_case

                regression = settings.model_copy(deep=True)
                regression.memory = None
                summary["single_turn"] = run_case(
                    regression,
                    output / "single-turn",
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
            if args.targeted_controls:
                from run_g5 import run_case
                from v21_targeted import controls

                jobs, labels = controls()
                summary["targeted_control_registration"] = {
                    "repeats": 3,
                    "jobs": jobs,
                    "labels": labels,
                    "stop_rule": f"all {len(jobs) * 3} attempts once; no score-based retry",
                    "role": "development controls, not unseen acceptance",
                }
                write(output / "summary.json", summary)
                summary["targeted_controls"] = []
                controls_settings = settings.model_copy(deep=True)
                controls_settings.memory = None
                for repeat in range(3):
                    summary["targeted_controls"].append(
                        run_case(
                            controls_settings,
                            output / f"targeted-{repeat}",
                            jobs,
                            labels,
                            concurrency=4,
                            warmup=False,
                        )
                    )
            summary["resource_metrics"] = cpu.get("/metrics").json()
        summary["state"] = "completed"
    except BaseException as error:
        summary.update(state="failed", error=repr(error))
        raise
    finally:
        summary["query_databases_unchanged"] = all(
            hashlib.sha256(v.path.read_bytes()).hexdigest() == hashes[k]
            for k, v in settings.databases.items()
        )
        write(output / "summary.json", summary)


if __name__ == "__main__":
    main()
