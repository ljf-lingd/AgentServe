"""V2.2.1 qualification and one registered two-arm GPU acceptance."""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sqlite3
import time
from collections import defaultdict
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np
from gpu_monitor import monitor
from http_server import http_server
from memory_recovery import run_recovery
from model_server import temporary_model
from run_g5 import run_case
from run_v21 import ROOT, fingerprint, run_sessions, summarize, write
from v21_targeted import controls

from agentserve.config import Settings
from agentserve.evaluation.sql import SQLEvaluator
from agentserve.workloads.memory_v22 import prepare, validate_references

BASE = ROOT / "results/v2.2.1-memory"
CONFIG = ROOT / "configs/app.v221.C1.json"
ARMS = ["window", "state_on_demand"]
INSTANCES = ("primary", "catalogue", "grouping", "conditions")
RATES = {
    "independent": 0.95,
    "recent": 0.90,
    "correction": 0.85,
    "latest": 0.90,
    "clarification": 1,
    "distant_7": 0.80,
    "distant_39": 0.80,
}
EXPECTED = {
    "independent": 528,
    "recent": 24,
    "correction": 48,
    "latest": 24,
    "clarification": 24,
    "distant_7": 12,
    "distant_39": 12,
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_files(data):
    return {path.name: digest(path) for path in sorted(data.iterdir()) if path.is_file()}


def review_ids(gold):
    ids = []
    for family in range(12):
        for bucket in ("recent", "correction", "distant", "latest", "independent"):
            candidates = [
                k
                for k, label in gold.items()
                if label["family"] == family
                and label["bucket"] == bucket
                and (bucket != "distant" or label["distance"] == 39)
            ]
            ids.append(
                min(
                    candidates,
                    key=lambda k: hashlib.sha256(("v221-review-1:" + k).encode()).digest(),
                )
            )
    assert len(set(ids)) == 60
    return ids


def verify_owned_services(output):
    receipts = [output / "lifecycle.json", *output.rglob("*-lifecycle.json")]
    survivors = []
    for path in receipts:
        if not path.exists():
            continue
        record = load(path)
        pid = record.get("server_pid", record.get("pid"))
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            pass
        else:
            survivors.append({"process_group": pid, "receipt": str(path.relative_to(output))})
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", record["port"])) == 0:
                survivors.append(
                    {"listening_port": record["port"], "receipt": str(path.relative_to(output))}
                )
    return survivors


async def audit(rows, gold, databases, arm, state_path, output):
    evaluator = SQLEvaluator(databases)
    result_cache = {}
    audited = []
    started = time.monotonic()
    try:
        with closing(sqlite3.connect(state_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            activated = {x[0] for x in db.execute("SELECT task_id FROM memory_entries")}
            events = defaultdict(list)
            for task_id, kind, payload in db.execute(
                "SELECT task_id,type,payload_json FROM events WHERE type IN "
                "('SQL_COMPILED','MODEL_STARTED','MODEL_COMPLETED','MODEL_FAILED','MEMORY_PREPARED') "
                "ORDER BY event_id"
            ):
                events[task_id].append((kind, json.loads(payload)))
        for row in rows:
            label = gold[row["id"]]
            task = row["task"]
            report = task["report"] or {}
            task_events = events[task["task_id"]]
            calls = []
            phase = None
            for event, payload in task_events:
                if event == "MODEL_STARTED":
                    phase = payload["phase"]
                elif event in {"MODEL_COMPLETED", "MODEL_FAILED"}:
                    usage = payload.get("usage") or {}
                    calls.append(
                        {
                            "phase": phase,
                            "seconds": payload.get("client_seconds"),
                            "prompt": usage.get("prompt_tokens"),
                            "completion": usage.get("completion_tokens"),
                        }
                    )
            prediction = report.get("sql")
            plan = (row["context"] or {}).get("plan") or {}
            state = plan.get("state") or {}
            fields = state.get("fields", {})
            version = label["source_turn"]
            values = {key: field["value"] for key, field in fields.items()}
            operators = {key: field["operator"] for key, field in fields.items()}
            selected = (row["context"] or {}).get("selected", [])
            based_on = selected[0]["state"]["version_id"] if selected else None
            expected_version = f"turn:{version}" if version else None
            if label["clarify"]:
                passed = (
                    row["evaluation"]["status"] == "correct" and task["task_id"] not in activated
                )
                suite = {}
                state_ok = passed
            else:
                suite = {}
                for name in databases:
                    cache_key = (name, prediction, label["sql"], label["ordered"])
                    if cache_key not in result_cache:
                        result_cache[cache_key] = await evaluator.evaluate(
                            name, prediction, label["sql"], ordered=label["ordered"]
                        )
                    suite[name] = result_cache[cache_key]
                passed = (
                    task["status"] == "COMPLETED"
                    and bool(report.get("answer"))
                    and not (report.get("query_result") or {}).get("truncated")
                    and all(value["status"] == "correct" for value in suite.values())
                )
                state_ok = (
                    (
                        values == label["state"]
                        and operators == label["operators"]
                        and state.get("removed_fields") == label["removed"]
                        and based_on == expected_version
                        and (task["task_id"] in activated) == (task["status"] == "COMPLETED")
                    )
                    if arm == "state_on_demand"
                    else None
                )
            key = label["bucket"]
            if key == "distant":
                key = f"distant_{label['distance']}"
            audited.append(
                {
                    "id": row["id"],
                    "conversation": row["conversation"],
                    "bucket": key,
                    "length": label["length"],
                    "family": label["family"],
                    "passed": passed,
                    "state_ok": state_ok,
                    "state_actual": values,
                    "state_operators": operators,
                    "state_source": based_on,
                    "primary": row["evaluation"],
                    "test_suite": suite,
                    "activated": task["task_id"] in activated,
                    "model_calls": report.get("model_calls", 0),
                    "memory_calls": report.get("memory_calls", 0),
                    "status": task["status"],
                    "error": task["error"],
                    "jct_seconds": row["jct_seconds"],
                    "calls": calls,
                    "memory_prepare_seconds": sum(
                        p["client_seconds"]
                        for event, p in task_events
                        if event == "MEMORY_PREPARED"
                    ),
                    "route": "compiled"
                    if any(event == "SQL_COMPILED" for event, _ in task_events)
                    else "model_or_clarify",
                }
            )
    finally:
        evaluator.close()
    write(output / "test-suite.json", audited)
    return audited, time.monotonic() - started


def acceptance_metrics(audited):
    groups = defaultdict(list)
    sessions = defaultdict(list)
    for row in audited:
        groups[row["bucket"]].append(row)
        sessions[row["conversation"]].append(row)
    buckets = {
        key: {
            "n": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "primary": sum(r["primary"]["status"] == "correct" for r in rows),
        }
        for key, rows in groups.items()
    }
    lengths = defaultdict(list)
    for rows in sessions.values():
        lengths[rows[0]["length"]].append(all(r["passed"] for r in rows))
    nonclarify = [row for row in audited if row["bucket"] != "clarification"]
    successful = [r["jct_seconds"] for r in audited if r["passed"]]
    phases = defaultdict(list)
    for row in audited:
        for call in row["calls"]:
            phases[call["phase"]].append(call)
    costs = {}
    for phase, calls in phases.items():
        known = [c for c in calls if c["prompt"] is not None and c["completion"] is not None]
        costs[phase] = {
            "attempts": len(calls),
            "known_usage": len(known),
            "unknown_usage": len(calls) - len(known),
            "prompt_tokens_known": sum(c["prompt"] for c in known),
            "completion_tokens_known": sum(c["completion"] for c in known),
            "client_seconds": sum(c["seconds"] for c in calls if c["seconds"] is not None),
        }
    return {
        "buckets": buckets,
        "sessions": {n: {"n": len(rows), "passed": sum(rows)} for n, rows in lengths.items()},
        "state_correct": sum(row["state_ok"] is True for row in audited),
        "state_wrong": sum(row["state_ok"] is False for row in audited),
        "completed": sum(row["status"] == "COMPLETED" for row in nonclarify),
        "nonclarification_n": len(nonclarify),
        "attempts": sum(row["model_calls"] for row in audited),
        "memory_attempts": sum(row["memory_calls"] for row in audited),
        "memory_prepare_seconds": sum(row["memory_prepare_seconds"] for row in audited),
        "model_cost_by_phase": costs,
        "routes": {
            route: sum(r["route"] == route for r in audited)
            for route in ("compiled", "model_or_clarify")
        },
        "successful_p95_jct": float(np.percentile(successful, 95)) if successful else None,
    }


def qualification_counts(path):
    rows = load(path / "evaluations.json")
    if isinstance(rows, dict):
        rows = list(rows.values())
    return {
        "n": len(rows),
        "correct": sum(r["status"] == "correct" for r in rows),
        "completed": load(path / "metrics.json")["completed"],
    }


def paired_memory_interval(output):
    window, state = (
        {row["id"]: row for row in load(output / arm / "test-suite.json")} for arm in ARMS
    )
    assert window.keys() == state.keys()
    paired = defaultdict(list)
    for key, row in state.items():
        if row["bucket"] not in {"clarification", "independent"}:
            paired[row["family"]].append(int(row["passed"]) - int(window[key]["passed"]))
    assert len(paired) == 12 and sum(map(len, paired.values())) == 120
    family_effects = [np.mean(paired[f]) for f in sorted(paired)]
    samples = (
        np.random.default_rng(2212).choice(family_effects, size=(10000, len(paired))).mean(axis=1)
    )
    return {
        "n_families": 12,
        "n_paired_turns": 120,
        "delta": float(np.mean(family_effects)),
        "bootstrap_95pct": list(map(float, np.percentile(samples, [2.5, 97.5]))),
    }


def regressions(settings, output):
    regression = settings.model_copy(deep=True)
    regression.memory = None
    jobs = load(ROOT / "data/processed/business_orders_v1/questions.json")
    labels = {q["id"]: q for q in load(ROOT / "data/gold/business_orders_v1/answers.json")}
    run_case(regression, output / "single-turn", jobs, labels, concurrency=4, warmup=False)
    jobs, labels = controls()
    counts = {}
    for repeat in range(3):
        path = output / f"targeted-{repeat}"
        run_case(regression, path, jobs, labels, concurrency=4, warmup=False)
        counts[str(repeat)] = qualification_counts(path)
    return qualification_counts(output / "single-turn"), counts


def unit_probes(settings, output):
    cases = [
        (
            "revenue",
            "统计completed订单营收合计，按历史成交价计算。只输出一列，用 SQL 别名 metric，单位分。",
            "SELECT SUM(i.quantity*i.unit_price_cents) FROM order_items i JOIN orders o USING(order_id) WHERE o.status='completed'",
            True,
        ),
        (
            "average",
            "计算 Office 目录商品当前标价的平均值，单位分。只输出一列，用 SQL 别名 metric。",
            "SELECT AVG(price_cents) FROM products WHERE category='Office'",
            True,
        ),
        (
            "count",
            "统计目录中标价非空的商品数量。只输出一列，用 SQL 别名 metric。",
            "SELECT COUNT(price_cents) FROM products",
            False,
        ),
    ]
    probe_settings = settings.model_copy(deep=True)
    probe_settings.memory = None
    jobs = [
        {"id": key, "database_id": "business_orders_v1", "question": question}
        for key, question, _, _ in cases
    ]
    labels = {key: {"sql": sql, "ordered": True} for key, _, sql, _ in cases}
    run_case(probe_settings, output, jobs, labels, concurrency=4, warmup=False)
    rows = {row["id"]: row for row in load(output / "evaluations.json")}
    checks = []
    for i, (key, _, _, monetary) in enumerate(cases):
        task = load(output / f"tasks/{i:04d}.json")["task"]
        report = task["report"] or {}
        answer = report.get("answer") or ""
        columns = (report.get("query_result") or {}).get("columns", [])
        checks.append(
            {
                "id": key,
                "passed": task["status"] == "COMPLETED"
                and rows[key]["status"] == "correct"
                and [c.lower() for c in columns] == ["metric"]
                and ("金额列（第1列）的单位为分" in answer) == monetary,
                "columns": columns,
                "sql": report.get("sql"),
                "answer": answer,
            }
        )
    return checks


def register(output, data, development):
    qualification = load(development / "summary.json")
    current = fingerprint()
    if not qualification["qualified"] or current != qualification["sources"]:
        raise ValueError(
            "Registration requires qualified runtime, workload and shared execution sources."
        )
    paths = {name: data / f"{name}.sqlite" for name in INSTANCES}
    gold = load(data / "gold.json")
    validate_references(gold, paths)
    record = {
        "revision": "2.2.1",
        "sources": current,
        "config_sha256": digest(CONFIG),
        "arms": ARMS,
        "stage_inputs": input_files(data),
        "schema_sha256": digest(ROOT / "data/processed/business_orders_v1/schema.sql"),
        "model_manifest_sha256": digest(ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json"),
        "launch_sha256": digest(ROOT / "serving/vllm/launch-qwen.sh"),
        "resources": qualification["resources"],
        "review_ids": review_ids(gold),
        "expected": EXPECTED,
        "min_rates": RATES,
        "thresholds": {
            "p95_jct": 15,
            "completed": 0.95,
            "review_correct": 54,
            "review_faithful": 60,
            "state": 1,
            "whole_12": 0.50,
            "whole_44": 0.25,
        },
        "serving": {
            "task_concurrency": 4,
            "llm_concurrency": 4,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.45,
            "max_model_len": 8192,
        },
        "design": {
            "arm_order": ARMS,
            "model_warmup": "no excluded query; engine reused across fixed arm order",
            "rule": "one formal prediction per arm; stop for hard infrastructure failure; no score-based rerun",
            "family_bootstrap": {"seed": 2212, "replicates": 10000, "primary_memory_turns": 120},
        },
        "development_summary_sha256": digest(development / "summary.json"),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)


def acceptance_failures(summary):
    metrics = summary["runs"]["state_on_demand"]
    failures = []
    for bucket, threshold in RATES.items():
        data = metrics["buckets"][bucket]
        if data["n"] != EXPECTED[bucket] or data["passed"] / data["n"] < threshold:
            failures.append(bucket)
    if metrics["state_wrong"] or metrics["state_correct"] != 672:
        failures.append("state_semantics")
    if metrics["completed"] / metrics["nonclarification_n"] < 0.95:
        failures.append("nonclarification_coverage")
    for length, threshold in ((12, 0.5), (44, 0.25)):
        group = metrics["sessions"][length]
        if group["n"] != 12 or group["passed"] / group["n"] < threshold:
            failures.append(f"whole_session_{length}")
    if metrics["p95_jct"] > 15:
        failures.append("jct")
    if not all(case["passed"] for case in summary["recovery"]):
        failures.append("recovery")
    if not summary["cli"]["passed"]:
        failures.append("cli")
    return failures


def make_review_packet(output, run, gold, ids, rows):
    selected = {r["id"]: r for r in rows if r["id"] in ids}
    assert set(selected) == set(ids)
    packet = []
    for case_id in ids:
        row = selected[case_id]
        report = row["task"]["report"] or {}
        packet.append(
            {
                "id": case_id,
                "question": row["question"],
                "sql": report.get("sql"),
                "answer": report.get("answer"),
                "tool_result": report.get("query_result"),
                "reference_sql": gold[case_id]["sql"],
                "primary_evaluation": row["evaluation"],
            }
        )
    write(output / f"review-{run}.json", packet)


def run(stage, data, output, registration=None):
    settings = Settings.load(CONFIG)
    before = input_files(data)
    legacy_db = settings.databases["business_orders_v1"].path
    legacy_hash = digest(legacy_db)
    databases = {name: data / f"{name}.sqlite" for name in INSTANCES}
    conversations, gold = load(data / "questions.json"), load(data / "gold.json")
    validate_references(gold, databases)
    sources = fingerprint()
    if registration:
        assert registration["sources"] == sources and registration["stage_inputs"] == before
        assert registration["config_sha256"] == digest(CONFIG)
        assert registration["arms"] == ARMS
    output.mkdir()
    summary = {
        "state": "started",
        "stage": stage,
        "sources": sources,
        "runs": {},
        "qualified": False,
    }
    print(json.dumps({"output": str(output), "stage": stage}), flush=True)
    try:
        with ExitStack() as stack:
            stack.enter_context(monitor(output))
            cpu = stack.enter_context(
                http_server(
                    "memory_resources:create_app",
                    output,
                    "cpu",
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
            summary["resources"] = cpu.get("/info").json()
            if registration:
                assert summary["resources"] == registration["resources"]
            settings.memory.resources_url = str(cpu.base_url).rstrip("/")
            settings.memory.resources_identity = summary["resources"]["identity"]
            settings.base_url = stack.enter_context(
                temporary_model(output, max_model_len=8192, max_num_seqs=4, verify_gpu=True)
            )
            if stage == "development":
                summary["single_turn"], summary["controls"] = regressions(settings, output)
                summary["unit_probes"] = unit_probes(settings, output / "unit-probes")
            settings.databases["business_orders_v1"].path = databases["primary"]
            for arm in ARMS if registration else ARMS[-1:]:
                path = output / arm
                path.mkdir()
                settings.memory.policy = arm
                settings.state_database = path / "tasks.sqlite"
                config = path / "app.json"
                write(config, settings.model_dump(mode="json"))
                with http_server(
                    "agentserve.api.app:create_app",
                    path,
                    "api",
                    env={"AGENTSERVE_CONFIG": str(config)},
                ) as api:
                    rows, wall = asyncio.run(
                        run_sessions(str(api.base_url), conversations, gold, settings, path)
                    )
                audited, audit_wall = asyncio.run(
                    audit(rows, gold, databases, arm, settings.state_database, path)
                )
                summary["runs"][arm] = {
                    **summarize(rows, wall),
                    **acceptance_metrics(audited),
                    "suite_audit_wall_seconds": audit_wall,
                }
                if registration:
                    make_review_packet(output, arm, gold, registration["review_ids"], rows)
                write(output / "summary.json", summary)
            if registration:
                from v22_product import run_cli

                summary["paired_memory_effect"] = paired_memory_interval(output)
                summary["recovery"] = run_recovery(
                    settings, output / "state-recovery", policy="state_on_demand"
                )
                summary["cli"] = run_cli(settings, output / "cli", databases["conditions"])
                summary["failed_exits"] = acceptance_failures(summary)
            else:
                metrics = summary["runs"]["state_on_demand"]
                summary["qualified"] = (
                    summary["single_turn"] == {"n": 40, "correct": 40, "completed": 40}
                    and all(
                        x == {"n": 13, "correct": 13, "completed": 13}
                        for x in summary["controls"].values()
                    )
                    and all(probe["passed"] for probe in summary["unit_probes"])
                    and metrics["state_correct"] == metrics["turns"] == 64
                    and metrics["routes"]["compiled"] == 64
                    and all(x["n"] == x["passed"] for x in metrics["buckets"].values())
                )
            summary["resource_metrics"] = cpu.get("/metrics").json()
            summary["state"] = "awaiting_review" if registration else "completed"
    except BaseException as error:
        summary.update(state="failed", error=repr(error), qualified=False)
        raise
    finally:
        summary["query_databases_unchanged"] = (
            input_files(data) == before and digest(legacy_db) == legacy_hash
        )
        summary["owned_service_survivors"] = verify_owned_services(output)
        if summary["owned_service_survivors"] or not summary["query_databases_unchanged"]:
            summary.update(state="integrity_failure", qualified=False)
        write(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "development", "register", "acceptance"])
    parser.add_argument("--batch", type=int, choices=[1, 2], default=1)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    frozen = BASE / "frozen"
    registration = BASE / "registration.json"
    if args.stage == "prepare":
        for split in ("development", "acceptance"):
            print(json.dumps({"split": split, "sha256": prepare(ROOT, frozen / split, split)}))
    elif args.stage == "development":
        if args.batch == 2 and not (BASE / "development-1/summary.json").exists():
            parser.error("First development batch receipt is required.")
        run("development", frozen / "development", BASE / f"development-{args.batch}")
    elif args.stage == "register":
        if not args.qualification:
            parser.error("--qualification is required.")
        register(registration, frozen / "acceptance", args.qualification)
    else:
        run("acceptance", frozen / "acceptance", BASE / "acceptance", load(registration))


if __name__ == "__main__":
    main()
