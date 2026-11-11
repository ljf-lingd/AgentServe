"""Freeze and gate V3.2 on four SQLite instances plus real C1 compatibility tests."""

import argparse
import asyncio
import json
from contextlib import ExitStack
import os
from pathlib import Path

from gpu_monitor import monitor
from http_server import http_server
from model_server import temporary_model
from run_v21 import fingerprint, write
from run_v22 import verify_owned_services
from run_v31 import (
    EVALUATION as KNOWLEDGE_EVALUATION,
    FIXTURES,
    PRODUCT as KNOWLEDGE_MANIFEST,
    application,
    read,
    run_group,
    sha,
    sql_regression,
    subset_snapshot,
)
from v31_cases import DATABASE, metrics
from v32_cases import EVALUATION, PRODUCT, evaluate, metric_summary
from v32_controls import counterfactual, recovery

from agentserve.config import Settings
from agentserve.knowledge.snapshot import assemble, build_snapshot

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/app.v32.C1.json"
PROTOCOL = "v32-approved-metric-binding-1"


def knowledge_cases(stage):
    dataset = read(KNOWLEDGE_EVALUATION / f"contract-{stage}.json")
    cases = dataset if stage == "development" else dataset["cases"]
    n = 12 if stage == "development" else 24
    assert len(cases) == 2 * n and sum(c["outcome"] == "answered" for c in cases) == n
    return cases


def frozen_inputs():
    product = [*PRODUCT.parent.iterdir(), *KNOWLEDGE_MANIFEST.parent.iterdir()]
    if any(not path.is_file() or path.suffix not in {".md", ".json"} for path in product):
        raise ValueError("Unexpected product source")
    development, formal = [read(EVALUATION / f"{stage}.json") for stage in ("development", "acceptance")]
    for rows, count in ((development, 12), (formal, 24)):
        assert len({c["id"] for c in rows}) == len(rows) == count
    for stage in ("development", "acceptance"):
        knowledge_cases(stage)
    assert not ({c["family"] for c in development} & {c["family"] for c in formal})
    paths = [
        *product, *EVALUATION.glob("*.json"),
        KNOWLEDGE_EVALUATION / "contract-development.json",
        KNOWLEDGE_EVALUATION / "contract-acceptance.json", *FIXTURES.parent.glob("*.md"), FIXTURES,
        ROOT / "data/processed/business_orders_v1/questions.json",
        ROOT / "data/processed/business_orders_v1/schema.sql",
        ROOT / "data/gold/business_orders_v1/answers.json",
    ]
    return {
        "protocol": PROTOCOL, "sources": fingerprint(),
        "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)},
    }


def knowledge_regression(stage, base, snapshot, output):
    output.mkdir()
    cases = knowledge_cases(stage)
    main = [case for case in cases if case.get("scenario") != "fixture"]
    rows = run_group(base, output / "product", snapshot, "bm25", "product", main)
    conflicts = [c for c in cases if c.get("scenario") == "fixture"]
    if conflicts:
        ids = {key.split("@")[0] for case in conflicts for key in case["conflict_ids"]}
        fixture = assemble(FIXTURES)
        conflicts_path = output / "conflicts.json"
        subset_snapshot(fixture, conflicts_path, ids)
        rows.extend(run_group(base, output / "conflicts", conflicts_path, "bm25", "test", conflicts))
    write(output / "labels.json", rows)
    return metrics(rows)


def gate(summary, stage):
    n = 12 if stage == "development" else 24
    threshold = 11 if stage == "development" else 22
    knowledge = summary["knowledge_regression"]
    regression = summary["sql_regression"]
    return {
        "four_databases": len(summary["metric_by_db"]) == 4,
        "metric_results": summary["metric_all_db"]["passed"] == n
        and summary["metric_all_db"]["n"] == n,
        "latency": max(
            *(x["jct_p95"] for x in summary["metric_by_db"].values()),
            regression["jct_p95"], knowledge["jct_p95"],
        ) <= 15,
        "knowledge_compatibility": knowledge["positive_n"] == knowledge["negative_n"] == n
        and knowledge["grounded_by_labels"] >= threshold
        and knowledge["evidence_complete"] >= threshold and knowledge["negative_correct"] == n
        and not knowledge["safety_errors"],
        "sql_regression": regression["passed"] == regression["n"] == n,
        "counterfactual_and_boundaries": all(summary["counterfactual"].values()),
        "truncation": summary["truncation"],
        "recovery_and_cli": all(summary["recovery"].values()),
        "database_hashes": summary["databases_unchanged"],
        "owned_services_exited": not summary["owned_survivors"],
    }


def start_runtime(stack, base, output, summary, registration):
    stack.enter_context(monitor(output))
    cpu = stack.enter_context(http_server(
        "memory_resources:create_app", output, "cpu", app_dir=ROOT / "scripts/experiments",
        env={
            "AGENT_EMBEDDING_PATH": str(ROOT / "models/Qwen3-Embedding-0.6B"),
            "AGENT_TOKENIZER_PATH": str(ROOT / "models/Qwen2.5-7B-Instruct"),
            "CUDA_VISIBLE_DEVICES": "",
        },
        python_executable=os.environ.get("VLLM_PYTHON", str(Path.home() / "miniconda3/envs/agent-vllm/bin/python")),
        readiness_seconds=120,
    ))
    summary["resources"] = cpu.get("/info").raise_for_status().json()
    base.memory.resources_url = str(cpu.base_url).rstrip("/")
    base.memory.resources_identity = summary["resources"]["identity"]
    if registration:
        assert registration["resources"] == summary["resources"]
        assert registration["knowledge_snapshot_sha256"] == summary["knowledge_snapshot_sha256"]
    base.base_url = stack.enter_context(
        temporary_model(output, max_model_len=8192, max_num_seqs=4, verify_gpu=True)
    )
    return cpu


def final_checks(stage, base, snapshot, output, summary, cpu, cases):
    summary["knowledge_regression"] = knowledge_regression(stage, base, snapshot, output / "knowledge")
    summary["counterfactual"] = counterfactual(base, snapshot, output / "controls")
    limited = base.model_copy(update={"row_limit": 1})
    directory = output / "truncation"
    with application(limited, snapshot, "product", "bm25", directory) as api:
        grouped = [c for c in cases if c["request"].get("group_by") and "city" not in c["request"]]
        clipped = evaluate(api, grouped[:1], limited, PRODUCT, directory, row_limit=1)
    summary["truncation"] = bool(clipped) and clipped[0]["passed"] and clipped[0]["truncated"]
    summary["recovery"] = recovery(base, snapshot, output / "recovery")
    summary["cpu_metrics"] = cpu.get("/metrics").raise_for_status().json()


def run(stage, output, db_dir, registration):
    frozen = frozen_inputs()
    if registration:
        assert registration["frozen"] == frozen
    output.mkdir(parents=True)
    snapshot = ROOT / "cache/v32-knowledge-bm25.json"
    if not snapshot.exists():
        asyncio.run(build_snapshot(KNOWLEDGE_MANIFEST, snapshot))
    expected_snapshot = assemble(KNOWLEDGE_MANIFEST)
    loaded_snapshot = read(snapshot)
    loaded_snapshot.pop("snapshot_sha256")
    assert loaded_snapshot == expected_snapshot
    base = Settings.load(CONFIG)
    db_paths = {
        "primary": base.databases[DATABASE].path,
        **{name: (db_dir / f"{name}.sqlite").resolve(strict=True)
           for name in ("catalogue", "conditions", "grouping")},
    }
    db_before = {key: sha(path) for key, path in db_paths.items()}
    demo = base.databases["demo_orders"].path
    demo_before = sha(demo)
    assert len(set(db_before.values())) == 4
    summary = {
        "stage": stage, "state": "started", "frozen": frozen, "database_sha256": db_before,
        "demo_database_sha256": demo_before, "knowledge_snapshot_sha256": sha(snapshot),
        "metric_by_db": {}, "reviewer": "codex_assisted",
    }
    write(output / "summary.json", summary)
    try:
        with ExitStack() as stack:
            cpu = start_runtime(stack, base, output, summary, registration)
            n = 12 if stage == "development" else 24
            cases = read(EVALUATION / f"{stage}.json")
            labels = {}
            for name, path in db_paths.items():
                settings = base.model_copy(deep=True)
                settings.databases[DATABASE].path = path
                directory = output / "metric" / name
                with application(settings, snapshot, "product", "bm25", directory) as api:
                    labels[name] = evaluate(api, cases, settings, PRODUCT, directory)
                    if name == "primary":
                        summary["sql_regression"] = asyncio.run(
                            sql_regression(settings, str(api.base_url), n, output)
                        )
                summary["metric_by_db"][name] = metric_summary(labels[name])
                print(json.dumps({"stage": stage, "database": name, **summary["metric_by_db"][name]}), flush=True)
            summary["metric_all_db"] = {
                "n": n, "passed": sum(
                    all(labels[key][index]["passed"] for key in db_paths) for index in range(n)
                ),
            }
            final_checks(stage, base, snapshot, output, summary, cpu, cases)
        summary["state"] = "completed"
    finally:
        summary["databases_unchanged"] = db_before == {
            key: sha(path) for key, path in db_paths.items()
        } and demo_before == sha(demo)
        summary["owned_survivors"] = verify_owned_services(output)
        if summary["state"] == "completed":
            summary["gate"] = gate(summary, stage)
            summary["failed_exits"] = [k for k, value in summary["gate"].items() if not value]
            summary["can_enter_v4"] = stage == "acceptance" and not summary["failed_exits"]
        write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "register", "acceptance"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--database-dir", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        summary = read(args.development / "summary.json")
        assert summary["stage"] == "development" and summary["state"] == "completed"
        assert not summary["failed_exits"] and summary["frozen"] == frozen_inputs()
        write(args.output, {
            **{key: summary[key] for key in ("frozen", "resources", "knowledge_snapshot_sha256")},
            "development_summary_sha256": sha(args.development / "summary.json"),
        })
        print(sha(args.output))
        return
    if args.stage == "acceptance" and args.registration is None:
        parser.error("Acceptance requires a frozen registration")
    registration = read(args.registration) if args.registration else None
    summary = run(args.stage, args.output.resolve(), args.database_dir, registration)
    print(json.dumps({key: summary[key] for key in ("stage", "failed_exits", "can_enter_v4")}))
    raise SystemExit(bool(summary["failed_exits"]))


if __name__ == "__main__":
    main()
