"""Freeze the monthly brief and verify four database instances on the C1 stack."""

import argparse
import asyncio
from contextlib import ExitStack
from pathlib import Path

from run_v21 import write
from run_v22 import verify_owned_services
from run_v31 import application, read, sha, sql_regression
from run_v32 import frozen_inputs, knowledge_regression, start_runtime
from v4_cases import CASES, DATABASE, ROOT, evaluate, stats
from v4_controls import boundaries, recovery

from agentserve.config import Settings

CONFIG = ROOT / "configs/app.v4.C1.json"


def inputs():
    result = frozen_inputs()
    dev, formal = [read(CASES / f"{stage}.json") for stage in ("development", "acceptance")]
    assert len(dev) == 12 and len(formal) == 24
    assert not ({c["family"] for c in dev} & {c["family"] for c in formal})
    assert len({tuple(sorted(c["request"].items())) for c in dev + formal}) == 36
    result["protocol"] = "v4-monthly-brief-1"
    result["inputs"].update({p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(CASES.glob("*.json"))})
    return result


def run(stage, output, db_dir, registration):
    cases = read(CASES / f"{stage}.json")
    frozen = inputs()
    if registration:
        assert registration["frozen"] == frozen
    settings = Settings.load(CONFIG)
    snapshot = settings.knowledge.snapshot_path
    databases = {
        "primary": settings.databases[DATABASE].path,
        **{key: (db_dir / f"{key}.sqlite").resolve(strict=True)
           for key in ("catalogue", "conditions", "grouping")},
    }
    before = {key: sha(path) for key, path in databases.items()}
    assert len(set(before.values())) == 4
    demo = settings.databases["demo_orders"].path
    if registration:
        assert before == registration["formal_db"]
    output.mkdir(parents=True)
    summary = {
        "stage": stage, "state": "running", "frozen": frozen,
        "database_sha256": before, "demo_sha256": sha(demo),
        "knowledge_snapshot_sha256": sha(snapshot), "by_db": {},
    }
    try:
        with ExitStack() as stack:
            cpu = start_runtime(stack, settings, output, summary, registration)
            results = {}
            for key, path in databases.items():
                directory = output / key
                variant = settings.model_copy(deep=True)
                variant.databases[DATABASE].path = path
                with application(variant, snapshot, "product", "bm25", directory) as api:
                    labels = evaluate(api, variant, cases, directory)
                    if key == "primary":
                        summary["sql"] = asyncio.run(
                            sql_regression(variant, str(api.base_url), len(cases), output)
                        )
                results[key] = {row["id"] for row in labels if row["passed"]}
                summary["by_db"][key] = stats(labels)
                print(key, summary["by_db"][key], flush=True)
            summary["four_db"] = {"n": len(cases), "passed": len(set.intersection(*results.values()))}
            summary["knowledge"] = knowledge_regression(stage, settings, snapshot, output / "knowledge")
            summary["boundaries"] = boundaries(settings, snapshot, output / "boundaries")
            summary["recovery"], summary["downloads"] = recovery(settings, snapshot, output / "recovery")
            summary["cpu_metrics"] = cpu.get("/metrics").raise_for_status().json()
        summary["state"] = "completed"
    finally:
        summary["databases_unchanged"] = (
            before == {key: sha(path) for key, path in databases.items()}
            and summary["demo_sha256"] == sha(demo)
        )
        summary["owned_survivors"] = verify_owned_services(output)
        if summary["state"] == "completed":
            n = len(cases)
            k = summary["knowledge"]
            summary["gate"] = {
                "four_db": summary["four_db"]["passed"] == n,
                "brief_p95": max(v["jct_seconds_p95"] for v in summary["by_db"].values()) <= 2,
                "legacy_sql": summary["sql"]["passed"] == n and summary["sql"]["jct_p95"] <= 15,
                "legacy_knowledge": k["positive_n"] == k["negative_n"] == k["negative_correct"] == n
                and min(k["grounded_by_labels"], k["evidence_complete"]) >= (11 if n == 12 else 22)
                and not k["safety_errors"] and k["jct_p95"] <= 15,
                "boundaries": all(summary["boundaries"].values()),
                "recovery": all(summary["recovery"].values()),
                "artifacts": bool(summary["downloads"]),
                "database_hashes": summary["databases_unchanged"],
                "services_exited": not summary["owned_survivors"],
            }
            summary["passed"] = all(summary["gate"].values())
        write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "register", "acceptance"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        dev = read(args.development / "summary.json")
        assert dev["stage"] == "development" and dev["passed"] and dev["frozen"] == inputs()
        config = Settings.load(CONFIG)
        formal_db = {
            "primary": sha(config.databases[DATABASE].path),
            **{key: sha(args.database_dir / f"{key}.sqlite") for key in ("catalogue", "conditions", "grouping")},
        }
        write(args.output, {
            **{key: dev[key] for key in ("frozen", "resources", "knowledge_snapshot_sha256")},
            "formal_db": formal_db,
            "development_sha256": sha(args.development / "summary.json"),
        })
        print(sha(args.output), flush=True)
        return
    if args.stage == "acceptance" and args.registration is None:
        parser.error("Acceptance requires a frozen registration")
    registration = read(args.registration) if args.registration else None
    summary = run(args.stage, args.output.resolve(), args.database_dir, registration)
    print(summary["gate"], flush=True)
    raise SystemExit(not summary["passed"])


if __name__ == "__main__":
    main()
