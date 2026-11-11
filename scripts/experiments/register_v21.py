"""Freeze product gates, complete input identities and answer-review IDs before acceptance."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from run_v21 import ROOT, fingerprint

from agentserve.config import DatabaseConfig, Settings
from agentserve.workloads.memory_v21 import WORKLOAD_NAME, prepare


def development_failures(development, path):
    if development["state"] != "completed":
        return [{"stage": "development_not_completed"}]
    selected = next(r for r in development["runs"] if r["policy"] == "state_on_demand")
    failures = []
    for tag, minimum in [
        ("inheritance", 0.9),
        ("correction", 0.8),
        ("old_version", 0.8),
        ("clarification", 1.0),
    ]:
        row = selected["subsets"][tag]
        if row["correct"] / row["n"] < minimum:
            failures.append({"development_subset": tag, "observed": row, "minimum": minimum})
    if development.get("single_turn", {}).get("quality", {}).get("correct", 0) < 38:
        failures.append({"single_turn": "less than38/40"})
    failed_ids = [
        row["id"]
        for row in json.loads((path / "single-turn/evaluations.json").read_text())
        if row["status"] != "correct"
    ]
    if not set(failed_ids) <= {"business.average_order", "business.no_completed"}:
        failures.append({"single_turn_new_failures": failed_ids})
    if not development.get("targeted_controls") or any(
        r["quality"]["correct"] != r["submitted"] for r in development["targeted_controls"]
    ):
        failures.append({"targeted": "not all registered controls passed"})
    rows = [json.loads(line) for line in (path / "state_on_demand/turns.jsonl").open()]
    independent = [r for r in rows if not r["tags"]]
    if (
        sum(
            r["evaluation"]["status"] == "correct" and r["task"]["status"] == "COMPLETED"
            for r in independent
        )
        / len(independent)
        < 0.95
    ):
        failures.append({"independent": "less than95% on development"})
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    args.development = args.development.resolve()
    args.calibration = args.calibration.resolve()
    development = json.loads((args.development / "summary.json").read_text())
    failures = development_failures(development, args.development)
    if failures:
        raise ValueError("Candidate not qualified for acceptance: " + json.dumps(failures))
    calibration = json.loads(args.calibration.read_text())
    assert len(calibration["runs"]) == 4
    for run in calibration["runs"]:
        receipt = json.loads((ROOT / run["path"] / "summary.json").read_text())
        assert receipt["state"] == "completed" and receipt["sources"] == fingerprint()
    prepare(ROOT)
    questions = json.loads((ROOT / f"data/splits/{WORKLOAD_NAME}/acceptance.json").read_text())
    gold = json.loads((ROOT / f"data/gold/{WORKLOAD_NAME}/acceptance.json").read_text())
    public = json.loads((ROOT / "data/splits/sparc_v21/manifest.json").read_text())
    public_gold = json.loads((ROOT / "data/gold/sparc_v21/acceptance.json").read_text())
    reviews = []

    def rank(key):
        return hashlib.sha256(("v21-review:" + key).encode()).hexdigest()

    for family in range(12):
        for group in ["inheritance", "old_version", "correction", "independent"]:
            candidates = [
                key
                for key, row in gold.items()
                if row["family"] == family
                and (group in row["tags"] if group != "independent" else not row["tags"])
                and (row["distance"] == 39 if group == "old_version" else True)
            ]
            reviews.append(min(candidates, key=rank))
    reviews += sorted(public_gold, key=rank)[:12]
    settings = Settings.load(ROOT / "configs/app.v2.json")
    for key, db in public["databases"].items():
        settings.databases[key] = DatabaseConfig(
            path=ROOT / db["path"], schema_path=ROOT / db["schema_path"], context=db["context"]
        )
    files = [
        ROOT / f"data/{folder}/{dataset}/acceptance.json"
        for folder in ["splits", "gold"]
        for dataset in [WORKLOAD_NAME, "sparc_v21"]
    ]
    files += [
        ROOT / "data/splits/sparc_v21/manifest.json",
        ROOT / f"data/splits/{WORKLOAD_NAME}/manifest.json",
    ]
    output = (
        ROOT
        / "results/v2.1-memory"
        / ("registration-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir()
    value = {
        "stage": "frozen_before_any_new_acceptance_predictions",
        "sources": fingerprint(),
        "policies": ["window", "state_on_demand"],
        "candidate_scope": "Typed state for the two registered order schemas; other schemas use the matched two-turn recent window. Public results test fallback compatibility, not cross-domain long-state gains.",
        "serving": calibration["selected"],
        "resources": json.loads(
            (ROOT / calibration["runs"][0]["path"] / "summary.json").read_text()
        )["resources"],
        "database_sha256": {
            k: hashlib.sha256(v.path.read_bytes()).hexdigest()
            for k, v in settings.databases.items()
        },
        "schema_sha256": {
            k: hashlib.sha256(v.schema_path.read_bytes()).hexdigest()
            for k, v in settings.databases.items()
        },
        "input_files_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "inputs_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(files[2].read_bytes()).hexdigest(),
        "synthetic_sessions": len(questions),
        "synthetic_turns": len(gold),
        "public_turns": len(public_gold),
        "development": str(args.development.relative_to(ROOT)),
        "calibration": str(args.calibration.relative_to(ROOT)),
        "review_ids": reviews,
        "review_rule": "same60 cases per arm; source=codex_assisted; absent answer not passed; "
        "SQL correctness, faithfulness and sufficiency separately labeled; unrevewed never assumed correct",
        "gates": {
            "recent_min": 0.90,
            "recent_noninferiority_lower95": -0.05,
            "correction_min": 0.85,
            "distant_min_each_distance": 0.80,
            "independent_min": 0.95,
            "independent_noninferiority_lower95": -0.03,
            "clarification_min": 1.0,
            "synthetic_overall_min": 0.90,
            "public_max_point_decline": 0.05,
            "whole_conversation_min_by_length": {"12": 0.5, "24": 0.25, "44": 0.25},
            "all_task_p95_seconds": 15,
            "review_business_success_min": 0.90,
            "review_faithfulness_min": 1.0,
            "review_business_scope": "Synthetic order product database; public review scores reported separately",
            "single_turn_allowed_failures": ["business.average_order", "business.no_completed"],
            "targeted_controls": "all39 correct and completed; all3 repetitions retained",
            "reliability": "four current state windows; isolation, ordering, snapshots, cancellation and atomic commit checks pass",
            "audit": "every registered task has exactly one terminal audit; query DB unchanged; all owned services closed",
        },
        "statistics": {
            "unit": "paired conversation; bootstrap clusters by synthetic family/public DB",
            "seed": 20261104,
            "bootstrap_repeats": 5000,
            "synthetic_family_clusters": 12,
            "limitations": "Dependent variants, few clusters, one synthetic DB, single acceptance run; no superiority claim required",
        },
        "precision_rationale": "72 synthetic sessions cover12 distinct task structures and36 genuine distance39 restorations "
        "(V2 had12);20 public DB sessions are supplemental cross-domain evidence. This is an engineering gate, "
        "not a powered population superiority study. Correlation is retained in clustered intervals.",
        "resource_budget": {
            "session_concurrency": 4,
            "task_concurrency": 4,
            "llm_concurrency": 2 if calibration["selected"] == "C0" else 4,
            "max_num_seqs": 2 if calibration["selected"] == "C0" else 4,
            "gpu_memory_utilization": 0.45,
            "context_tokens": 8192,
            "memory_pack_tokens": 2048,
            "sql_tokens": 256,
            "answer_tokens": 192,
        },
        "method_order": "window then state_on_demand; shared model/prefix cache; cross-method latency is descriptive",
        "stop_rule": "One complete registered acceptance; infrastructure failures retained. No result-based algorithm, "
        "threshold, sample or endpoint changes. Failing a gate means no adoption/no V3.",
    }
    path = output / "registration.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "registration": str(path),
                "serving": calibration["selected"],
                "sessions": len(questions) + 20,
                "turns_per_arm": len(gold) + len(public_gold),
                "reviews_per_arm": len(reviews),
            }
        )
    )


if __name__ == "__main__":
    main()
