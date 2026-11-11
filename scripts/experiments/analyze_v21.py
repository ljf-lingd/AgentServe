"""Offline V2.1 complete denominators, clustered pairing, cost and review evidence."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from analyze_memory import distribution, evidence, summarize, turn_success


def subset(row):
    tags = row["tags"]
    if row["clarify"]:
        return "clarification"
    if "public_sparc" in tags:
        return "public_sparc"
    if "inheritance" in tags:
        return "recent_inheritance"
    if "correction" in tags:
        return "correction"
    if (row.get("provenance", {}).get("distance") or 0) >= 7:
        return "distant_recovery"
    if "long_reference" in tags:
        return "near_recovery"
    return "independent"


def score(rows):
    return {
        "total": len(rows),
        "sql_correct": sum(r["evaluation"]["status"] == "correct" for r in rows),
        "workflow_correct": sum(turn_success(r) for r in rows),
    }


def paired_cluster(left, right, kind):
    a = {r["id"]: r for r in left if not r["clarify"] and (kind == "overall" or subset(r) == kind)}
    b = {r["id"]: r for r in right if r["id"] in a}
    assert a.keys() == b.keys()
    groups = defaultdict(list)
    for key, row in a.items():
        family = row.get("provenance", {}).get("family", row["conversation"])
        cluster = ("public:" if subset(row) == "public_sparc" else "synthetic:") + str(family)
        groups[cluster].append(int(turn_success(b[key])) - int(turn_success(row)))
    means = [sum(v) / len(v) for v in groups.values()]
    if not means:
        return None
    rng = random.Random(20261104)
    estimates = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(5000))
    return {
        "clusters": len(means),
        "turns": len(a),
        "endpoint": "SQL correct and task completed",
        "cluster_equal_mean_difference": sum(means) / len(means),
        "bootstrap95": [estimates[125], estimates[4874]],
        "seed": 20261104,
        "repetitions": 5000,
        "limitation": "Few synthetic families; cluster bootstrap is descriptive and may be unstable.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    root = Path(__file__).resolve().parents[2]
    info = json.loads((run / "summary.json").read_text())
    output = run / "derived-v21"
    output.mkdir(exist_ok=False)
    groups = {
        p: [json.loads(x) for x in (run / p / "turns.jsonl").open()] for p in info["policies"]
    }
    metrics = {}
    all_evidence = []
    for policy, rows in groups.items():
        metric = summarize(rows, run / policy / "tasks.sqlite")
        metric.pop("long_session_sql", None)
        metric.pop("39_turn_reference", None)
        metric["mutually_exclusive"] = {
            k: score([r for r in rows if subset(r) == k])
            for k in [
                "independent",
                "recent_inheritance",
                "correction",
                "distant_recovery",
                "near_recovery",
                "public_sparc",
                "clarification",
            ]
        }
        distances = sorted(
            {
                r.get("provenance", {}).get("distance")
                for r in rows
                if r.get("provenance", {}).get("distance") is not None
            }
        )
        metric["actual_distance"] = {
            str(d): score([r for r in rows if r.get("provenance", {}).get("distance") == d])
            for d in distances
        }
        lengths = sorted(
            {
                r.get("provenance", {}).get("length")
                for r in rows
                if r.get("provenance", {}).get("length")
            }
        )
        metric["lengths"] = {
            str(n): score([r for r in rows if r.get("provenance", {}).get("length") == n])
            for n in lengths
        }
        metric["memory_calls"] = {
            str(n): sum((r["task"]["report"] or {}).get("memory_calls") == n for r in rows)
            for n in range(3)
        }
        metric["route_counts"] = {
            route: sum(
                (r["context"] or {}).get("plan", {}).get("route") == route
                for r in rows
                if (r["context"] or {}).get("plan")
            )
            for route in ["explicit_independent", "recent_window"]
        }
        metric["success_jct_seconds"] = distribution(
            [r["jct_seconds"] for r in rows if turn_success(r)]
        )
        metrics[policy] = metric
    comparisons = {}
    if "window" in groups:
        for policy, rows in groups.items():
            if policy == "window":
                continue
            comparisons[policy] = {
                kind: paired_cluster(groups["window"], rows, kind)
                for kind in [
                    "overall",
                    "recent_inheritance",
                    "correction",
                    "distant_recovery",
                    "independent",
                    "public_sparc",
                ]
            }
    registration = json.loads(args.registration.read_text()) if args.registration else {}
    review_ids = registration.get("review_ids", [])
    gold = {}
    split = "acceptance" if info["stage"] == "acceptance" else "development"
    for folder in ["memory_v21", "memory_v21_r2", "sparc_v21"]:
        path = root / "data/gold" / folder / f"{split}.json"
        if path.exists():
            gold.update(json.loads(path.read_text()))
    for policy, rows in groups.items():
        for row in rows:
            if row["id"] in review_ids:
                item = evidence(row, policy, gold)
                item["resolved_context"] = (row["context"] or {}).get("text")
                all_evidence.append(item)
    result = {
        "state": info["state"],
        "metrics": metrics,
        "paired": comparisons,
        "source_run": str(run.relative_to(root)),
        "registration_sha256": info.get("registration_sha256"),
        "reviews_status": "unreviewed_until_attributed_local_review",
        "no_claim_of_official_sparc_test_suite_accuracy": True,
    }
    (output / "compact.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / "answer-review-evidence.json").write_text(
        json.dumps(all_evidence, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "scores": {p: m["mutually_exclusive"] for p, m in metrics.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
