"""Offline V2 evidence inventory. Never invokes models or changes historical runs."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/v2-memory/acceptance-20261103T085452Z"
OUT = ROOT / "results/v2.1-memory/diagnosis-20261104"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    for policy in ["window", "simplemem", "sql_episode", "no_binding", "no_updates"]:
        groups[policy] = {
            r["id"]: r for r in map(json.loads, (OLD / policy / "turns.jsonl").open())
        }
    evidence = []
    for case_id, f in groups["sql_episode"].items():
        if not (
            "inheritance" in f["tags"]
            or "long_reference" in f["tags"]
            or (
                not f["tags"]
                and f["evaluation"]["status"] != groups["window"][case_id]["evaluation"]["status"]
            )
        ):
            continue
        comparison = {}
        for policy, rows in groups.items():
            r = rows[case_id]
            comparison[policy] = {k: r[k] for k in ["question", "evaluation", "context", "steps"]}
            comparison[policy]["report"] = r["task"].get("report")
            comparison[policy]["error"] = r["task"].get("error")
            comparison[policy]["task_id"] = r["task"]["task_id"]
        evidence.append({"id": case_id, "tags": f["tags"], "comparison": comparison})
    (OUT / "case-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    compact = {}
    for policy, rows in groups.items():
        compact[policy] = {}
        for tag in ["inheritance", "long_reference", "correction"]:
            subset = [r for r in rows.values() if tag in r["tags"]]
            compact[policy][tag] = {
                "n": len(subset),
                "correct": sum(r["evaluation"]["status"] == "correct" for r in subset),
                "errors": dict(
                    Counter((r["task"].get("error") or {}).get("code", "none") for r in subset)
                ),
            }
    (OUT / "compact.json").write_text(json.dumps(compact, indent=2))
    print(json.dumps(compact))
    for case in evidence:
        if "inheritance" not in case["tags"]:
            continue
        f = case["comparison"]["sql_episode"]
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "correct": f["evaluation"]["status"],
                    "error": f["error"],
                    "context": f["context"],
                    "steps": f["steps"],
                },
                ensure_ascii=False,
            )[:14000]
        )
        break


if __name__ == "__main__":
    main()
