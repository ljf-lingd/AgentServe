"""Apply registered non-answer product gates; attributed answer reviews are merged locally."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_memory import turn_success
from analyze_v21 import subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    reg = json.loads(args.registration.read_text())
    summary = json.loads((run / "summary.json").read_text())
    derived = json.loads((run / "derived-v21/compact.json").read_text())
    policy = "state_on_demand"
    rows = {
        p: [json.loads(line) for line in (run / p / "turns.jsonl").open()] for p in reg["policies"]
    }
    selected = rows[policy]
    criteria = reg["gates"]
    checks = []

    def check(name, passed, evidence):
        checks.append({"gate": name, "passed": bool(passed), "evidence": evidence})

    def accuracy(data):
        return sum(turn_success(r) for r in data) / len(data) if data else 0

    for kind, key in [
        ("recent_inheritance", "recent_min"),
        ("correction", "correction_min"),
        ("independent", "independent_min"),
    ]:
        data = [r for r in selected if subset(r) == kind]
        check(
            kind,
            accuracy(data) >= criteria[key],
            {
                "correct": sum(turn_success(r) for r in data),
                "n": len(data),
                "minimum": criteria[key],
            },
        )
    for kind, key in [
        ("recent_inheritance", "recent_noninferiority_lower95"),
        ("independent", "independent_noninferiority_lower95"),
    ]:
        paired = derived["paired"][policy][kind]
        check(
            kind + "_noninferiority", paired and paired["bootstrap95"][0] >= criteria[key], paired
        )
    for distance in [7, 19, 39]:
        data = [r for r in selected if r.get("provenance", {}).get("distance") == distance]
        check(
            "restore_distance_" + str(distance),
            accuracy(data) >= criteria["distant_min_each_distance"],
            {"correct": sum(turn_success(r) for r in data), "n": len(data)},
        )
    clarifications = [r for r in selected if r["clarify"]]
    check(
        "clarification",
        accuracy(clarifications) >= criteria["clarification_min"],
        {"correct": sum(turn_success(r) for r in clarifications), "n": len(clarifications)},
    )
    business = [r for r in selected if not r["clarify"] and subset(r) != "public_sparc"]
    check(
        "synthetic_overall",
        accuracy(business) >= criteria["synthetic_overall_min"],
        {"correct": sum(turn_success(r) for r in business), "n": len(business)},
    )
    public = {p: [r for r in group if subset(r) == "public_sparc"] for p, group in rows.items()}
    public_delta = accuracy(public[policy]) - accuracy(public["window"])
    check(
        "public_point_guard",
        public_delta >= -criteria["public_max_point_decline"],
        {
            "candidate_minus_window": public_delta,
            "window": accuracy(public["window"]),
            "candidate": accuracy(public[policy]),
        },
    )
    sessions = defaultdict(list)
    for row in selected:
        if subset(row) != "public_sparc":
            sessions[row["conversation"]].append(row)
    for length, minimum in criteria["whole_conversation_min_by_length"].items():
        chosen = [r for r in sessions.values() if len(r) == int(length)]
        successes = sum(all(turn_success(r) for r in group) for group in chosen)
        check(
            "whole_conversation_" + length,
            bool(chosen) and successes / len(chosen) >= minimum,
            {"correct": successes, "n": len(chosen), "minimum": minimum},
        )
    single = json.loads((run / "single-turn/evaluations.json").read_text())
    failed = [r["id"] for r in single if r["status"] != "correct"]
    check(
        "single_turn_no_new_failure",
        len(single) == 40 and set(failed) <= set(criteria["single_turn_allowed_failures"]),
        {"failed_ids": failed, "n": len(single)},
    )
    controls = summary.get("targeted_controls", [])
    check(
        "targeted39",
        len(controls) == 3
        and all(
            r["submitted"] == 13 and r["completed"] == 13 and r["quality"]["correct"] == 13
            for r in controls
        ),
        {
            "runs": [
                {
                    "n": r["submitted"],
                    "correct": r["quality"]["correct"],
                    "completed": r["completed"],
                }
                for r in controls
            ]
        },
    )
    recovery = summary.get("state_recovery", [])
    check("state_recovery", len(recovery) == 4 and all(r["passed"] for r in recovery), recovery)
    metric = derived["metrics"][policy]
    check(
        "p95",
        metric["jct_server_seconds"]["p95"] <= criteria["all_task_p95_seconds"],
        metric["jct_server_seconds"],
    )
    expected = reg["synthetic_turns"] + reg["public_turns"]
    check(
        "complete_audit",
        all(
            len(group) == expected and len({r["id"] for r in group}) == expected
            for group in rows.values()
        ),
        {p: len(group) for p, group in rows.items()},
    )
    check(
        "query_database_unchanged",
        summary.get("query_databases_unchanged"),
        summary.get("database_sha256"),
    )
    lifecycle = json.loads((run / "lifecycle.json").read_text())
    closed = lifecycle.get("temporary_model_closed") and all(
        json.loads(p.read_text()).get("closed") for p in run.rglob("*-lifecycle.json")
    )
    check(
        "owned_services_closed",
        closed,
        {"temporary_model_closed": lifecycle.get("temporary_model_closed")},
    )
    product_index = run / "derived-v21/product-index.json"
    product = (
        json.loads(
            (Path(json.loads(product_index.read_text())["path"]) / "summary.json").read_text()
        )
        if product_index.exists()
        else {}
    )
    check(
        "actual_cli_restart_and_relative_date",
        product.get("state") == "passed"
        and product.get("relative_frame_correct")
        and product.get("cross_database_status") == 409
        and product.get("query_database_unchanged"),
        {
            k: product.get(k)
            for k in [
                "state",
                "product_sql_correct",
                "cli_exit_codes",
                "relative_frame_correct",
                "relative_expected_range",
                "cross_database_status",
                "query_database_unchanged",
            ]
        },
    )
    value = {
        "non_answer_gates_passed": all(r["passed"] for r in checks),
        "checks": checks,
        "answer_review": "pending local attributed review; unreviewed never passed",
        "can_enter_v3": False,
        "scope": "Only V2.1. Even a passing final decision does not start V3.",
    }
    (run / "derived-v21/preliminary-gate.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "non_answer_gates_passed": value["non_answer_gates_passed"],
                "failed": [r["gate"] for r in checks if not r["passed"]],
            }
        )
    )


if __name__ == "__main__":
    main()
