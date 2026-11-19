"""Summarize full submissions, paired latency and representative failure receipts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from run_v5_phase import PAIRS
from summarize_v5_phase import lifecycle, matching_requests
from v5_expansion_live import groups, index_cases
from v5_workbench_contract import check
from v6_phase_benchmark import distribution, dump, file_sha


def read(path):
    return json.loads(path.read_bytes())


def deltas(pairs):
    return distribution(b["jct"] - a["jct"] for a, b in pairs
                        if a["jct"] is not None and b["jct"] is not None)


def compare(left, right):
    pairs = [(row, right["cases"][key]) for key, row in left["cases"].items()]
    both_correct = [(a, b) for a, b in pairs if a["correct"] and b["correct"]]
    session_sums = {}
    for a, b in pairs:
        if a["kind"] == "sql":
            session_sums.setdefault(a["group"], []).append((a, b))
    sessions = [{"session": name, "turns": len(members),
                 "correct_left": sum(a["correct"] for a, _ in members),
                 "correct_right": sum(b["correct"] for _, b in members),
                 "jct_sum_delta": (sum(b["jct"] - a["jct"] for a, b in members)
                                   if all(a["jct"] is not None and b["jct"] is not None for a, b in members) else None)}
                for name, members in sorted(session_sums.items())]
    return {"planned_pairs": len(pairs), "both_correct": len(both_correct),
            "became_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "became_incorrect": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "jct_delta": deltas(pairs), "correct_jct_delta": deltas(both_correct),
            "correct_model_jct_delta": deltas((a, b) for a, b in both_correct if a["model_steps"] and b["model_steps"]),
            "correct_direct_jct_delta": deltas((a, b) for a, b in both_correct if not (a["model_steps"] or b["model_steps"])),
            "sessions": sessions, "matching_requests": matching_requests(left, right)}


def failure_example(case, raw):
    task = raw["task"] or {}
    report = task.get("report") or {}
    knowledge = report.get("knowledge") or {}
    query = report.get("query_result") or {}
    return {"request": case["request"], "expected_sql": case.get("label", {}).get("sql"),
            "expected_query_result": case.get("expected_query_result"),
            "expected_knowledge": case.get("label", {}).get("outcome"),
            "expected_supports": case.get("label", {}).get("supports"),
            "actual_sql": report.get("sql"), "actual_query_result": {k: query[k] for k in ("columns", "rows", "truncated") if k in query},
            "actual_knowledge": {k: knowledge[k] for k in ("outcome", "candidate_ids", "retrieved_ids") if k in knowledge},
            "citations": [c["chunk_id"] for c in knowledge.get("citations", [])],
            "error": task.get("error"), "rejection": raw.get("rejection"),
            "events": [event["type"] for event in raw["events"]],
            "step_kinds": [step["kind"] for step in raw["journal"]["steps"]]}


def collect(root, manifest_path, *, smoke=False, workbench=None):
    registration = read(root / "run.json")
    complete = read(root / "completed.json")
    if registration["identity"] != complete["identity"] or complete["identity"]["manifest_sha256"] != file_sha(manifest_path):
        raise ValueError("Registered run and manifest do not match")
    workload = index_cases(groups(read(manifest_path), smoke=smoke))
    if len(workload) != complete["arms"][0]["submitted"]:
        raise ValueError("Workload and terminal submission counts differ")
    if workbench is not None and (workbench["manifest_sha256"] != file_sha(manifest_path)
                                  or not workbench["rules"].keys() <= workload.keys()):
        raise ValueError("Workbench review rules require this full frozen workload")
    examples = {}
    failures_by_case = Counter()
    blocks = []
    for number, block in enumerate(registration["blocks"], 1):
        results, compact = {}, {}
        arm_receipts = [receipt for receipt in complete["arms"] if receipt["block"] == number]
        if [receipt["arm"] for receipt in arm_receipts] != block["order"]:
            raise ValueError("Completed arms do not match the registered order")
        for receipt in arm_receipts:
            arm = receipt["arm"]
            attempt = root / receipt["attempt"]
            result = read(attempt / "result.json")
            if result["cases"].keys() != workload.keys():
                raise ValueError("Unaccounted submission ID")
            failures = {key: row for key, row in result["cases"].items() if not row["correct"]}
            failures_by_case.update(failures.keys())
            new_issues = {row["issue"] for row in failures.values()} - examples.keys()
            reviews = {}
            if new_issues or workbench is not None:
                with (attempt / "cases.jsonl").open(encoding="utf-8") as ledger:
                    for line in ledger:
                        raw = json.loads(line)
                        case_id = raw["case_id"]
                        issue = failures.get(case_id, {}).get("issue")
                        if workbench is not None and case_id in workbench["rules"]:
                            rule = workbench["rules"][case_id]
                            reviews[case_id] = {"reason": rule["reason"],
                                                "correct": check(rule, workload[case_id][2], raw, None)}
                        if issue in new_issues:
                            examples[issue] = {"block": number, "arm": arm, "case_id": case_id,
                                               "attempt": receipt["attempt"], "task_id": failures[case_id]["task_id"],
                                               **failure_example(workload[case_id][2], raw)}
                            new_issues.remove(issue)
                            if not new_issues and workbench is None:
                                break
            results[arm] = {**result, "trace": read(attempt / "joined-trace.json")}
            compact[arm] = {**{k: v for k, v in result.items() if k != "cases"},
                            "attempt": receipt["attempt"], "failures": failures, "lifecycle": lifecycle(attempt)}
            if workbench is not None:
                if reviews.keys() != workbench["rules"].keys():
                    raise ValueError("Incomplete workbench review")
                adjusted = {key: (reviews[key]["correct"] if key in reviews else row["correct"])
                            for key, row in result["cases"].items()}
                compact[arm]["workbench_review"] = {
                    "correct": sum(adjusted.values()), "submitted": len(adjusted), "reviews": reviews,
                    "correct_by_kind": dict(Counter(row["kind"] for key, row in result["cases"].items() if adjusted[key])),
                    "remaining_failures": [key for key, correct in adjusted.items() if not correct],
                }
        blocks.append({"block": number, **block, "arms": compact,
                       "pairs": {f"{a}_to_{b}": compare(results[a], results[b]) for a, b in PAIRS}})
    return {"identity": registration["identity"], "stage": registration["stage"], "cases_per_arm": len(workload),
            "blocks": blocks, "failure_occurrences_by_case": dict(failures_by_case), "failure_examples": examples,
            "workbench_review_contract": workbench}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workbench-contract", type=Path)
    args = parser.parse_args()
    workbench = read(args.workbench_contract) if args.workbench_contract else None
    dump(args.output, collect(args.run, args.manifest, smoke=args.smoke, workbench=workbench))


if __name__ == "__main__":
    main()
