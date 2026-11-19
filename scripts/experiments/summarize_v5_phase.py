"""Condense complete v5 runs into a portable report with per-request comparisons."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from run_v5_phase import PAIRS
from v6_phase_benchmark import BLOCKS, TASKS, dump, file_sha


def load(path):
    return json.loads(path.read_bytes())


def matching_requests(left, right):
    phases = defaultdict(list)
    for name, lhs in left["cases"].items():
        if not lhs["correct"] or not right["cases"][name]["correct"]:
            continue
        previous = {(step["step_no"], step["phase"]): step["call"]
                    for step in left["trace"]["cases"][name]}
        for step in right["trace"]["cases"][name]:
            phase = step["phase"]
            a, b = previous.get((step["step_no"], phase)), step["call"]
            if a and b and all(a[key] == b[key] for key in (
                "model_input_sha256", "model_output_sha256", "usage", "finish_reason",
            )):
                phases[phase].append((a, b))
    result = {}
    for phase, requests in sorted(phases.items()):
        http_deltas = [b["http_seconds"] - a["http_seconds"] for a, b in requests]
        decode_deltas = [(b["metrics"]["generation_time_ms"] - a["metrics"]["generation_time_ms"]) / 1000
                         for a, b in requests if "generation_time_ms" in (a["metrics"] or {})
                         and "generation_time_ms" in (b["metrics"] or {})]
        result[phase] = {
            "paired_requests": len(requests),
            "median_http_delta_seconds": median(http_deltas),
            "median_http_delta_percent": median(100 * delta / a["http_seconds"]
                                                for (a, _), delta in zip(requests, http_deltas)),
            "paired_generation_times": len(decode_deltas),
            "median_generation_delta_seconds": median(decode_deltas) if decode_deltas else None,
        }
    return result


def compact_cases(result):
    rows = {}
    for name, case in result["cases"].items():
        task = case["task"]
        row = {"task_id": task["task_id"], "status": task["status"], "correct": case["correct"],
               "jct_seconds": case["jct"], "model_steps": len(result["trace"]["cases"][name])}
        if not case["correct"]:
            report = task.get("report") or {}
            row.update(error=task.get("error"), sql=report.get("sql"),
                       rows=(report.get("query_result") or {}).get("rows"))
        rows[name] = row
    return rows


def lifecycle(receipts):
    cmd = load(receipts / "engine-command.json")
    ready = load(receipts / "gpu-ready.json")
    stopped = load(receipts / "engine-closed.json")
    resource = load(receipts / "resources-lifecycle.json")
    alive = {process["pid"] for process in stopped["gpu_after"]["processes"]}
    return {"gpu_owned": ready["owned"], "engine_pid": stopped["pid"], "engine_port": stopped["port"],
            "gpu_released": all(process["pid"] not in alive for process in ready["owned"]),
            "cpu_closed": resource["closed"], "resource_port": resource["port"],
            "speculative_config": "--speculative-config" in cmd["argv"]}


def collect(matrix, pilot, manifest):
    manifest_hash = file_sha(manifest)
    comparisons = load(matrix / "comparison.json")
    paths = load(matrix / "paired-paths.json")
    if len(comparisons) != len(BLOCKS) or len(paths) != len(BLOCKS):
        raise ValueError("All frozen blocks are required")
    blocks = []
    for index, block in enumerate(BLOCKS, 1):
        compact = paths[index - 1]
        results = {arm: load(matrix / f"b{index}-{arm}/evidence.json") for arm in block["order"]}
        for arm, result in results.items():
            if (result["freeze_sha256"] != manifest_hash or result["block"] != index
                    or result["arm"] != arm or set(result["cases"]) != set(TASKS)):
                raise ValueError(f"Invalid run evidence for block {index} {arm}")
            compact["arms"][arm]["cases"] = compact_cases(result)
            compact["arms"][arm]["lifecycle"] = lifecycle(matrix / f"b{index}-{arm}-engine")
        for lhs, rhs in PAIRS:
            key = f"{lhs}_to_{rhs}"
            compact["pairs"][key].update(comparisons[index - 1]["pairs"][key])
            compact["pairs"][key]["matching_requests"] = matching_requests(results[lhs], results[rhs])
        blocks.append(compact)
    pilot_arms = {}
    for arm in BLOCKS[0]["order"]:
        directory = pilot / f"b1-{arm}-engine"
        result = load(directory / "pilot.json")
        pilot_arms[arm] = {"summary": result["summary"], "lifecycle": lifecycle(directory),
                           "trace": {k: v for k, v in result["trace"].items() if k != "cases"}}
    return {"freeze_sha256": manifest_hash, "manifest_gold": load(manifest)["gold"],
            "pilot_identity": load(pilot / "run.json"), "matrix_identity": load(matrix / "run.json"),
            "pilot": pilot_arms, "blocks": blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dump(args.output, collect(args.matrix, args.pilot, args.manifest))


if __name__ == "__main__":
    main()
