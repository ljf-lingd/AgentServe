"""Group actual workload conditions and derive evidence without changing labels."""

import json
import statistics
from collections import Counter, defaultdict


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_conditions(runs):
    groups = defaultdict(list)
    for run in runs:
        key = run.get("condition_id") or (
            f"c{run['concurrency']}-m{run.get('llm_concurrency', 2)}"
            f"-s{run.get('sql_workers', 2)}-w{run['sql_wait_seconds']}"
        )
        groups[key].append(run)
    return dict(groups)


def aggregate(runs):
    result = {
        "runs": len(runs),
        **{
            key: sum(r[key] for r in runs)
            for key in ["submitted", "accepted", "completed", "failed", "completed_correct"]
        },
    }
    for field in [
        "completed_tasks_per_second",
        "correct_tasks_per_second",
        "mean_jct_seconds",
        "p95_jct_seconds_descriptive",
        "mean_model_capacity_wait_seconds",
    ]:
        values = [r[field] for r in runs if r[field] is not None]
        result[field] = (
            {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
            if values
            else None
        )
    result["stage_sums"] = {
        key: sum(r[key] for r in runs)
        for key in [
            "jct_seconds_sum",
            "model_client_seconds_sum",
            "model_capacity_wait_seconds_sum",
            "application_queue_seconds_sum",
            "sql_stage_seconds_sum",
        ]
    }
    result["quality"] = dict(sum((Counter(r["quality"]) for r in runs), Counter()))
    return result


def derive(root, summary):
    report = {"quality": {}, "load": {}, "tool_wait": {}}
    for case in summary["quality_runs"]:
        labels = read(root / case["case"] / "evaluations.json")
        report["quality"][case["case"]] = {
            **case,
            "comparison_reasons": dict(Counter(q["reason"] for q in labels)),
            "not_evaluated_ids": [q["id"] for q in labels if q["status"] == "not_evaluated"],
        }
    for target, source in [("load", "performance_runs"), ("tool_wait", "tool_wait_runs")]:
        report[target] = {
            key: aggregate(runs) for key, runs in group_conditions(summary[source]).items()
        }
    records = [read(path) for path in root.glob("*/tasks/*.json")]
    warmups = [read(path) for path in root.glob("*/warmup/*.json")]
    all_records = records + warmups
    attempts = [
        a
        for record in all_records
        for a in record.get("history", {}).get("attempts", [])
        if a["step_no"]
        in {
            s["step_no"] for s in record.get("history", {}).get("steps", []) if s["kind"] == "model"
        }
    ]
    ids = [a["request_id"] for a in attempts if a["request_id"] is not None]
    owned = {p["pid"] for p in read(root / "gpu-model-ready.json")["owned_model_processes"]}
    samples = [json.loads(line) for line in (root / "gpu-samples.jsonl").read_text().splitlines()]
    valid = [s for s in samples if "gpus" in s]
    report["evidence"] = {
        "task_records_including_prewarm": len(all_records),
        "separate_run_warmups": len(warmups),
        "model_attempts": len(attempts),
        "model_request_ids": len(ids),
        "unique_model_request_ids": len(set(ids)),
        "unknown_usage_attempts": sum(a["usage"] is None for a in attempts),
        "known_prompt_tokens": sum(
            a["usage"]["prompt_tokens"] for a in attempts if a["usage"] is not None
        ),
        "known_completion_tokens": sum(
            a["usage"]["completion_tokens"] for a in attempts if a["usage"] is not None
        ),
        "gpu_samples": len(valid),
        "gpu_sample_errors": len(samples) - len(valid),
        "samples_with_owned_gpu_process": sum(
            any(p["pid"] in owned for p in s["processes"]) for s in valid
        ),
    }
    return report
