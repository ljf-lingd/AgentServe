"""Explicit denominators and event-derived stage measurements for completed workload records."""

import statistics
from collections import Counter


def summarize(batch):
    records = batch["records"]
    statuses = Counter(r["status"] for r in records)
    quality = Counter(r.get("evaluation", {}).get("status", "not_evaluated") for r in records)
    tasks = [r["task"] for r in records if r["task"] and r["task"]["ended_at"] is not None]
    jcts = [t["ended_at"] - t["created_at"] for t in tasks]
    queue = [t["started_at"] - t["created_at"] for t in tasks if t["started_at"] is not None]
    model_seconds = []
    model_wait = []
    sql_stages = []
    known_prompt = known_completion = unknown_usage = 0
    for record in records:
        sql_start = None
        for event in record.get("events", []):
            payload = event["payload"]
            if event["type"] == "MODEL_STARTED":
                model_wait.append(payload["capacity_wait_seconds"])
            elif event["type"] in {"MODEL_COMPLETED", "MODEL_FAILED"}:
                model_seconds.append(payload["client_seconds"])
            elif event["type"] == "SQL_STARTED":
                sql_start = event
            elif (
                event["type"] == "SQL_COMPLETED"
                and sql_start
                and event["process_epoch"] == sql_start["process_epoch"]
            ):
                sql_stages.append(event["wall_time"] - sql_start["wall_time"])
                sql_start = None
        for attempt in record.get("history", {}).get("attempts", []):
            if attempt["usage"] is not None:
                known_prompt += attempt["usage"]["prompt_tokens"]
                known_completion += attempt["usage"]["completion_tokens"]
            elif attempt["step_no"] in {
                s["step_no"]
                for s in record.get("history", {}).get("steps", [])
                if s["kind"] == "model"
            }:
                unknown_usage += 1

    def mean(values):
        return statistics.mean(values) if values else None

    evaluated = quality["correct"] + quality["incorrect"]
    completed_correct = sum(
        r["status"] == "COMPLETED" and r.get("evaluation", {}).get("status") == "correct"
        for r in records
    )
    return {
        "submitted": len(records),
        "accepted": sum(r["accepted"] for r in records),
        "statuses": dict(statuses),
        "completed": statuses["COMPLETED"],
        "failed": statuses["FAILED"],
        "quality": dict(quality),
        "evaluated": evaluated,
        "accuracy_on_evaluated": quality["correct"] / evaluated if evaluated else None,
        "correct_over_all": quality["correct"] / len(records) if records else None,
        "window_seconds": batch["window_seconds"],
        "completed_tasks_per_second": statuses["COMPLETED"] / batch["window_seconds"],
        "completed_correct": completed_correct,
        "correct_tasks_per_second": completed_correct / batch["window_seconds"],
        "mean_jct_seconds": mean(jcts),
        "p95_jct_seconds_descriptive": statistics.quantiles(jcts, n=100, method="inclusive")[94]
        if len(jcts) > 1
        else mean(jcts),
        "jct_sample_count": len(jcts),
        "mean_application_queue_seconds": mean(queue),
        "mean_model_capacity_wait_seconds": mean(model_wait),
        "model_client_seconds_sum": sum(model_seconds),
        "model_capacity_wait_seconds_sum": sum(model_wait),
        "application_queue_seconds_sum": sum(queue),
        "sql_stage_seconds_sum": sum(sql_stages),
        "jct_seconds_sum": sum(jcts),
        "known_prompt_tokens": known_prompt,
        "known_completion_tokens": known_completion,
        "unknown_usage_attempts": unknown_usage,
        "failures": dict(
            Counter(
                (r["task"]["error"] or {}).get("code", "none") if r["task"] else r["status"]
                for r in records
                if r["status"] != "COMPLETED"
            )
        ),
    }
