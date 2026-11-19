"""Check retail quality, request comparability and end-to-end timing by arm."""

from statistics import median

from retail_history import exact_trace
from v6_phase_benchmark import distribution


def summarize(sessions):
    rows = [row for session in sessions for row in session["turns"]]
    calls = [call for row in rows for call in row["trace"]]
    metrics = [call["spec"] for call in calls if call.get("spec") is not None]
    return {
        "n": len(rows), "correct": sum(row["correct"] for row in rows), "sessions": len(sessions),
        "elapsed": sum(s["active_seconds"] for s in sessions),
        "jct": distribution(row["jct_seconds"] for row in rows),
        "model_requests": len(calls), "exact_traces": sum(exact_trace(call) for call in calls),
        "draft_metric_coverage": len(metrics),
        "draft_tokens": sum(item["num_draft_tokens"] for item in metrics),
        "accepted_tokens": sum(item["num_accepted_draft_tokens"] for item in metrics),
        "failures": [{"session": i, "turn": j, "task_id": row["task_id"], "status": row["status"],
                      "question": row["question"], "error": row["error"], "sql": row["sql"],
                      "expected": row["expected"], "observed": (row["result"] or {}).get("rows")}
                     for i, session in enumerate(sessions, 1) for j, row in enumerate(session["turns"], 1)
                     if not row["correct"]],
    }


def compare(left, right):
    deltas, matched, request_deltas, changed, grouped = [], [], [], [], []
    for i, (lhs, rhs) in enumerate(zip(left, right, strict=True), 1):
        group_delta = rhs["active_seconds"] - lhs["active_seconds"]
        grouped.append({"session": i, "turns": len(lhs["turns"]), "elapsed_delta": group_delta})
        for j, (a, b) in enumerate(zip(lhs["turns"], rhs["turns"], strict=True), 1):
            if (a["sql"], a["result"], a["status"]) != (b["sql"], b["result"], b["status"]):
                changed.append([i, j])
            if not (a["correct"] and b["correct"]):
                continue
            delta = b["jct_seconds"] - a["jct_seconds"]
            deltas.append(delta)
            identical = len(a["trace"]) == len(b["trace"])
            for x, y in zip(a["trace"], b["trace"]):
                equal = (exact_trace(x) and exact_trace(y) and
                         all(x[key] == y[key] for key in ("prompt", "tokens", "finish", "phase")))
                identical &= equal
                if equal:
                    request_deltas.append((y["end"] - y["start"]) - (x["end"] - x["start"]))
            if identical and a["trace"] and a["sql"] == b["sql"]:
                matched.append(delta)
    return {"both_correct": len(deltas), "correct_jct_delta": distribution(deltas),
            "strict_jct_delta": distribution(matched), "model_http_delta": distribution(request_deltas),
            "median_model_http_delta": median(request_deltas) if request_deltas else None,
            "changed_sql_result_or_status": changed, "sessions": grouped}
