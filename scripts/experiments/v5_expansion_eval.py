"""Independent grading and compact evidence for the expanded business workload."""

from collections import Counter

from agentserve.analysis.metrics import summarize as summarize_tasks
from v31_cases import label as judge_knowledge
from v5_phase_evidence import audit
from v6_phase_benchmark import distribution, phase_stats


def grade(case, row, snapshot):
    if row["task"] is None:
        return f"submission_{row['rejection']['status']}"
    task, kind = row["task"], row["kind"]
    report = task["report"] or {}
    error = (task.get("error") or {}).get("code")
    if kind == "sql" and case["label"]["clarify"]:
        if error == "MEMORY_CONTEXT_REQUIRED" and not report.get("sql"):
            return None
        return "expected_sql_clarification"
    if task["status"] != "COMPLETED":
        return error or task["status"]
    if kind == "knowledge":
        outcome = judge_knowledge(case["label"], {**task, "trace": row["events"],
                                                  "jct_seconds": row["jct"]}, snapshot)
        if outcome["safety"]:
            return "knowledge_" + outcome["safety"][0]
        if outcome["expected"] != outcome["outcome"]:
            return "knowledge_outcome"
        return None if outcome["passed"] else "knowledge_evidence"
    observed = report.get("query_result") or {}
    if observed.get("status") != "ok" or observed["truncated"]:
        return "query_unavailable_or_truncated"
    if kind == "analysis":
        analysis = report.get("analysis") or {}
        return None if (analysis.get("months") == case["expected_months"] and
                        analysis.get("totals") == case["expected_totals"]) else "analysis_values"
    expected = case["expected_query_result"]
    if observed["rows"] != expected["rows"] or len(observed["columns"]) != len(expected["columns"]):
        return "query_result"
    if kind == "metric":
        label = case["label"]
        binding = report.get("metric") or {}
        rule, cite = binding.get("rule") or {}, binding.get("citation") or {}
        if (observed["columns"] != expected["columns"] or cite.get("chunk_id") != label["chunk_id"]
                or cite.get("version") != label["version"] or rule.get("statuses") != label["statuses"]
                or rule.get("unit") != label["unit"]):
            return "metric_binding"
    return None


def summarize(rows, workload, snapshot, calls, elapsed, arm):
    if rows.keys() != workload.keys():
        raise ValueError("Completed workload is missing frozen submissions")
    for key, row in rows.items():
        row["issue"] = grade(workload[key][2], row, snapshot)
        row["correct"] = row["issue"] is None
    trace = audit(rows, calls, arm)
    compact = {}
    for key, row in rows.items():
        task = row["task"] or {}
        compact[key] = {"kind": row["kind"], "group": row["group"], "task_id": task.get("task_id"),
                        "status": task.get("status", "REJECTED"), "jct": row["jct"],
                        "issue": row["issue"], "correct": row["correct"],
                        "model_steps": len(trace["cases"][key])}
    records = [{"accepted": row["task"] is not None, "task": row["task"],
                "status": compact[key]["status"], "events": row["events"], "history": row["journal"],
                "evaluation": {"status": "correct" if row["correct"] else "incorrect"}}
               for key, row in rows.items()]
    scores = {}
    for kind in sorted({row["kind"] for row in compact.values()}):
        members = [row for row in compact.values() if row["kind"] == kind]
        scores[kind] = {"submitted": len(members), "correct": sum(row["correct"] for row in members),
                        "statuses": dict(Counter(row["status"] for row in members)),
                        "jct": distribution(row["jct"] for row in members if row["jct"] is not None),
                        "issues": dict(Counter(row["issue"] for row in members if row["issue"]))}
    timing = summarize_tasks({"records": records, "window_seconds": elapsed})
    return {"timing": timing, "scores": scores, "cases": compact, "phases": phase_stats(rows),
            "resolution_per_second": sum(row["correct"] for row in compact.values()) / elapsed,
            "jct": distribution(row["jct"] for row in compact.values() if row["jct"] is not None),
            "trace": {k: v for k, v in trace.items() if k != "cases"}}, trace
