"""Frozen V3.1 tasks, evidence labels and reproducible task API probes."""

import asyncio
import math
import statistics
import time

import httpx

DATABASE = "business_orders_v1"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


def request_for(case):
    return {
        "mode": "knowledge",
        "database_id": case.get("database_id", DATABASE),
        **{k: case[k] for k in ("question", "as_of", "clause_id") if k in case},
    }


async def wait_for_task(client, task_id, seconds=105):
    async with asyncio.timeout(seconds):
        while True:
            row = (await client.get(f"/tasks/{task_id}")).raise_for_status().json()
            if row["status"] in TERMINAL:
                return row
            await asyncio.sleep(0.08)


async def run_requests(url, requests, concurrency):
    slots = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=10, trust_env=False) as client:

        async def run(case_id, body):
            async with slots:
                start = time.monotonic()
                submitted = (await client.post("/tasks", json=body)).raise_for_status().json()
                task_id = submitted["task_id"]
                row = await wait_for_task(client, task_id)
                row["client_seconds"] = time.monotonic() - start
                row["jct_seconds"] = row["ended_at"] - row["created_at"]
                row["trace"] = (
                    (await client.get(f"/tasks/{task_id}/events", params={"limit": 1000}))
                    .raise_for_status()
                    .json()["events"]
                )
                return case_id, row

        return dict(await asyncio.gather(*(run(case_id, body) for case_id, body in requests)))


def label(case, row, snapshot):
    report = row["report"] or {}
    knowledge = report.get("knowledge", {})
    actual = knowledge.get("outcome")
    expected = case["outcome"]
    candidates = set(knowledge.get("candidate_ids", []))
    evidence = set(knowledge.get("retrieved_ids", []))
    citations = knowledge.get("citations", [])
    cited = {c["chunk_id"] for c in citations}
    as_of = case.get("as_of", snapshot.as_of)
    active = snapshot.active(case.get("database_id", DATABASE), as_of, case.get("clause_id"))
    safety = []
    if (candidates | evidence | cited) - active.keys():
        safety.append("ineligible_evidence")
    if evidence - candidates or cited - evidence:
        safety.append("out_of_context_citation")
    if (
        report.get("sql") is not None
        or report.get("query_result") is not None
        or any(event["type"].startswith("SQL_") for event in row["trace"])
    ):
        safety.append("knowledge_sql")
    if knowledge and (
        knowledge["as_of"] != as_of or knowledge["snapshot_sha256"] != snapshot.identity
    ):
        safety.append("wrong_snapshot")
    for citation in citations:
        original = active.get(citation["chunk_id"])
        if original is None:
            continue
        if citation["quote"] not in original["text"] or any(
            citation[k] != original[k]
            for k in ("source_sha256", "content_sha256", "anchor", "version", "path")
        ):
            safety.append("invalid_citation")
    outcome_correct = row["status"] == "COMPLETED" and actual == expected
    complete, covered = None, None
    if expected == "answered":
        required = {s["chunk_id"] for s in case["supports"]}
        covered = required <= candidates
        complete = required <= evidence
        for item in case["supports"]:
            quote = "\n".join(c["quote"] for c in citations if c["chunk_id"] == item["chunk_id"])
            outcome_correct &= all(phrase in quote for phrase in item["must"])
    elif expected == "conflicting_evidence":
        outcome_correct &= set(case["conflict_ids"]) <= cited
    else:
        outcome_correct &= not citations
    model_events = [e["payload"] for e in row["trace"] if e["type"] == "MODEL_COMPLETED"]
    usage = [event["usage"] for event in model_events if event.get("usage")]
    return {
        "id": case["id"],
        "task_id": row["task_id"],
        "expected": expected,
        "outcome": actual,
        "status": row["status"],
        "error": row["error"],
        "passed": bool(outcome_correct and not safety),
        "safety": safety,
        "candidate_complete": covered,
        "evidence_complete": complete,
        "evidence_tokens": knowledge.get("evidence_tokens", 0),
        "jct": row["jct_seconds"],
        "retrieval_seconds": next(
            (
                e["payload"]["client_seconds"]
                for e in row["trace"]
                if e["type"] == "KNOWLEDGE_RETRIEVED"
            ),
            None,
        ),
        "generation_seconds": sum(event["client_seconds"] for event in model_events),
        "model_calls": report.get("model_calls", 0),
        "usage_records": len(usage),
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
        "completion_tokens": sum(u["completion_tokens"] for u in usage),
    }


def p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None


def metrics(rows):
    positives = [r for r in rows if r["expected"] == "answered"]
    negatives = [r for r in rows if r["expected"] != "answered"]
    return {
        "positive_n": len(positives),
        "negative_n": len(negatives),
        "candidate_complete": sum(r["candidate_complete"] is True for r in positives),
        "evidence_complete": sum(r["evidence_complete"] is True for r in positives),
        "grounded_by_labels": sum(r["passed"] for r in positives),
        "negative_correct": sum(r["passed"] for r in negatives),
        "safety_errors": sum(bool(r["safety"]) for r in rows),
        "evidence_tokens_median": statistics.median(r["evidence_tokens"] for r in positives),
        "retrieval_p95": p95(
            [r["retrieval_seconds"] for r in rows if r["retrieval_seconds"] is not None]
        ),
        "jct_p95": p95([r["jct"] for r in rows]),
        "generation_p95": p95([r["generation_seconds"] for r in rows if r["model_calls"]]),
        **{
            name: sum(r[name] for r in rows)
            for name in ("model_calls", "usage_records", "prompt_tokens", "completion_tokens")
        },
    }
