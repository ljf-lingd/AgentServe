"""Offline SQL workload scoring. The HTTP workload driver never receives gold."""

from agentserve.evaluation.sql import SQLEvaluator


async def evaluate_batch(batch, gold, databases):
    evaluator = SQLEvaluator(databases)
    try:
        for record in batch["records"]:
            job = record["job"]
            reference = gold[job.get("source_id", job["id"])]
            task = record["task"]
            predicted = (task["report"] or {}).get("sql") if task else None
            record["evaluation"] = await evaluator.evaluate(
                job["database_id"], predicted, reference["sql"], ordered=reference["ordered"]
            )
        return [
            {"index": r["index"], "id": r["job"]["id"], "task_id": r["task_id"], **r["evaluation"]}
            for r in batch["records"]
        ]
    finally:
        evaluator.close()
