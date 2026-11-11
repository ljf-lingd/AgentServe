"""Bounded single-database execution comparison, not official test-suite accuracy."""

import time
from collections import Counter

from agentserve.domain.models import QueryResult
from agentserve.tools.sqlite import SQLTool

EVALUATION_PROTOCOL = "single-db-strict-v1-plus-replace"


def compare(predicted: QueryResult, reference: QueryResult, *, ordered: bool):
    if reference.status != "ok" or reference.truncated:
        return {
            "status": "not_evaluated",
            "reason": "reference_" + (reference.limit_reason or reference.status),
        }
    if predicted.truncated:
        return {"status": "not_evaluated", "reason": "predicted_result_limit"}
    if predicted.status != "ok":
        return {"status": "incorrect", "reason": predicted.status}
    if len(predicted.columns) != len(reference.columns):
        return {"status": "incorrect", "reason": "column_count"}

    def key(row):
        return tuple(("blob", v["hex"]) if isinstance(v, dict) else v for v in row)

    if ordered:
        equal = [key(r) for r in predicted.rows] == [key(r) for r in reference.rows]
    else:
        equal = Counter(key(r) for r in predicted.rows) == Counter(key(r) for r in reference.rows)
    return {
        "status": "correct" if equal else "incorrect",
        "reason": "result_match" if equal else "rows",
    }


class SQLEvaluator:
    def __init__(self, databases):
        self.tool = SQLTool(
            databases,
            workers=2,
            timeout_seconds=3,
            row_limit=10000,
            byte_limit=1048576,
            function_profile="with_replace_v1",
        )

    def close(self):
        self.tool.close()

    async def evaluate(self, database_id, predicted_sql, reference_sql, *, ordered):
        reference = await self.tool.execute(
            database_id, reference_sql, deadline=time.monotonic() + 4
        )
        if not predicted_sql:
            predicted = QueryResult(status="NO_SQL", error="Task produced no SQL")
        else:
            predicted = await self.tool.execute(
                database_id, predicted_sql, deadline=time.monotonic() + 4
            )
        return {
            **compare(predicted, reference, ordered=ordered),
            "reference_status": reference.status,
            "predicted_status": predicted.status,
            "reference_rows": len(reference.rows),
            "predicted_rows": len(predicted.rows),
        }
