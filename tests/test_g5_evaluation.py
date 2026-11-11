import pytest

from agentserve.domain.models import QueryResult
from agentserve.evaluation.sql import SQLEvaluator, compare


@pytest.mark.parametrize(
    "predicted,reference,ordered,correct",
    [
        ([[1], [1], [2]], [[2], [1], [1]], False, True),
        ([[1], [2]], [[2], [1], [1]], False, False),
        ([[1], [2]], [[2], [1]], True, False),
        ([[1.0, None]], [[1, None]], True, True),
        ([["1"]], [[1]], True, False),
    ],
)
def test_result_comparison_preserves_duplicates_order_types_and_nulls(
    predicted, reference, ordered, correct
):
    result = compare(
        QueryResult(columns=["x"] * len(predicted[0]), rows=predicted),
        QueryResult(columns=["y"] * len(reference[0]), rows=reference),
        ordered=ordered,
    )
    assert result["status"] == ("correct" if correct else "incorrect")


def test_empty_results_still_require_same_projection_width():
    assert (
        compare(QueryResult(columns=["a", "b"]), QueryResult(columns=["a"]), ordered=False)[
            "status"
        ]
        == "incorrect"
    )


def test_truncated_or_invalid_reference_is_not_scored_as_correct():
    assert (
        compare(QueryResult(), QueryResult(truncated=True), ordered=False)["status"]
        == "not_evaluated"
    )
    assert (
        compare(QueryResult(truncated=True), QueryResult(), ordered=False)["status"]
        == "not_evaluated"
    )
    assert (
        compare(QueryResult(status="SQL_EXECUTION_ERROR"), QueryResult(), ordered=False)["status"]
        == "incorrect"
    )


async def test_evaluator_reexecutes_full_query_and_supports_readonly_like(settings):
    evaluator = SQLEvaluator({"demo_orders": settings.databases["demo_orders"].path})
    try:
        result = await evaluator.evaluate(
            "demo_orders",
            "SELECT name FROM customers WHERE name LIKE 'A%'",
            "SELECT name FROM customers WHERE customer_id=1",
            ordered=False,
        )
        assert result["status"] == "correct"
    finally:
        evaluator.close()
