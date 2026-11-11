"""Explicit, attributed answer review; an absent review never becomes a passing score."""

from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass
class AnswerReview:
    # CLI TypeAdapter validates external JSON; pure aggregation needs only the standard library.
    __pydantic_config__: ClassVar[dict] = {"extra": "forbid", "strict": True}
    task_id: str
    reviewer: str
    source: Literal["human", "codex_assisted"]
    grounded: bool
    adequate: bool
    truncation_disclosed: bool
    evidence: str

    def __post_init__(self):
        if not self.task_id or not self.reviewer or not self.evidence:
            raise ValueError("Review attribution and evidence must be nonempty")


def summarize_answers(records, evaluations, reviews: list[AnswerReview]):
    ids = {r["task_id"] for r in records}
    by_review = {r.task_id: r for r in reviews}
    if len(by_review) != len(reviews) or not set(by_review) <= ids:
        raise ValueError("Duplicate or unknown reviewed task")
    by_quality = {q["task_id"]: q for q in evaluations}
    full = bounded = partial = unreviewed = 0
    details = []
    for record in records:
        task_id = record["task_id"]
        review = by_review.get(task_id)
        eligible = record["status"] == "COMPLETED" and by_quality[task_id]["status"] == "correct"
        truncated = (
            bool(((record["task"]["report"] or {}).get("query_result") or {}).get("truncated"))
            if record["task"]
            else False
        )
        faithful = bool(
            eligible
            and review
            and review.grounded
            and (not truncated or review.truncation_disclosed)
        )
        complete = bool(faithful and review.adequate and not truncated)
        full += complete
        bounded += faithful
        partial += faithful and truncated
        unreviewed += record["status"] == "COMPLETED" and review is None
        details.append(
            {
                "task_id": task_id,
                "sql_and_workflow_ok": eligible,
                "reviewed": review is not None,
                "faithful_bounded_response": faithful,
                "fully_successful": complete,
                "truncated": truncated,
            }
        )
    return {
        "tasks": len(records),
        "reviews_provided": len(reviews),
        "completed_unreviewed": unreviewed,
        "review_sources": sorted({r.source for r in reviews}),
        "fully_successful": full,
        "fully_successful_over_all": full / len(records) if records else None,
        "faithful_bounded_responses": bounded,
        "honest_partial_responses": partial,
        "details": details,
    }
