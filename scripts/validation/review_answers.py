"""Export attributable answer review inputs, or summarize explicitly supplied reviews."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from agentserve.evaluation.answers import AnswerReview, summarize_answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((args.case / "tasks").glob("*.json"))
    ]
    labels = json.loads((args.case / "evaluations.json").read_text(encoding="utf-8"))
    if args.reviews:
        reviews = TypeAdapter(list[AnswerReview]).validate_json(args.reviews.read_bytes())
        result = summarize_answers(records, labels, reviews)
        result["reviews_sha256"] = hashlib.sha256(args.reviews.read_bytes()).hexdigest()
    else:
        quality = {q["task_id"]: q for q in labels}
        result = [
            {
                "task_id": r["task_id"],
                "job": r["job"],
                "status": r["status"],
                "report": r["task"]["report"] if r["task"] else None,
                "evaluation": quality[r["task_id"]],
                "review": None,
            }
            for r in records
        ]
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
