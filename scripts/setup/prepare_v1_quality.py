"""Prepare synthetic acceptance and freeze fresh public-dev IDs without inspecting results."""

import hashlib
import json
from pathlib import Path

from agentserve.workloads.business import prepare
from agentserve.workloads.spider import balanced, write_frozen

ROOT = Path(__file__).resolve().parents[2]


def main():
    business = prepare(ROOT)
    old = json.loads((ROOT / "data/splits/spider_g5_v1/evaluation.json").read_text())
    old_ids = {q["id"] for q in old}
    old_text = {(q["source_db_id"], " ".join(q["question"].lower().split())) for q in old}
    source = ROOT / "data/raw/spider/dev.json"
    rows = [
        {**row, "id": f"spider.dev.{i:06d}"} for i, row in enumerate(json.loads(source.read_text()))
    ]
    eligible = [
        r
        for r in rows
        if r["id"] not in old_ids
        and (r["db_id"], " ".join(r["question"].lower().split())) not in old_text
    ]
    selected = balanced(eligible, 100, 43)
    base = ROOT / "data/splits/v1_quality_v1"
    questions = [
        {
            "id": r["id"],
            "source_db_id": r["db_id"],
            "database_id": "spider_" + r["db_id"],
            "question": r["question"],
        }
        for r in selected
    ]
    gold = [
        {"id": r["id"], "sql": r["query"], "ordered": bool(r["sql"]["orderBy"])} for r in selected
    ]
    write_frozen(base / "acceptance.json", questions)
    write_frozen(ROOT / "data/gold/v1_quality_v1/acceptance.json", gold)
    databases = json.loads((ROOT / "data/splits/spider_g5_v1/databases.json").read_text())
    databases["business_orders_v1"] = business["database"]
    write_frozen(base / "databases.json", databases)
    manifest = {
        "version": "v1_quality_v1",
        "seed": 43,
        "acceptance_count": len(questions),
        "databases": sorted({r["db_id"] for r in selected}),
        "old_evaluation_id_overlap": len(old_ids & {q["id"] for q in questions}),
        "old_evaluation_exact_question_overlap": 0,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "acceptance_questions_sha256": hashlib.sha256(
            (base / "acceptance.json").read_bytes()
        ).hexdigest(),
        "acceptance_gold_sha256": hashlib.sha256(
            (ROOT / "data/gold/v1_quality_v1/acceptance.json").read_bytes()
        ).hexdigest(),
        "business": business,
        "purpose": "fresh public-dev engineering acceptance after development selection; not hidden test or official TSA",
    }
    write_frozen(base / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "acceptance_count": 100,
                "databases": len(manifest["databases"]),
                "overlap": manifest["old_evaluation_id_overlap"],
                "business_questions": business["question_count"],
                "business_rows": business["rows"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
