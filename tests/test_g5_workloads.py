import asyncio
import json
import sqlite3

import httpx
import pytest

from agentserve.analysis.metrics import summarize
from agentserve.workloads.runner import run_closed_loop
from agentserve.workloads.spider import balanced, prepare, write_frozen


def test_balanced_selection_is_deterministic_and_excludes_smoke_ids():
    rows = [{"id": f"{db}-{i}", "db_id": db} for db in ("a", "b", "c") for i in range(4)]
    smoke = balanced(rows, 3, 42)
    development = balanced(rows, 6, 42, exclude=[r["id"] for r in smoke])
    assert len({r["db_id"] for r in smoke}) == 3
    assert {r["id"] for r in smoke}.isdisjoint(r["id"] for r in development)
    assert development == balanced(list(reversed(rows)), 6, 42, exclude=[r["id"] for r in smoke])


def test_frozen_files_cannot_be_silently_reselected(tmp_path):
    path = tmp_path / "manifest.json"
    write_frozen(path, {"ids": [1, 2]})
    write_frozen(path, {"ids": [1, 2]})
    with pytest.raises(ValueError, match="Frozen output differs"):
        write_frozen(path, {"ids": [2, 3]})


def test_metrics_keep_failures_unknown_quality_and_denominators():
    records = [
        {
            "status": "COMPLETED",
            "accepted": True,
            "task": {"created_at": 0, "started_at": 1, "ended_at": 3, "error": None},
            "evaluation": {"status": "incorrect"},
        },
        {
            "status": "FAILED",
            "accepted": True,
            "task": {
                "created_at": 0,
                "started_at": 1,
                "ended_at": 2,
                "error": {"code": "MODEL_HTTP_ERROR"},
            },
            "evaluation": {"status": "not_evaluated"},
        },
        {
            "status": "REJECTED",
            "accepted": False,
            "task": None,
            "evaluation": {"status": "not_evaluated"},
        },
    ]
    result = summarize({"records": records, "window_seconds": 4})
    assert result["submitted"] == 3 and result["accepted"] == 2 and result["completed"] == 1
    assert result["correct_over_all"] == 0 and result["evaluated"] == 1
    assert result["completed_tasks_per_second"] == 0.25 and result["correct_tasks_per_second"] == 0


def test_prepared_splits_separate_gold_and_official_database_domains(tmp_path):
    raw = tmp_path / "data/raw/spider"
    raw.mkdir(parents=True)
    for split, dbs in [("train_spider", ["train_a", "train_b"]), ("dev", ["dev_a", "dev_b"])]:
        rows = []
        for db in dbs:
            path = tmp_path / "data/databases/spider" / db / (db + ".sqlite")
            path.parent.mkdir(parents=True)
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE t(x INTEGER)")
            rows.extend(
                {
                    "db_id": db,
                    "question": "Count rows",
                    "query": "SELECT COUNT(*) FROM t",
                    "sql": {"orderBy": []},
                }
                for _ in range(3)
            )
        (raw / (split + ".json")).write_text(json.dumps(rows))
    manifest = prepare(tmp_path, sizes=(2, 3, 4))
    assert manifest == prepare(tmp_path, sizes=(2, 3, 4))
    base = tmp_path / "data/splits/spider_g5_v1"
    groups = {
        key: json.loads((base / (key + ".json")).read_text())
        for key in ("smoke", "development", "evaluation")
    }
    assert {q["id"] for q in groups["smoke"]}.isdisjoint(q["id"] for q in groups["development"])
    assert {q["source_db_id"] for q in groups["development"]}.isdisjoint(
        q["source_db_id"] for q in groups["evaluation"]
    )
    assert all(
        set(q) == {"id", "source_db_id", "database_id", "question"}
        for group in groups.values()
        for q in group
    )


async def test_closed_loop_is_bounded_and_never_posts_extra_gold_fields(tmp_path):
    active = peak = counter = 0
    observed = []

    async def handle(request):
        nonlocal active, peak, counter
        if request.method == "POST":
            observed.append(json.loads(request.content))
            active += 1
            peak = max(peak, active)
            counter += 1
            task_id = str(counter)
            await asyncio.sleep(0.01)
            return httpx.Response(202, json={"task_id": task_id})
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"events": []})
        if request.url.path.endswith("/steps"):
            return httpx.Response(200, json={"steps": [], "attempts": []})
        active -= 1
        return httpx.Response(
            200,
            json={"status": "COMPLETED", "task_id": request.url.path.split("/")[-1], "error": None},
        )

    jobs = [
        {"id": str(i), "database_id": "demo", "question": "Count rows", "sql": "SECRET_REFERENCE"}
        for i in range(7)
    ]
    result = await run_closed_loop(
        "http://test",
        jobs,
        concurrency=2,
        output=tmp_path / "run",
        transport=httpx.MockTransport(handle),
    )
    assert peak == 2 and len(result["records"]) == 7
    assert all(set(body) == {"database_id", "question"} for body in observed)
