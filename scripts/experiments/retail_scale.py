"""Measure retail result accuracy and latency at three data scales."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import httpx
from retail_replay import DATABASE, gold, match, workloads

from agentserve.api.app import create_app
from agentserve.config import Settings


def load_importer():
    path = Path(__file__).resolve().parents[1] / "setup/prepare_online_retail_ii.py"
    spec = importlib.util.spec_from_file_location("prepare_online_retail_ii", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def samples(source, output):
    importer = load_importer()
    groups = defaultdict(list)
    uri = source.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        for month, cancelled, invoice, count in db.execute(
            """SELECT substr(min(invoice_date),1,7),is_cancellation,invoice,count(*)
               FROM retail_lines GROUP BY invoice ORDER BY invoice"""
        ):
            groups[(month, cancelled)].append((invoice, count))
    seed = random.Random(20261111)
    for group in groups.values():
        seed.shuffle(group)
    cases = []
    for target in (10_000, 100_000):
        path = output / f"retail-{target}.sqlite"
        with sqlite3.connect(path, uri=True) as db:
            db.executescript(importer.SCHEMA)
            db.execute("ATTACH DATABASE ? AS source", (uri,))
            db.execute("CREATE TEMP TABLE chosen(invoice TEXT PRIMARY KEY)")
            selected = []
            for group in groups.values():
                quota = max(1, round(sum(count for _, count in group) * target / importer.EXPECTED_ROWS))
                current = 0
                for invoice, count in group:
                    selected.append((invoice,))
                    current += count
                    if current >= quota:
                        break
            db.executemany("INSERT INTO chosen VALUES (?)", selected)
            db.execute(
                """INSERT INTO retail_lines(line_id,source_sheet,source_row,invoice,stock_code,
                   description,quantity,invoice_date,unit_price_millionths,customer_id,country)
                   SELECT line_id,source_sheet,source_row,invoice,stock_code,
                   description,quantity,invoice_date,unit_price_millionths,customer_id,country
                   FROM source.retail_lines JOIN chosen USING(invoice) ORDER BY line_id"""
            )
            db.executescript(importer.INDEXES + importer.VIEW)
            db.execute("ANALYZE main")
            count = db.execute("SELECT count(*) FROM retail_lines").fetchone()[0]
        db.close()
        cases.append((str(target), count, path))
    return [*cases, ("full", importer.EXPECTED_ROWS, source)]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1] if ordered else None


async def query(client, turn, gate, *, session_id=None):
    async with gate:
        accepted = await client.post("/tasks", json={
            "database_id": DATABASE, "question": turn["question"], "mode": "sql",
            **({"session_id": session_id} if session_id else {}),
        })
        task_id = accepted.raise_for_status().json()["task_id"]
        while True:
            task = (await client.get(f"/tasks/{task_id}")).raise_for_status().json()
            if task["status"] not in {"QUEUED", "RUNNING"}:
                break
            await asyncio.sleep(.08)
        events = (await client.get(f"/tasks/{task_id}/events")).raise_for_status().json()["events"]
        started = [e["wall_time"] for e in events if e["type"] == "SQL_STARTED"]
        completed = [e["wall_time"] for e in events if e["type"] == "SQL_COMPLETED"]
        report = task["report"] or {}
        return {
            "question": turn["question"], "expected": turn["expected"], "task_id": task_id,
            "result": report.get("query_result"), "status": task["status"], "error": task["error"],
            "correct": task["status"] == "COMPLETED" and match(report.get("query_result"), turn),
            "jct_seconds": task["ended_at"] - task["created_at"],
            "sql_seconds": sum(b - a for a, b in zip(started, completed)) if completed else None,
        }


async def run(args):
    args.output.mkdir()
    base = Settings.load(args.config)
    base.base_url = args.model_url
    base.memory.resources_url = args.resources_url
    db = base.databases[DATABASE].path
    turns = workloads("formal")
    fixed = [turns[n] for n in (0, 1, 4, 5, 8, 12, 16, 18, 20, 21, 22, 23)]
    summary = []
    for label, count, path in samples(db, args.output):
        (args.output / f"schema-{label}.sha256").write_text(
            hashlib.sha256(base.databases[DATABASE].schema_path.read_bytes()).hexdigest(),
            encoding="utf-8",
        )
        with path.open("rb") as file:
            database_hash = hashlib.file_digest(file, "sha256").hexdigest()
        reference = [sequence[0] for sequence in gold(path, fixed)]
        settings = base.model_copy(deep=True)
        settings.databases[DATABASE].path = path
        settings.state_database = args.output / f"state-{label}.sqlite"
        app = create_app(settings, web_mode="")
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            await query(client, reference[0], asyncio.Semaphore(1))
            for concurrency in (1, 4):
                results = []
                for block in range(3):
                    gate = asyncio.Semaphore(concurrency)
                    results += await asyncio.gather(*(query(client, turn, gate) for turn in reference))
                measured = {
                    "scale": label, "rows": count, "database_sha256": database_hash,
                    "concurrency": concurrency, "n": len(results),
                    "correct": sum(r["correct"] for r in results),
                    "timeouts": sum("TIMEOUT" in (r["error"] or {}).get("code", "") for r in results),
                    "jct_p95_seconds": percentile([r["jct_seconds"] for r in results], .95),
                    "sql_p95_seconds": percentile([r["sql_seconds"] for r in results if r["sql_seconds"] is not None], .95),
                }
                (args.output / f"{label}-c{concurrency}.json").write_text(
                    json.dumps({"metrics": measured, "tasks": results}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                summary.append(measured)
                print(json.dumps(measured), flush=True)
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as source:
        query_plan = source.execute("EXPLAIN QUERY PLAN " + reference[0]["sql"], reference[0]["params"]).fetchall()
    (args.output / "summary.json").write_text(
        json.dumps({"measurements": summary, "query_plan": query_plan}, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--resources-url", required=True)
    asyncio.run(run(parser.parse_args()))

