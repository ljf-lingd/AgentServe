"""Replay frozen Spider smoke questions and BIRD Mini-Dev 2026 SQLite."""

import argparse
import asyncio
import hashlib
import json
import shutil
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import httpx

from agentserve.api.app import create_app
from agentserve.config import DatabaseConfig, Settings
from agentserve.evaluation.sql import SQLEvaluator

ROOT = Path(__file__).resolve().parents[2]
BIRD_SQLITE_SHA = "926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606"
BIRD_SCORER = "bird-bench/mini_dev@1c49a375e76c471e218564da3f4617513d33e371:evaluation/evaluation_ex.py"
POLL_SECONDS = .12


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_sqlite(source, database_ids, output, *, directory=None):
    output.mkdir(exist_ok=True)
    found = {}
    with ZipFile(source) as archive:
        for member in archive.namelist():
            name = Path(member)
            if directory is not None and name.parent.parent.name != directory:
                continue
            database = name.stem
            if name.suffix != ".sqlite" or name.parent.name != database or database not in database_ids:
                continue
            if database in found:
                raise ValueError(f"Duplicate database in archive: {database}")
            path = output / (database + ".sqlite")
            with archive.open(member) as data, path.open("wb") as file:
                shutil.copyfileobj(data, file, 1024 * 1024)
            found[database] = path
    if found.keys() != set(database_ids):
        raise ValueError(f"Missing database files: {set(database_ids) - found.keys()}")
    return found


def sqlite_query(path, sql, timeout=30):
    started = time.monotonic()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        db.set_progress_handler(lambda: time.monotonic() - started > timeout, 10000)
        return db.execute(sql).fetchall()


def bird_inputs(source, output):
    questions = source / "mini_dev_sqlite.jsonl"
    if hashlib.sha256(questions.read_bytes()).hexdigest() != BIRD_SQLITE_SHA:
        raise ValueError("BIRD SQLite questions do not match the frozen release manifest")
    with questions.open(encoding="utf-8") as f:
        rows = list(map(json.loads, f))
    if sorted(row["question_id"] for row in rows) != list(range(500)):
        raise ValueError("Expected 500 distinct official BIRD question IDs")
    nested = output / "dev_databases.zip"
    with ZipFile(source / "dev.zip") as archive:
        (member,) = [name for name in archive.namelist() if name.endswith("/dev_databases.zip")]
        with archive.open(member) as data, nested.open("wb") as file:
            shutil.copyfileobj(data, file, 1024 * 1024)
    paths = extract_sqlite(nested, {row["db_id"] for row in rows}, output / "bird-db")
    databases = {}
    for name, path in paths.items():
        schema = output / (name + ".sql")
        tables = sqlite_query(path, "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY name")
        schema.write_text(";\n".join(row[0] for row in tables) + ";\n", encoding="utf-8")
        databases[name] = DatabaseConfig(path=path, schema_path=schema, context="Use SQLite syntax.")
    for row in rows:
        row["database_id"] = row["db_id"]
        row["id"] = row["question_id"]
        row["sql"] = row["SQL"]
    return rows, databases


def spider_inputs(output):
    rows = read(ROOT / "data/splits/spider_g5_v1/smoke.json")
    refs = {row["id"]: row for row in read(ROOT / "data/gold/spider_g5_v1/smoke.json")}
    paths = extract_sqlite(ROOT / "data/raw/spider/spider_data.zip",
                           {row["source_db_id"] for row in rows}, output / "spider-db", directory="database")
    configured = read(ROOT / "data/splits/spider_g5_v1/databases.json")
    databases = {}
    for row in rows:
        db = configured[row["database_id"]]
        databases[row["database_id"]] = DatabaseConfig(
            path=paths[row["source_db_id"]], schema_path=ROOT / db["schema_path"], context=db["context"]
        )
        row.update(refs[row["id"]])
    return rows, databases


async def task(client, row, semaphore):
    async with semaphore:
        question = row["question"]
        if row.get("evidence"):
            question += "\nBenchmark evidence: " + row["evidence"]
        response = await client.post("/tasks", json={
            "database_id": row["database_id"], "question": question, "mode": "sql",
        })
        task_id = response.raise_for_status().json()["task_id"]
        while True:
            result = (await client.get(f"/tasks/{task_id}")).raise_for_status().json()
            if result["status"] not in {"QUEUED", "RUNNING"}:
                break
            await asyncio.sleep(POLL_SECONDS)
    report = result["report"] or {}
    return {
        "id": row["id"], "database": row["database_id"], "difficulty": row.get("difficulty"),
        "sql": report.get("sql"), "status": result["status"], "error": result["error"],
        "task_id": task_id, "question": row["question"], "jct": result["ended_at"] - result["created_at"],
    }


async def generate(rows, base, databases, output, label, concurrency):
    settings = base.model_copy(deep=True)
    settings.databases = databases
    settings.state_database = output / (label + "-state.sqlite")
    settings.task_concurrency = concurrency
    app = create_app(settings, web_mode="")
    results = []
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
    ) as client:
        semaphore = asyncio.Semaphore(concurrency)
        for start in range(0, len(rows), 25):
            results += await asyncio.gather(*(task(client, row, semaphore) for row in rows[start:start+25]))
            write(output / (label + "-predictions.json"), results)
            print(f"{label}: {len(results)}/{len(rows)}", flush=True)
    return results


def gold_rows(row, database):
    try:
        return set(sqlite_query(database.path, row["sql"])), None
    except sqlite3.Error as error:
        return None, str(error)


def bird_score(predicted, database, reference):
    if predicted["status"] != "COMPLETED" or not predicted["sql"] or reference is None:
        return False
    try:
        return set(sqlite_query(database.path, predicted["sql"])) == reference
    except sqlite3.Error:
        return False


async def run(args):
    args.output.mkdir()
    base = Settings.load(args.config)
    base.base_url = args.model_url
    base.session_sam = True
    base.session_sam_all_phases = False
    base.memory = None
    base.knowledge = None
    base.metric = None
    base.analysis_enabled = False
    spider, spider_dbs = spider_inputs(args.output)
    predicted = await generate(spider, base, spider_dbs, args.output, "spider", 4)
    evaluator = SQLEvaluator({name: db.path for name, db in spider_dbs.items()})
    try:
        spider_labels = [
            await evaluator.evaluate(row["database_id"], pred["sql"], row["sql"], ordered=row["ordered"])
            for row, pred in zip(spider, predicted, strict=True)
        ]
    finally:
        evaluator.close()
    spider_summary = {"n": len(spider), "correct": sum(label["status"] == "correct" for label in spider_labels)}
    write(args.output / "spider-summary.json", {**spider_summary, "labels": spider_labels})
    print("Spider", spider_summary, flush=True)

    rows, databases = bird_inputs(args.bird_source, args.output)
    references = [gold_rows(row, databases[row["database_id"]]) for row in rows]
    gold_errors = {str(row["id"]): error for row, (_, error) in zip(rows, references, strict=True) if error}
    write(args.output / "bird-gold-check.json", {
        "n": len(rows), "gold_correct": len(rows) - len(gold_errors), "errors": gold_errors,
        "question_sha256": BIRD_SQLITE_SHA, "scorer_reference": BIRD_SCORER,
    })
    predicted = await generate(rows, base, databases, args.output, "bird", 4)
    by_db = defaultdict(lambda: {"n": 0, "correct": 0})
    by_level = defaultdict(lambda: {"n": 0, "correct": 0})
    for row, result, (reference, _) in zip(rows, predicted, references, strict=True):
        result["correct"] = bird_score(result, databases[row["database_id"]], reference)
        for group in (by_db[result["database"]], by_level[result["difficulty"]]):
            group["n"] += 1
            group["correct"] += result["correct"]
    write(args.output / "bird-scored.json", predicted)
    report = {
        "n": len(rows), "correct": sum(row["correct"] for row in predicted),
        "completed": sum(row["status"] == "COMPLETED" for row in predicted),
        "gold_correct": len(rows) - len(gold_errors), "by_database": by_db, "by_difficulty": by_level,
        "question_sha256": BIRD_SQLITE_SHA, "scorer_reference": BIRD_SCORER,
        "score_protocol": "Official unordered unique result-set equality; failed AgentServe tasks count as wrong.",
    }
    write(args.output / "bird-summary.json", report)
    print("BIRD", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("bird_source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-url", required=True)
    asyncio.run(run(parser.parse_args()))

