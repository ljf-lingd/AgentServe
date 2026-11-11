"""Independent four-database reference and per-task measurements for the monthly brief."""

import asyncio
import calendar
import csv
import io
import sqlite3
import time
from contextlib import closing
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from run_v21 import write
from run_v31 import read, sha
from v31_cases import DATABASE, p95, run_requests
from v32_cases import PRODUCT

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data/evaluation/analysis_v4"


def body(window):
    return {"mode": "analysis", "database_id": DATABASE, "analysis": window}


def independent_reference(path, window):
    """Read order and line relations, aggregate independently in Python."""
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        customers = dict(conn.execute("SELECT customer_id, city FROM customers"))
        items = {}
        for key, quantity, price in conn.execute(
            "SELECT order_id, quantity, unit_price_cents FROM order_items"
        ):
            items[key] = items.get(key, 0) + quantity * price
        orders = conn.execute("SELECT order_id, customer_id, status, order_date FROM orders").fetchall()
    cursor = date.fromisoformat(window["start"])
    end = date.fromisoformat(window["end"])
    months = {}
    while cursor < end:
        key = cursor.isoformat()[:7]
        months[key] = {"month": key, "revenue_cents": None, "order_count": 0}
        cursor = date.fromordinal(cursor.toordinal() + calendar.monthrange(cursor.year, cursor.month)[1])
    for key, customer, status, placed in orders:
        row = months.get(placed[:7])
        if row is None or status != "completed":
            continue
        if "city" in window and customers[customer] != window["city"]:
            continue
        row["order_count"] += 1
        if key in items:
            row["revenue_cents"] = (row["revenue_cents"] or 0) + items[key]
    rows = list(months.values())
    known = [r["revenue_cents"] for r in rows if r["revenue_cents"] is not None]
    totals = {
        "revenue_cents": sum(known) if known else None,
        "order_count": sum(r["order_count"] for r in rows),
    }
    return rows, totals


def valid_sources(analysis):
    bindings = analysis.get("metrics", [])
    if len(bindings) != 2:
        return False
    rules = read(PRODUCT)["rules"]
    manifest = {doc["id"]: doc for doc in read(PRODUCT.with_name("manifest.json"))["documents"]}
    for binding, rule in zip(bindings, rules, strict=True):
        source = binding["citation"]
        document = manifest[rule["source"].split("@")[0]]
        file = PRODUCT.with_name(document["path"])
        start, end = (int(i[1:]) for i in source["anchor"].split("-"))
        original = "\n".join(file.read_text(encoding="utf-8").splitlines()[start - 1:end])
        if not (
            binding["rule"] == rule and binding["rule_sha256"] == rule["source_binding_sha256"]
            and source["chunk_id"] == rule["source"] and source["text"] == original
            and source["source_sha256"] == document["source_sha256"] == sha(file)
            and all(source[k] == document[k] for k in ("version", "approval_ref", "path"))
        ):
            return False
    return True


def verify(case, task, db):
    report = task["report"] or {}
    analysis = report.get("analysis", {})
    expected_rows, expected_totals = independent_reference(db, case["request"])
    trace = task["trace"]
    kinds = [event["type"] for event in trace]
    stamps = {event["type"]: event["wall_time"] for event in trace}
    bound = [case["request"]["start"], case["request"]["end"], "completed"]
    if "city" in case["request"]:
        bound.append(case["request"]["city"])
    started = [event["payload"] for event in trace if event["type"] == "SQL_STARTED"]
    checks = {
        "completed": task["status"] == "COMPLETED" and report.get("answer_source") == "analysis",
        "results": analysis.get("months") == expected_rows and analysis.get("totals") == expected_totals,
        "inputs_and_units": analysis.get("window") == case["request"]
        and analysis.get("units") == {"revenue_cents": "分", "order_count": "笔"},
        "bindings": valid_sources(analysis),
        "single_query": len(started) == 1 and kinds.count("SQL_COMPLETED") == 1
        and started[0]["parameters"] == bound * 2
        and kinds.count("ANALYSIS_BINDING_COMMITTED") == 1
        and kinds.index("ANALYSIS_BINDING_COMMITTED") < kinds.index("SQL_STARTED"),
        "isolated": report.get("model_calls") == 0 and not any(
            name.startswith(("MODEL_", "KNOWLEDGE_", "MEMORY_")) for name in kinds
        ),
    }
    return {
        "id": case["id"], "task_id": task["task_id"], "checks": checks,
        "passed": all(checks.values()), "expected_rows": expected_rows, "expected_totals": expected_totals,
        "jct_seconds": task["jct_seconds"],
        "queue_seconds": (task["started_at"] or task["created_at"]) - task["created_at"],
        "binding_seconds": next(
            (event["payload"]["seconds"] for event in trace if event["type"] == "ANALYSIS_BOUND"), None
        ),
        "sql_seconds": stamps["SQL_COMPLETED"] - stamps["SQL_STARTED"] if "SQL_COMPLETED" in stamps else None,
        "terminal_seconds": stamps["TASK_COMPLETED"] - stamps["SQL_COMPLETED"] if "TASK_COMPLETED" in stamps else None,
    }


def evaluate(api, settings, cases, output):
    rows = asyncio.run(run_requests(
        str(api.base_url), [(case["id"], body(case["request"])) for case in cases], settings.task_concurrency
    ))
    db = settings.databases[DATABASE].path
    labels = [verify(case, rows[case["id"]], db) for case in cases]
    write(output / "brief-tasks.json", {"labels": labels, "tasks": rows})
    return labels


def stats(labels):
    return {
        "n": len(labels), "passed": sum(label["passed"] for label in labels),
        **{
            f"{field}_p95": p95([r[field] for r in labels if r[field] is not None])
            for field in ("jct_seconds", "queue_seconds", "binding_seconds", "sql_seconds", "terminal_seconds")
        },
        "failed": [row["id"] for row in labels if not row["passed"]],
    }


def download(api, task, output):
    ident = task["task_id"]
    results = {}
    files = {}
    for name in ("chart.svg", "report.md", "monthly.csv"):
        start = time.monotonic()
        reply = api.get(f"/tasks/{ident}/artifacts/{name}").raise_for_status()
        seconds = time.monotonic() - start
        path = output / name
        path.write_bytes(reply.content)
        files[name] = reply.text
        results[name] = {
            "seconds": seconds, "sha256": sha(path), "content_type": reply.headers["content-type"],
        }
    chart = ElementTree.fromstring(files["chart.svg"])
    csv_rows = list(csv.DictReader(io.StringIO(files["monthly.csv"])))
    monthly = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in task["report"]["analysis"]["months"]
    ]
    assert csv_rows == monthly
    assert chart.tag.endswith("svg") and "成交营收" in files["chart.svg"] and "完成订单量" in files["chart.svg"]
    assert all(metric["rule_sha256"] in files["report.md"] for metric in task["report"]["analysis"]["metrics"])
    return results
