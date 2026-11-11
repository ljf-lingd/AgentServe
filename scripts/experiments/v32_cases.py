"""Independent metric references and isolated version/counterfactual fixtures."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from run_v21 import write
from run_v31 import read, sha
from v31_cases import DATABASE, p95, run_requests

from agentserve.knowledge.metrics import citation_for
from agentserve.knowledge.snapshot import assemble, digest

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "data/knowledge_v32/registry.json"
EVALUATION = ROOT / "data/evaluation/metrics_v32"
FORMULAS = {"revenue": ("revenue_cents", "分"), "orders": ("order_count", "笔")}


def body(request):
    return {"mode": "metric", "database_id": DATABASE, "metric": request}


def reference_query(request, expected):
    """Order-grain reference with a correlated item aggregate and city lookup."""
    group = request.get("group_by")
    dimension = {
        "city": "(SELECT city FROM customers WHERE customer_id = ord.customer_id)",
        "month": "substr(ord.order_date, 1, 7)",
    }
    city = dimension["city"]
    revenue = expected["formula"] == "revenue"
    aggregate = (
        "sum((SELECT sum(unit_price_cents * quantity) FROM order_items"
        " WHERE order_id = ord.order_id))"
        if revenue else "count(ord.order_id)"
    )
    alias, _ = FORMULAS[expected["formula"]]
    projection = f"{aggregate} AS {alias}"
    if group:
        projection = f"{dimension[group]} AS {group}, {projection}"
    predicates = ["ord.order_date BETWEEN ? AND ?", "ord.order_date <> ?"]
    parameters = [request["start"], request["end"], request["end"]]
    placeholders = ", ".join("?" for _ in expected["statuses"])
    predicates.append(f"ord.status IN ({placeholders})")
    parameters.extend(expected["statuses"])
    if revenue:
        predicates.append("EXISTS (SELECT 1 FROM order_items WHERE order_id = ord.order_id)")
    if "city" in request:
        predicates.append(f"{city} = ?")
        parameters.append(request["city"])
    sql = f"SELECT {projection} FROM orders ord WHERE " + " AND ".join(predicates)
    if group:
        sql += f" GROUP BY {group} ORDER BY {group}"
    return sql, parameters


def query(db, sql, parameters):
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        cursor = conn.execute(sql, parameters)
        return [c[0] for c in cursor.description], [list(row) for row in cursor]


def verify(case, row, db, sources, *, row_limit=None):
    report, trace = row["report"] or {}, row["trace"]
    kinds = [event["type"] for event in trace]
    report_binding = report.get("metric", {})
    rule = report_binding.get("rule", {})
    source = report_binding.get("citation", {})
    expected = case["reference"]
    request = case["request"]
    sql, parameters = reference_query(request, expected)
    columns, rows = query(db, sql, parameters)
    truncated = row_limit is not None and len(rows) > row_limit
    if truncated:
        rows = rows[:row_limit]
    result = report.get("query_result") or {}
    started = [e for e in trace if e["type"] == "SQL_STARTED"]
    finished = [e for e in trace if e["type"] == "SQL_COMPLETED"]
    checks = {
        "terminal": row["status"] == "COMPLETED" and report.get("answer_source") == "metric",
        "binding": (
            rule.get("statuses") == expected["statuses"]
            and rule.get("formula") == {"revenue": "revenue", "orders": "order_count"}[expected["formula"]]
            and source.get("version") == expected["version"]
            and report_binding.get("query") == request
        ),
        "citation": source == sources[expected["chunk_id"]] and source["approval_ref"] is not None,
        "unit": rule.get("unit") == expected["unit"] and expected["unit"] in (report.get("answer") or ""),
        "rows": result.get("rows") == rows and result.get("columns") == columns,
        "truncation": result.get("truncated") is truncated
        and ("截断" in (report.get("answer") or "")) is truncated,
        "isolated": report.get("model_calls") == 0 and not any(
            kind.startswith(("MODEL_", "KNOWLEDGE_", "MEMORY_")) for kind in kinds
        ),
        "once": (
            len(started) == len(finished) == 1 and kinds.count("METRIC_BINDING_COMMITTED") == 1
            and kinds.index("METRIC_BINDING_COMMITTED") < kinds.index("SQL_STARTED")
        ),
    }
    actual_sql = report.get("sql") or ""
    bound = [request["start"], request["end"], *expected["statuses"]]
    if "city" in request:
        bound.append(request["city"])
    checks["sql"] = (
        actual_sql.startswith("SELECT ") and actual_sql.count("?") == len(bound)
        and bool(started) and started[0]["payload"].get("parameters") == bound
        and started[0]["payload"].get("sql") == actual_sql
    )
    return {
        "id": case["id"], "task_id": row["task_id"], "checks": checks,
        "passed": all(checks.values()), "reference_sql": sql, "reference_parameters": parameters,
        "expected_rows": rows, "rows": result.get("rows"), "version": source.get("version"),
        "rule_sha256": report_binding.get("rule_sha256"), "citation": source.get("chunk_id"),
        "parameters": started[0]["payload"].get("parameters") if started else None,
        "truncated": result.get("truncated"),
        "jct": row["jct_seconds"],
        "sql_seconds": finished[0]["wall_time"] - started[0]["wall_time"] if started and finished else None,
    }


def evaluate(api, cases, base, registry, output, *, row_limit=None):
    results = asyncio.run(run_requests(
        str(api.base_url), [(case["id"], body(case["request"])) for case in cases],
        base.task_concurrency,
    ))
    source_dir = registry.parent
    sources = {c["chunk_id"]: citation_for(c) for c in assemble(source_dir / "manifest.json")["chunks"]}
    for source in sources.values():
        begin, end = (int(line[1:]) for line in source["anchor"].split("-"))
        path = source_dir / source["path"]
        assert sha(path) == source["source_sha256"]
        assert "\n".join(path.read_text(encoding="utf-8").splitlines()[begin - 1:end]) == source["text"]
    db = base.databases[DATABASE].path
    labels = [
        verify(case, results[case["id"]], db, sources, row_limit=row_limit) for case in cases
    ]
    write(output / "tasks.json", {"labels": labels, "tasks": results})
    return labels


def metric_summary(labels):
    return {
        "passed": sum(c["passed"] for c in labels), "n": len(labels),
        "jct_p95": p95([c["jct"] for c in labels]),
        "sql_p95": p95([c["sql_seconds"] for c in labels if c["sql_seconds"] is not None]),
        "failed": [{k: c[k] for k in ("id", "checks")} for c in labels if not c["passed"]],
    }


def test_registry(directory, variant):
    """Generate fixture-only approvals. Each perturbation changes a named factor."""
    directory.mkdir(parents=True)
    product_rules = read(PRODUCT)["rules"]
    documents, rules = [], []
    if variant == "versions":
        editions = (
            ("1", "2026-01-01", "2026-04-01", ["completed"], True),
            ("2", "2026-04-01", "2026-07-01", ["pending"], True),
            ("draft", "2026-08-01", None, ["cancelled"], False),
        )
    elif variant == "object-a":
        editions = (("A", "2026-01-01", None, ["completed"], True),)
    else:
        editions = (("B", "2026-01-01", None, ["completed", "pending"], True),)
    for rule in product_rules:
        source = rule["source"].split("@")[0]
        for version, start, end, statuses, approved in editions:
            path = directory / f"{source}-{version}.md"
            expression = (
                "SUM(order_items.quantity * order_items.unit_price_cents)"
                if rule["formula"] == "revenue" else "按订单行计数，每笔一次"
            )
            path.write_bytes(
                f"# {rule['aliases'][0]} {version}\n\n仅计 {', '.join(statuses)}。"
                f"{expression}，单位{rule['unit']}。支持城市、月份分组。\n".encode()
            )
            documents.append({
                "id": source, "clause_id": source, "path": path.name, "database_id": DATABASE,
                "version": version, "effective_from": start, "effective_to": end, "purpose": "test",
                "approval_ref": "V3.2/synthetic-fixture" if approved else None, "source_sha256": sha(path),
            })
            if approved:
                rules.append({**rule, "source": f"{source}@{version}#1", "statuses": statuses})
    manifest = directory / "manifest.json"
    write(manifest, {"purpose": "test", "as_of": "2026-04-01", "static": [], "documents": documents})
    chunks = {c["chunk_id"]: c for c in assemble(manifest)["chunks"]}
    for rule in rules:
        rule["source_binding_sha256"] = digest({
            "rule": {k: v for k, v in rule.items() if k != "source_binding_sha256"},
            "citation": citation_for(chunks[rule["source"]]),
        })
    registry = directory / "registry.json"
    write(registry, {"format": 1, "rules": rules})
    return registry


def probe_cases(variant):
    if variant == "versions":
        windows = (
            ("1", "2026-01-01", "2026-04-01", ["completed"]),
            ("2", "2026-04-01", "2026-07-01", ["pending"]),
        )
    else:
        windows = ((
            "A" if variant == "object-a" else "B", "2026-01-01", "2026-04-01",
            ["completed"] if variant == "object-a" else ["completed", "pending"],
        ),)
    return [
        {
            "id": f"{formula}-{version}",
            "request": {"name": name, "start": start, "end": end},
            "reference": {
                "formula": formula, "statuses": statuses, "version": version,
                "unit": FORMULAS[formula][1], "chunk_id": f"{source}@{version}#1",
            },
        }
        for version, start, end, statuses in windows
        for formula, source, name in (
            ("revenue", "settled-revenue", "成交营收"),
            ("orders", "completed-orders", "完成订单量"),
        )
    ]
