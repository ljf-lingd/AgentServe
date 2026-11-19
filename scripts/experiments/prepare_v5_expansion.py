"""Freeze the larger v5 business mix using the existing independent references."""

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from agentserve.knowledge.snapshot import KnowledgeSnapshot
from agentserve.workloads import memory_v22
from v31_cases import request_for
from v32_cases import reference_query
from v4_cases import independent_reference
from v6_phase_benchmark import DATABASE, ROOT, TASKS, dump, file_sha

EVALUATION = ROOT / "data/evaluation"
KNOWLEDGE = EVALUATION / "knowledge_v31/contract-acceptance.json"
METRIC = EVALUATION / "metrics_v32/acceptance.json"
ANALYSIS = EVALUATION / "analysis_v4/acceptance.json"
DATABASE_PATH = ROOT / "data/databases/business_orders_v1/business_orders_v1.sqlite"
SNAPSHOT = ROOT / "cache/v32-knowledge-bm25.json"


def read(path):
    return json.loads(path.read_bytes())


def query(connection, sql, parameters=()):
    cursor = connection.execute(sql, parameters)
    return {"columns": [column[0] for column in cursor.description],
            "rows": [list(row) for row in cursor]}


def sql_sessions(connection):
    conversations, labels = memory_v22.build("formal")
    sessions = []
    for session in conversations:
        if len(session["turns"]) != 12:
            continue
        cases = []
        for item in session["turns"]:
            label = labels[item["id"]]
            cases.append({"id": item["id"], "request": {
                "database_id": DATABASE, "question": item["question"],
            }, "label": label, "expected_query_result": None if label["clarify"]
                          else query(connection, label["sql"])})
        sessions.append({"id": session["id"], "family": session["family"], "cases": cases})
    if len(sessions) != 12 or len({session["family"] for session in sessions}) != 12:
        raise ValueError("Expected one complete 12-turn conversation from each SQL family")
    return sessions


def knowledge_cases():
    snapshot = KnowledgeSnapshot(SNAPSHOT, "static", None)
    rows = []
    for case in read(KNOWLEDGE)["cases"]:
        if case.get("scenario") == "fixture":
            continue
        active = snapshot.active(DATABASE, case.get("as_of", snapshot.as_of), case.get("clause_id"))
        for support in case.get("supports", []):
            text = active[support["chunk_id"]]["text"]
            if any(phrase not in text for phrase in support["must"]):
                raise ValueError(f"Missing expected evidence for {case['id']}")
        rows.append({"id": case["id"], "request": request_for(case), "label": case})
    return rows


def freeze(parent_manifest):
    parent = read(parent_manifest)
    if parent["source_sha256"][DATABASE_PATH.relative_to(ROOT).as_posix()] != file_sha(DATABASE_PATH):
        raise ValueError("Expansion requires the same Chinese orders database as the parent matrix")
    if parent["source_sha256"][SNAPSHOT.relative_to(ROOT).as_posix()] != file_sha(SNAPSHOT):
        raise ValueError("Expansion requires the parent knowledge snapshot")
    with closing(sqlite3.connect(DATABASE_PATH.as_uri() + "?mode=ro", uri=True)) as connection:
        sql = sql_sessions(connection)
        metric = []
        for case in read(METRIC):
            request = case["request"]
            reference_sql, parameters = reference_query(request, case["reference"])
            metric.append({"id": case["id"], "family": case["family"],
                           "request": {"database_id": DATABASE, "mode": "metric", "metric": request},
                           "label": case["reference"], "expected_query_result": query(connection, reference_sql, parameters)})
    knowledge = knowledge_cases()
    reports = []
    for case in read(ANALYSIS):
        window = case["request"]
        months, totals = independent_reference(DATABASE_PATH, window)
        reports.append({"id": case["id"], "family": case["family"], "request": {
            "database_id": DATABASE, "mode": "analysis", "analysis": window,
        }, "expected_months": months, "expected_totals": totals})
    sql_rows = [case for session in sql for case in session["cases"]]
    all_rows = [*sql_rows, *knowledge, *metric, *reports]
    original = [{"database_id": DATABASE, **body} for body in TASKS.values()]
    if any(row["request"] in original for row in all_rows):
        raise ValueError("An expansion input repeats a request from the original ten cases")
    counts = {"sql": len(sql_rows), "knowledge": len(knowledge), "metric": len(metric), "analysis": len(reports)}
    if counts != {"sql": 144, "knowledge": 47, "metric": 24, "analysis": 24}:
        raise ValueError(f"Unexpected source workload counts: {counts}")
    return {"version": 1, "parent_freeze_sha256": file_sha(parent_manifest),
            "business_time": parent["business_time"], "cases": counts, "total": len(all_rows),
            "sql_sessions": len(sql), "sql_families": len({item["family"] for item in sql}),
            "sql_distinct_question_strings": len({row["request"]["question"] for row in sql_rows}),
            "sql_labels": dict(Counter(row["label"]["bucket"] for row in sql_rows)),
            "knowledge_labels": dict(Counter(row["label"]["outcome"] for row in knowledge)),
            "source_sha256": {path.relative_to(ROOT).as_posix(): file_sha(path) for path in (
                Path(__file__).resolve(), Path(memory_v22.__file__).resolve(), DATABASE_PATH, SNAPSHOT,
                KNOWLEDGE, METRIC, ANALYSIS, ROOT / "scripts/experiments/v31_cases.py",
                ROOT / "scripts/experiments/v32_cases.py", ROOT / "scripts/experiments/v4_cases.py",
            )}, "workload": {"sql": sql, "knowledge": knowledge, "metric": metric, "analysis": reports}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = freeze(args.parent_manifest)
    dump(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "cases", "total", "sql_sessions", "sql_distinct_question_strings", "knowledge_labels",
    )}))


if __name__ == "__main__":
    main()
