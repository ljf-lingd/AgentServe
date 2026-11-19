"""Translate backend golden cases to the workbench boundary without changing the run."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from v5_expansion_eval import grade
from v6_phase_benchmark import DATABASE, ROOT, dump, file_sha


def prepare(manifest_path):
    from v32_cases import reference_query
    from v4_cases import independent_reference

    manifest = json.loads(manifest_path.read_bytes())
    database = ROOT / f"data/databases/{DATABASE}/{DATABASE}.sqlite"
    database_hash = file_sha(database)
    if manifest["source_sha256"][database.relative_to(ROOT).as_posix()] != database_hash:
        raise ValueError("The gold references require the original data snapshot")
    workload = manifest["workload"]
    rules = {}
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        cities = {city.casefold(): city for city, in conn.execute("SELECT DISTINCT city FROM customers WHERE city IS NOT NULL")}
        for mode in ("metric", "analysis"):
            for case in workload[mode]:
                request = case["request"][mode]
                city = request.get("city")
                if city is None:
                    continue
                canonical = cities.get(city.strip().casefold())
                key = f"{mode}:{case['id']}"
                if canonical is None:
                    rules[key] = {"reason": "unavailable_city", "submitted_city": city}
                elif canonical != city:
                    normalized = {**request, "city": canonical}
                    rule = {"reason": "canonical_city", "canonical_city": canonical}
                    if mode == "analysis":
                        months, totals = independent_reference(database, normalized)
                        rule.update(expected_months=months, expected_totals=totals)
                    else:
                        sql, params = reference_query(normalized, case["label"])
                        cursor = conn.execute(sql, params)
                        rule["expected_query_result"] = {"columns": [column[0] for column in cursor.description],
                                                         "rows": [list(row) for row in cursor]}
                    rules[key] = rule
    for case in workload["knowledge"]:
        if case["request"]["database_id"] != DATABASE and case["label"]["outcome"] == "insufficient_evidence":
            rules[f"knowledge:{case['id']}"] = {"reason": "foreign_database"}
    return {"version": 1, "manifest_sha256": file_sha(manifest_path), "database_sha256": database_hash,
            "source_sha256": file_sha(Path(__file__)), "rules": rules}


def check(rule, case, row, snapshot):
    kind = rule["reason"]
    task = row["task"] or {}
    report = task.get("report") or {}
    if kind == "foreign_database":
        rejection = row.get("rejection") or {}
        return (rejection.get("status") == 422 and
                json.loads(rejection["body"]).get("error", {}).get("code") == "UNKNOWN_DATABASE")
    if kind == "unavailable_city":
        return (task.get("status") == "COMPLETED" and report.get("answer_source") == "workbench"
                and f"当前数据里没有「{rule['submitted_city']}」" in report.get("answer", "")
                and all(report.get(key) is None for key in ("sql", "query_result", "metric", "analysis")))
    mode = row["kind"]
    bound = (report.get(mode) or {}).get("window" if mode == "analysis" else "query") or {}
    expected = {**case, **{k: v for k, v in rule.items() if k.startswith("expected_")}}
    return bound.get("city") == rule["canonical_city"] and grade(expected, row, snapshot) is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare(args.manifest)
    dump(args.output, result)
    print(json.dumps({key: rule["reason"] for key, rule in result["rules"].items()}))


if __name__ == "__main__":
    main()
