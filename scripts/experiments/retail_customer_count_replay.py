"""Replay the UCI customer-column repair on archived SQL and unseen country-month pairs."""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from retail_history import exact_trace
from retail_replay import COUNTRIES, DATABASE, case, gold, match, title, workloads

from agentserve.agents.retail_sql import bind_retail_customer_count, use_retail_date_index
from agentserve.tools.sqlite import SQLTool

CUSTOMERS = "有 ID 的去重客户数"
PHRASINGS = ("有 ID 的不同客户数是多少？", "去重后的客户数有几位？只返回数字。")
MATRIX = tuple(f"matrix-b{block}-{arm}" for block in (1, 2)
               for arm in ("ar", "session", "suffix-local", "suffix-global"))


async def execute(tool, sql):
    return (await tool.execute(DATABASE, sql, deadline=time.monotonic() + 10)).to_dict()


async def holdout(tool, database, smoke):
    months = ("2010-01", "2010-02", "2010-03", "2010-04", "2010-06", "2010-07")
    pairs = [(country, month) for country in COUNTRIES for month in months]
    if smoke:
        pairs = pairs[::13]
    sessions = []
    for index, (country, month) in enumerate(pairs):
        question = f"{country} {title(month)}的{PHRASINGS[index % len(PHRASINGS)]}"
        sessions.append([case(country, month, CUSTOMERS, question), case(country, month, "销售发票数")])
    results = []
    for customer, invoice in gold(database, sessions):
        country, first, following = customer["params"]
        query = ("SELECT COUNT(DISTINCT invoice) FROM retail_transactions "
                 f"WHERE country = '{country}' AND invoice_date >= '{first}' AND invoice_date < '{following}'")
        repaired = bind_retail_customer_count(customer["question"], query)
        control = query + " AND is_cancellation = 0"
        baseline = await execute(tool, query)
        candidate = await execute(tool, repaired)
        control_result = await execute(tool, control)
        results.append({
            "country": country, "month": first[:7], "question": customer["question"],
            "expected": customer["expected"], "before": baseline["rows"], "after": candidate["rows"],
            "before_correct": match(baseline, customer), "after_correct": match(candidate, customer),
            "customer_sql_changed": repaired != query,
            "invoice_control_correct": match(control_result, invoice),
            "invoice_control_unchanged": bind_retail_customer_count(invoice["question"], control) == control,
        })
    return results


async def archived(tool, database, root, smoke):
    reference = gold(database, workloads("formal"))
    groups = {}
    for label in MATRIX[:1] if smoke else MATRIX:
        folder = root / label / "attempt-01"
        traces = json.loads((folder / "retail.json").read_bytes())["sessions"]
        ledger_path = (folder / "retail.sqlite").resolve()
        results = []
        with closing(sqlite3.connect(ledger_path.as_uri() + "?mode=ro", uri=True)) as ledger:
            for session, sequence in zip(traces, reference, strict=True):
                for row, turn in zip(session["turns"], sequence, strict=True):
                    if smoke and CUSTOMERS not in turn["question"]:
                        continue
                    assert row["question"] == turn["question"] and row["status"] == "COMPLETED"
                    trace = next(trace for trace in reversed(row["trace"]) if trace["sql"])
                    replies = ledger.execute(
                        "SELECT output_json FROM steps WHERE task_id=? AND kind='model' AND step_no=?",
                        (row["task_id"], trace["step_no"]),
                    ).fetchone()
                    reply = json.loads(replies[0])["reply"]
                    assert exact_trace(trace) and reply["request_id"] == trace["id"]
                    assert reply["sql"] == trace["sql"] and reply["finish_reason"] == trace["finish"]
                    initial = use_retail_date_index(reply["sql"])
                    assert initial == row["sql"]
                    patched = use_retail_date_index(
                        bind_retail_customer_count(turn["question"], reply["sql"])
                    )
                    baseline = await execute(tool, initial)
                    assert all(baseline[key] == row["result"][key]
                               for key in ("status", "rows", "truncated"))
                    candidate = baseline if patched == initial else await execute(tool, patched)
                    before, after = match(baseline, turn), match(candidate, turn)
                    assert before == row["correct"]
                    results.append({
                        "task_id": row["task_id"], "question": turn["question"], "request_id": trace["id"],
                        "before_correct": before, "after_correct": after, "changed": initial != patched,
                        **({"before": baseline["rows"], "after": candidate["rows"],
                            "expected": turn["expected"], "model_sql": reply["sql"], "patched_sql": patched}
                           if initial != patched or not after else {}),
                    })
        groups[label] = results
    return groups


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def main(args):
    database = args.database.resolve(strict=True)
    fingerprint = file_sha(database)
    tool = SQLTool({DATABASE: database})
    try:
        held = await holdout(tool, database, args.smoke)
        groups = await archived(tool, database, args.archive_root, args.smoke) if args.archive_root else {}
    finally:
        tool.close()
    assert file_sha(database) == fingerprint
    summary = {
        "database_sha256": fingerprint, "new_model_requests": 0, "smoke": args.smoke,
        "holdout_pairs": len(held), "holdout_baseline_correct": sum(r["before_correct"] for r in held),
        "holdout_candidate_correct": sum(r["after_correct"] for r in held),
        "holdout_invoice_controls": sum(r["invoice_control_correct"] for r in held),
        "holdout_customer_sql_changed": sum(r["customer_sql_changed"] for r in held),
        "holdout_invoice_sql_unchanged": sum(r["invoice_control_unchanged"] for r in held),
        "archive": {label: {k: sum(r[k] for r in rows)
                            for k in ("before_correct", "after_correct", "changed")} | {"n": len(rows)}
                    for label, rows in groups.items()},
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"summary": summary, "holdout": held, "archive": groups},
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary), flush=True)
    return not (all(r["customer_sql_changed"] and r["after_correct"] and r["invoice_control_correct"]
                    and r["invoice_control_unchanged"] for r in held)
                and all(row["after_correct"] and row["changed"] == (not row["before_correct"])
                        for rows in groups.values() for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
