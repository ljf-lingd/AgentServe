"""Independent SQL references and fixed evaluation questions for UCI Online Retail II."""

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import httpx

DATABASE = "online_retail_ii"
SALES = "substr(upper(invoice),1,1) != 'C' AND quantity > 0 AND unit_price_millionths > 0"
CANCEL = "substr(upper(invoice),1,1) = 'C' AND quantity < 0 AND unit_price_millionths > 0"
MONTHS = {
    "2009-12": "2010-01", "2010-01": "2010-02", "2010-02": "2010-03", "2010-03": "2010-04",
    "2010-04": "2010-05", "2010-06": "2010-07", "2010-07": "2010-08", "2010-08": "2010-09",
    "2010-09": "2010-10", "2010-10": "2010-11", "2010-11": "2010-12", "2010-12": "2011-01",
    "2011-01": "2011-02", "2011-02": "2011-03", "2011-03": "2011-04", "2011-04": "2011-05",
    "2011-05": "2011-06", "2011-06": "2011-07", "2011-07": "2011-08", "2011-08": "2011-09",
    "2011-09": "2011-10", "2011-10": "2011-11", "2011-11": "2011-12", "2011-12": "2012-01",
    "2012-01": "2012-02",
}
COUNTRIES = {
    "英国": "United Kingdom", "德国": "Germany", "法国": "France", "荷兰": "Netherlands",
    "澳大利亚": "Australia", "西班牙": "Spain", "比利时": "Belgium", "瑞士": "Switzerland",
    "丹麦": "Denmark", "美国": "USA", "日本": "Japan", "巴西": "Brazil",
}
METRICS = {
    "销售额": (SALES, "SUM(quantity * unit_price_millionths) / 1000000.0"),
    "取消金额": (CANCEL, "SUM(-quantity * unit_price_millionths) / 1000000.0"),
    "销售发票数": ("substr(upper(invoice),1,1) != 'C'", "COUNT(DISTINCT invoice)"),
    "取消发票数": ("substr(upper(invoice),1,1) = 'C'", "COUNT(DISTINCT invoice)"),
    "有 ID 的去重客户数": ("customer_id IS NOT NULL", "COUNT(DISTINCT customer_id)"),
}


def title(month):
    return f"{month[:4]} 年 {int(month[5:])} 月"


def case(country, month, metric, question=None):
    clause, projection = METRICS[metric]
    return {
        "question": question or f"{country} {title(month)}的{metric}是多少？只返回这个值。",
        "sql": f"SELECT {projection} FROM retail_lines WHERE {clause} "
               "AND country = ? AND invoice_date >= ? AND invoice_date < ?",
        "params": [COUNTRIES[country], month + "-01", MONTHS[month] + "-01"],
        "ordered": False,
    }


def rank(month, metric):
    clause, projection = METRICS[metric]
    return {
        "question": f"{title(month)}{metric}最高的 5 个国家是哪些？"
                    f"仅返回国家（原文）和{metric}，同值按国家原文升序。",
        "sql": f"SELECT country, {projection} AS amount FROM retail_lines "
               f"WHERE {clause} AND invoice_date >= ? AND invoice_date < ? "
               "GROUP BY country ORDER BY amount DESC, country LIMIT 5",
        "params": [month + "-01", MONTHS[month] + "-01"],
        "ordered": True,
    }


def countries(month):
    return {
        "question": f"{title(month)}有交易的国家有几个？只返回国家数。",
        "sql": "SELECT COUNT(DISTINCT country) FROM retail_lines "
               "WHERE invoice_date >= ? AND invoice_date < ?",
        "params": [month + "-01", MONTHS[month] + "-01"],
        "ordered": False,
    }


def conversation(country, next_country, month, next_month, metric):
    return [
        case(country, month, metric),
        case(next_country, month, metric, f"那{next_country}呢？"),
        case(next_country, next_month, metric, f"再看{title(next_month)}呢？"),
    ]


def workloads(split):
    if split == "development":
        return (
            [[case(c, m, metric)] for c, m, metric in (
                ("澳大利亚", "2011-06", "销售额"), ("西班牙", "2011-02", "取消金额"),
                ("荷兰", "2010-04", "销售发票数"), ("比利时", "2010-02", "取消发票数"),
            )] + [[rank("2010-07", "取消金额")], [countries("2009-12")]]
            + [conversation("瑞士", "法国", "2010-06", "2010-08", "取消金额")]
        )
    singles = [
        ("德国", "2011-05", "销售额"), ("法国", "2010-11", "销售额"),
        ("澳大利亚", "2011-03", "销售额"), ("荷兰", "2010-12", "销售额"),
        ("瑞士", "2011-08", "取消金额"), ("比利时", "2011-07", "取消金额"),
        ("德国", "2010-10", "取消金额"), ("英国", "2010-09", "取消金额"),
        ("西班牙", "2011-09", "销售发票数"), ("英国", "2011-08", "销售发票数"),
        ("法国", "2010-10", "销售发票数"), ("德国", "2011-04", "销售发票数"),
        ("英国", "2009-12", "取消发票数"), ("荷兰", "2011-12", "取消发票数"),
        ("澳大利亚", "2010-03", "取消发票数"), ("德国", "2011-02", "取消发票数"),
        ("英国", "2011-07", "有 ID 的去重客户数"), ("瑞士", "2010-12", "有 ID 的去重客户数"),
        ("德国", "2011-01", "有 ID 的去重客户数"), ("法国", "2011-05", "有 ID 的去重客户数"),
    ]
    return (
        [[case(*item)] for item in singles]
        + [[rank("2011-10", "销售额")], [rank("2010-12", "取消金额")],
           [countries("2011-04")], [countries("2010-10")]]
        + [conversation(*item) for item in (
            ("英国", "德国", "2011-11", "2011-10", "取消金额"),
            ("法国", "荷兰", "2011-08", "2011-09", "销售额"),
            ("西班牙", "瑞士", "2011-04", "2011-05", "取消金额"),
            ("比利时", "澳大利亚", "2011-11", "2011-12", "销售额"),
            ("德国", "法国", "2010-06", "2010-07", "销售发票数"),
            ("英国", "荷兰", "2011-06", "2011-07", "取消发票数"),
            ("澳大利亚", "德国", "2010-11", "2010-12", "销售额"),
            ("法国", "比利时", "2011-01", "2011-02", "取消发票数"),
        )]
    )


def gold(database, sessions):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        return [
            [{**turn, "expected": db.execute(turn["sql"], turn["params"]).fetchall()} for turn in s]
            for s in sessions
        ]


def match(result, reference):
    if not result or result["status"] != "ok" or result["truncated"]:
        return False
    def key(row):
        return tuple(round(value, 6) if isinstance(value, float) else value for value in row)
    observed = list(map(key, result["rows"]))
    expected = list(map(key, reference["expected"]))
    return observed == expected if reference["ordered"] else Counter(observed) == Counter(expected)


def replay_session(client, sequence):
    reply = client.post("/sessions", json={"database_id": DATABASE})
    reply.raise_for_status()
    session = reply.json()["session_id"]
    results = []
    for turn in sequence:
        started = time.monotonic()
        submit = client.post("/tasks", json={
            "database_id": DATABASE, "session_id": session, "question": turn["question"],
        })
        submit.raise_for_status()
        task_id = submit.json()["task_id"]
        for _ in range(600):
            response = client.get(f"/tasks/{task_id}")
            response.raise_for_status()
            task = response.json()
            if task["status"] in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                break
            time.sleep(0.15)
        report = task.get("report") or {}
        results.append({
            **turn, "task_id": task_id, "status": task["status"], "error": task.get("error"),
            "correct": task["status"] == "COMPLETED" and match(report.get("query_result"), turn),
            "answer": report.get("answer"), "result": report.get("query_result"),
            "client_seconds": round(time.monotonic() - started, 4),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "formal"), default="formal")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    sessions = gold(args.database, workloads(args.split))
    results = {"split": args.split, "sessions": sessions}
    frozen = json.dumps(sessions, ensure_ascii=False, sort_keys=True).encode()
    results["gold_sha256"] = hashlib.sha256(frozen).hexdigest()
    results["n"] = sum(map(len, sessions))
    if args.base_url:
        with httpx.Client(base_url=args.base_url, headers={"Origin": args.base_url}, trust_env=False, timeout=15) as client:
            runs = [replay_session(client, s) for s in sessions]
        results.update(
            sessions=runs, correct=sum(t["correct"] for s in runs for t in s),
            complete_conversations=sum(all(t["correct"] for t in s) for s in runs if len(s) > 1),
        )
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in results.items() if k != "sessions"})


if __name__ == "__main__":
    main()

