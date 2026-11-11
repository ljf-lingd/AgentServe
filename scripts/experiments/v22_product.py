"""Exercise the actual CLI with ordered session work, restart, and durable date evidence."""

import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from http_server import http_server
from run_v21 import write


def run_cli(base, output, database):
    output.mkdir()
    settings = base.model_copy(deep=True)
    settings.memory.policy = "state_on_demand"
    settings.state_database = output / "tasks.sqlite"
    settings.databases["business_orders_v1"].path = database
    config = output / "app.json"
    write(config, settings.model_dump(mode="json"))
    env = {"AGENTSERVE_CONFIG": str(config)}
    probe = output / "queue-probe.json"
    write(probe, {"sql_wait_seconds": 1.0})
    questions = (
        "新建命名分析“营收核对”：2026年7月31日Beijing客户completed订单营收合计，单位分，输出营收。",
        "继续“营收核对”，城市替换成Shanghai，其他字段不变。",
        "继续“营收核对”，月份改为上个月，保留输出和排序。",
    )
    children = []

    def submit(client, sid, index):
        path = output / f"cli-{index}.json"
        with (output / f"cli-{index}.log").open("wb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "agentserve",
                    "--database",
                    "business_orders_v1",
                    "--session-id",
                    sid,
                    "--api-url",
                    str(client.base_url).rstrip("/"),
                    "--question",
                    questions[index],
                    "--output",
                    str(path),
                ],
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        children.append(child)
        return child

    try:
        factory = "agentserve.api.app:create_app"
        with http_server(
            "workload_app:create_workload_app",
            output,
            "before",
            app_dir=Path(__file__).parent,
            env={**env, "AGENTSERVE_WORKLOAD_PROBE": str(probe)},
        ) as client:
            response = client.post("/sessions", json={"database_id": "business_orders_v1"})
            response.raise_for_status()
            sid = response.json()["session_id"]
            first = submit(client, sid, 0)
            deadline = time.monotonic() + 15
            while not client.get(f"/sessions/{sid}/history").json()["turns"]:
                if first.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("The first CLI task was not accepted")
                time.sleep(0.02)
            second = submit(client, sid, 1)
            exits = [first.wait(timeout=120), second.wait(timeout=120)]
            history = client.get(f"/sessions/{sid}/history").json()["turns"]
            mismatch = client.post(
                "/tasks",
                json={"session_id": sid, "database_id": "demo_orders", "question": "目录商品种数"},
            ).status_code
        with http_server(factory, output, "after", env=env) as client:
            preserved = client.get(f"/sessions/{sid}/history").json()["turns"] == history
            exits.append(submit(client, sid, 2).wait(timeout=120))
            turns = client.get(f"/sessions/{sid}/history").json()["turns"]
            rows = [client.get(f"/tasks/{turn['task_id']}").json() for turn in turns]
            context = client.get(f"/tasks/{turns[-1]['task_id']}/context").json()["context"]
            events = [
                client.get(f"/tasks/{turn['task_id']}/events").json()["events"] for turn in turns
            ]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)

    first, second, third = rows
    anchor = datetime.fromtimestamp(third["created_at"], ZoneInfo("Asia/Shanghai"))
    end = anchor.replace(day=1)
    start = (end - timedelta(days=1)).replace(day=1)
    date_range = [start.date().isoformat(), end.date().isoformat()]
    state = context["plan"]["state"]
    source = context["selected"][0]
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        values = [
            db.execute(
                "SELECT SUM(quantity * unit_price_cents) FROM order_items JOIN orders USING(order_id) "
                "JOIN customers USING(customer_id) WHERE city=? AND status='completed' AND order_date>=? AND order_date<?",
                (city, a, b),
            ).fetchone()[0]
            for city, (a, b) in zip(
                ("Beijing", "Shanghai", "Shanghai"),
                [("2026-07-31", "2026-08-01")] * 2 + [date_range],
                strict=True,
            )
        ]
    reports = [row["report"] for row in rows]
    null_preserved = any(
        value is None
        and json.loads(report["answer"].split("：", 1)[1].split("。", 1)[0])["rows"] == [[None]]
        for value, report in zip(values, reports, strict=True)
    )
    checks = {
        "three_turns": [t["turn_no"] for t in turns] == [1, 2, 3] and exits == [0, 0, 0],
        "queued": second["created_at"] < first["ended_at"] <= second["started_at"],
        "cross_database_rejected": mismatch == 409,
        "restart_source": preserved and source["task_id"] == second["task_id"],
        "accepted_date_anchor": state["fields"]["date"]["value"] == date_range,
        "compiled_path": all(any(e["type"] == "SQL_COMPILED" for e in trace) for trace in events),
        "independent_sql_oracle": all(
            report["query_result"]["rows"] == [[value]]
            for value, report in zip(values, reports, strict=True)
        ),
        "null_preserved": null_preserved,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "anchor": anchor.isoformat(),
        "relative_range": date_range,
        "exits": exits,
    }
    write(output / "evidence.json", {"result": result, "tasks": rows, "context": context})
    return result
