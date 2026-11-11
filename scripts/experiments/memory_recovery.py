"""Real SIGKILL checks at memory preparation, SQL and memory commit boundaries."""

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from http_server import http_server


def run_recovery(base, output, policy="simplemem"):
    output.mkdir()
    results = []
    boundaries = [
        ("prepare_unknown", "before_execute", 0),
        ("prepare_committed", "after_commit", 0),
        ("sql", "before_execute", 2),
        ("memory_commit", "after_commit", 4),
    ]
    state_policy = policy in {"state", "state_on_demand"}
    if state_policy:
        boundaries = [
            ("context_frozen_unknown_sql", "before_execute", 0),
            ("sql_model_committed", "after_commit", 0),
            ("sql_tool", "before_execute", 1),
            ("answer_before_state_commit", "after_commit", 2),
        ]
    compiled = base.sql_prompt_version == "evidence_v22"
    if compiled:
        boundaries = [
            ("compiled_sql_before_execute", "before_execute", 0),
            ("compiled_sql_before_commit", "after_execute", 0),
            ("compiled_sql_committed", "after_commit", 0),
            ("compiled_answer_before_state_commit", "after_commit", 1),
        ]
        path = base.databases["business_orders_v1"].path
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            expected_rows = [list(row) for row in db.execute("SELECT COUNT(*) FROM products")]
    else:
        expected_rows = [[1000]]
    for name, stage, step_no in boundaries:
        run = output / name
        run.mkdir()
        settings = base.model_copy(deep=True)
        settings.memory.policy = policy
        settings.state_database = run / "tasks.sqlite"
        config, marker, probe = run / "app.json", run / "marker.json", run / "probe.json"
        config.write_text(settings.model_dump_json(indent=2))
        probe.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "stage": stage,
                    "step_no": step_no,
                    "kind": "sql"
                    if step_no == (0 if compiled else 1 if state_policy else 2)
                    else "model",
                    "audit_path": str(run / "audit.jsonl"),
                    "marker_path": str(marker),
                }
            )
        )
        owned = []
        environment = {"AGENTSERVE_CONFIG": str(config), "AGENTSERVE_RECOVERY_PROBE": str(probe)}
        with http_server(
            "recovery_app:create_recovery_app",
            run,
            "before",
            env=environment,
            app_dir=Path(__file__).parent,
            on_ready=owned.append,
        ) as client:
            session = client.post("/sessions", json={"database_id": "business_orders_v1"})
            session.raise_for_status()
            response = client.post(
                "/tasks",
                json={
                    "database_id": "business_orders_v1",
                    "session_id": session.json()["session_id"],
                    "question": "目录商品种数，输出种数。"
                    if compiled
                    else "所有状态共有多少笔订单？只返回总数。",
                },
            )
            response.raise_for_status()
            tid = response.json()["task_id"]
            deadline = time.monotonic() + 70
            while not marker.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Memory crash boundary not reached")
                state = client.get(f"/tasks/{tid}").json()
                if state["status"] in {"FAILED", "COMPLETED", "CANCELLED"}:
                    raise AssertionError(state)
                time.sleep(0.05)
            before = client.get(f"/tasks/{tid}").json()
            snapshot = client.get(f"/tasks/{tid}/context").json()["context"]
            assert json.loads(marker.read_text())["pid"] == owned[0].pid
            owned[0].kill()
            owned[0].wait(timeout=5)
        with sqlite3.connect(settings.state_database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0] == 0
        probe.write_text(json.dumps({"enabled": False, "audit_path": str(run / "audit.jsonl")}))
        with http_server(
            "recovery_app:create_recovery_app",
            run,
            "after",
            env=environment,
            app_dir=Path(__file__).parent,
        ) as client:
            deadline = time.monotonic() + 80
            while True:
                after = client.get(f"/tasks/{tid}").json()
                if after["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Recovery deadline")
                time.sleep(0.05)
            final_snapshot = client.get(f"/tasks/{tid}/context").json()["context"]
            events = client.get(f"/tasks/{tid}/events?limit=1000").json()["events"]
        expected_failure = name == "prepare_unknown" and not state_policy
        if expected_failure:
            assert (
                after["status"] == "FAILED" and after["error"]["code"] == "MODEL_BUDGET_EXHAUSTED"
            ), after
            assert after["error"]["stage"] == "memory_write"
        else:
            assert after["status"] == "COMPLETED", after
        assert after["report"]["query_result"]["rows"] == expected_rows
        assert after["deadline_at"] == before["deadline_at"]
        assert snapshot is None or snapshot == final_snapshot
        assert after["report"]["memory_calls"] == (0 if state_policy else 2)
        if compiled:
            assert after["report"]["model_calls"] == 1
            assert sum(e["type"] == "STEP_REUSED" for e in events) == (
                2 if step_no == 1 else int(stage == "after_commit")
            )
        with sqlite3.connect(settings.state_database) as connection:
            count = connection.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
            assert count == 0 if expected_failure else count >= 1
            assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        results.append(
            {
                "case": name,
                "passed": True,
                "outcome": "bounded_failure_no_memory" if expected_failure else "completed",
                "snapshot_reused": snapshot is not None,
                "model_calls": after["report"]["model_calls"],
                "step_reuses": sum(e["type"] == "STEP_REUSED" for e in events),
            }
        )
        (run / "evidence.json").write_text(
            json.dumps({"before": before, "after": after, "events": events}, ensure_ascii=False)
        )
    return results
