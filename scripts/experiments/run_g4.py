"""Real Qwen + SQLite recovery experiments, terminating only API processes owned by this harness."""

import hashlib
import importlib.metadata
import json
import sqlite3
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from http_server import http_server
from model_server import temporary_model

from agentserve.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_terminal(client, task_id):
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        response = client.get(f"/tasks/{task_id}")
        response.raise_for_status()
        task = response.json()
        if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return task
        time.sleep(0.02)
    raise TimeoutError("Recovery task did not reach terminal state")


def disk_snapshot(database):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return {
            "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
            "tasks": [dict(r) for r in conn.execute("SELECT * FROM tasks")],
            "steps": [dict(r) for r in conn.execute("SELECT * FROM steps ORDER BY step_no")],
            "attempts": [
                dict(r) for r in conn.execute("SELECT * FROM attempts ORDER BY attempt_id")
            ],
        }


def run_case(endpoint, proxy_audit, output, *, stage, step_no, kind, mode="resume", crashes=1):
    output.mkdir()
    settings = Settings.load(ROOT / "configs/app.demo.json")
    settings.base_url, settings.state_database = endpoint, output / "tasks.sqlite"
    settings.task_concurrency, settings.llm_concurrency = 1, 1
    if mode == "deadline":
        settings.task_timeout_seconds = 5
    elif mode == "budget":
        settings.max_model_calls, settings.max_model_retries = 2, 2
    config = output / "app.json"
    config.write_text(settings.model_dump_json(indent=2))
    before_proxy = len(read_lines(proxy_audit))
    task_id = None
    original = None
    snapshots = []
    for crash_index in range(crashes):
        marker = output / f"marker-{crash_index}.json"
        probe = output / f"probe-{crash_index}.json"
        write(
            probe,
            {
                "enabled": True,
                "stage": stage,
                "step_no": step_no,
                "kind": kind,
                "audit_path": str(output / "step-audit.jsonl"),
                "marker_path": str(marker),
            },
        )
        owned = []
        with http_server(
            "recovery_app:create_recovery_app",
            output,
            f"before-crash-{crash_index}",
            env={"AGENTSERVE_CONFIG": str(config), "AGENTSERVE_RECOVERY_PROBE": str(probe)},
            app_dir=ROOT / "scripts/experiments",
            on_ready=owned.append,
        ) as client:
            if task_id is None:
                response = client.post(
                    "/tasks",
                    json={"database_id": "demo_orders", "question": "一共有多少位客户？"},
                    headers={"Idempotency-Key": "g4-recovery-case"},
                )
                assert response.status_code == 202, response.text
                task_id = response.json()["task_id"]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not marker.exists():
                task = client.get(f"/tasks/{task_id}").json()
                if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    raise RuntimeError(f"Task ended before crash marker: {task}")
                time.sleep(0.01)
            assert marker.exists(), "Crash boundary not reached"
            marked = json.loads(marker.read_text())
            assert marked["pid"] == owned[0].pid, "Marker does not belong to owned process"
            task = client.get(f"/tasks/{task_id}").json()
            if original is None:
                original = task
            write(output / f"before-crash-{crash_index}-task.json", task)
            write(
                output / f"before-crash-{crash_index}-steps.json",
                client.get(f"/tasks/{task_id}/steps").json(),
            )
            if mode == "cancel":
                assert client.post(f"/tasks/{task_id}/cancel").status_code == 202
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    events = client.get(f"/tasks/{task_id}/events?limit=1000").json()["events"]
                    if any(e["type"] == "APPLICATION_EXECUTION_STOPPED" for e in events):
                        break
                    time.sleep(0.01)
                else:
                    raise TimeoutError("Cancelled application execution did not stop")
            # This Popen was created by http_server for this exact phase. The model is a separate process.
            owned[0].kill()
            owned[0].wait(timeout=5)
        snapshot = disk_snapshot(settings.state_database)
        assert snapshot["quick_check"] == "ok"
        snapshots.append(snapshot)
        write(output / f"disk-after-kill-{crash_index}.json", snapshot)
    if mode == "deadline":
        time.sleep(max(0, original["deadline_at"] - time.time()) + 0.05)
    probe = output / "probe-restarted.json"
    write(probe, {"enabled": False, "audit_path": str(output / "step-audit.jsonl")})
    with http_server(
        "recovery_app:create_recovery_app",
        output,
        "restarted",
        env={"AGENTSERVE_CONFIG": str(config), "AGENTSERVE_RECOVERY_PROBE": str(probe)},
        app_dir=ROOT / "scripts/experiments",
    ) as client:
        task = wait_terminal(client, task_id)
        history = client.get(f"/tasks/{task_id}/steps").json()
        events = client.get(f"/tasks/{task_id}/events?limit=1000").json()["events"]
        replay = client.post(
            "/tasks",
            json={"database_id": "demo_orders", "question": "一共有多少位客户？"},
            headers={"Idempotency-Key": "g4-recovery-case"},
        )
        assert replay.status_code == 200 and replay.json()["task_id"] == task_id
    proxy_records = read_lines(proxy_audit)[before_proxy:]
    write(output / "proxy-receipts.json", proxy_records)
    write(output / "after-restart.json", {"task": task, "history": history, "events": events})
    audit = read_lines(output / "step-audit.jsonl")
    actual_model_calls = sum(r["type"] == "UPSTREAM_COMPLETED" for r in proxy_records)
    actual_sql_runs = sum(r["kind"] == "sql" and r["stage"] == "after_execute" for r in audit)
    assert (
        task["started_at"] == original["started_at"]
        and task["deadline_at"] == original["deadline_at"]
    )
    if mode == "cancel":
        assert task["status"] == "CANCELLED" and actual_model_calls == 1 and actual_sql_runs == 1
    elif mode == "deadline":
        assert (
            task["status"] == "FAILED"
            and task["error"]["code"] == "TASK_TIMEOUT"
            and actual_model_calls == 1
        )
    elif mode in {"budget", "retry_budget"}:
        expected = "MODEL_BUDGET_EXHAUSTED" if mode == "budget" else "MODEL_RETRY_EXHAUSTED"
        assert task["status"] == "FAILED" and task["error"]["code"] == expected
        assert history["budget"]["model_calls"] == 2 and actual_model_calls == 0
    else:
        assert task["status"] == "COMPLETED" and task["report"]["query_result"]["rows"] == [[6]]
        assert task["report"]["answer"].strip()
        expected_models = 3 if (kind, step_no, stage) == ("model", 0, "after_execute") else 2
        assert (
            actual_model_calls == expected_models
            and history["budget"]["model_calls"] == expected_models
        )
        expected_sql = 2 if kind == "sql" and stage == "after_execute" else 1
        assert actual_sql_runs == expected_sql
    result = {
        "case": output.name,
        "stage": stage,
        "step_no": step_no,
        "kind": kind,
        "mode": mode,
        "crashes": crashes,
        "status": task["status"],
        "error": task["error"],
        "actual_model_completions": actual_model_calls,
        "actual_sql_executions": actual_sql_runs,
        "budget": history["budget"],
        "attempts": len(history["attempts"]),
        "interrupted_attempts": sum(a["status"] == "INTERRUPTED" for a in history["attempts"]),
        "reused_steps": sum(e["type"] == "STEP_REUSED" for e in events),
        "started_at_preserved": True,
        "deadline_preserved": True,
        "passed": True,
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    output = ROOT / "results/slice4" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    paths = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted(Path(__file__).parent.glob("*.py"))
        + [ROOT / "configs/app.demo.json", ROOT / "pyproject.toml"]
    )
    settings = Settings.load(ROOT / "configs/app.demo.json")
    database = settings.databases["demo_orders"].path
    data_paths = [
        database,
        settings.databases["demo_orders"].schema_path,
        ROOT / "data/processed/demo_orders/manifest.json",
    ]
    report = {
        "run_id": output.name,
        "state": "prepared",
        "purpose": "owned_process_SIGKILL_recovery_with_real_Qwen",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "data_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_paths
        },
        "model_manifest": json.loads(
            (ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json").read_text()
        ),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["fastapi", "httpx", "aiosqlite", "pydantic", "uvicorn"]
        },
        "cases": [],
    }
    write(output / "summary.json", report)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    try:
        with temporary_model(output) as model_url:
            proxy_config = output / "proxy.json"
            proxy_audit = output / "model-audit.jsonl"
            write(proxy_config, {"target_url": model_url, "audit_path": str(proxy_audit)})
            with http_server(
                "fault_proxy:create_proxy",
                output,
                "model-audit-proxy",
                env={"AGENTSERVE_PROXY_CONFIG": str(proxy_config)},
                app_dir=ROOT / "scripts/experiments",
            ) as proxy:
                endpoint = str(proxy.base_url).rstrip("/") + "/v1"
                specs = []
                for stage in ("before_execute", "after_execute", "after_commit"):
                    for repetition in range(1, 4):
                        specs.append((f"sql-{stage}-{repetition}", stage, 1, "sql", "resume", 1))
                specs.extend(
                    [
                        ("model-uncommitted", "after_execute", 0, "model", "resume", 1),
                        ("answer-committed", "after_commit", 2, "model", "resume", 1),
                        ("cancel-checkpoint", "after_commit", 1, "sql", "cancel", 1),
                        ("expired-deadline", "after_commit", 1, "sql", "deadline", 1),
                        ("global-budget", "before_execute", 0, "model", "budget", 2),
                        ("step-retry-budget", "before_execute", 0, "model", "retry_budget", 2),
                    ]
                )
                for name, stage, step_no, kind, mode, crashes in specs:
                    result = run_case(
                        endpoint,
                        proxy_audit,
                        output / name,
                        stage=stage,
                        step_no=step_no,
                        kind=kind,
                        mode=mode,
                        crashes=crashes,
                    )
                    report["cases"].append(result)
                    report["state"] = "running"
                    write(output / "summary.json", report)
        report.update(
            state="completed",
            database_unchanged=before == hashlib.sha256(database.read_bytes()).hexdigest(),
        )
    except Exception as error:
        report.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(output / "summary.json", report)
    print(
        json.dumps(
            {
                "run_id": output.name,
                "state": report["state"],
                "cases": len(report["cases"]),
                "passed": sum(c["passed"] for c in report["cases"]),
                "database_unchanged": report["database_unchanged"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
