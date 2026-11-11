"""Run real API process interruptions, task controls and CLI against the V3.1 snapshot."""

import json
import subprocess
import sys
import time
from pathlib import Path

from run_v21 import write
from v31_cases import DATABASE, TERMINAL

PROBE_BODY = {
    "mode": "knowledge",
    "database_id": DATABASE,
    "as_of": "2026-11-05",
    "clause_id": "revenue",
    "question": "成交营收按什么历史价格计算？",
}


def terminal(api, task_id):
    limit = time.monotonic() + 100
    while time.monotonic() < limit:
        row = api.get(f"/tasks/{task_id}").raise_for_status().json()
        if row["status"] in TERMINAL:
            return row
        time.sleep(0.08)
    raise TimeoutError("Task did not finish after knowledge probe")


def events(api, task_id):
    return api.get(f"/tasks/{task_id}/events?limit=1000").raise_for_status().json()["events"]


def submit(api, body=PROBE_BODY, **kwargs):
    return api.post("/tasks", json=body, **kwargs).raise_for_status().json()["task_id"]


def probe_options(directory, stage, *, enabled=True, kind="knowledge", step_no=-1):
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "probe.json"
    marker = directory / "marker.json"
    write(
        config,
        {
            "enabled": enabled,
            "stage": stage,
            "kind": kind,
            "step_no": step_no,
            "audit_path": str(directory / "audit.jsonl"),
            "marker_path": str(marker),
        },
    )
    return marker, {
        "factory": "recovery_app:create_recovery_app",
        "app_dir": Path(__file__).parent,
        "env": {"AGENTSERVE_RECOVERY_PROBE": str(config)},
    }


def wait_marker(api, marker, task_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if marker.is_file() and marker.stat().st_size:
            return json.loads(marker.read_text(encoding="utf-8"))
        row = api.get(f"/tasks/{task_id}").raise_for_status().json()
        if row["status"] in TERMINAL:
            raise AssertionError(row["error"])
        time.sleep(0.04)
    raise TimeoutError("Knowledge probe boundary not reached")


def probe_recovery(base, snapshots, strategy, output):
    from run_v31 import application

    output.mkdir()
    product = snapshots / "product.json"
    checks = {}
    headers = {"Idempotency-Key": "v31-knowledge-recovery"}
    for name, stage in (
        ("before_context", "before_execute"),
        ("after_context", "after_commit"),
        ("changed_snapshot", "after_commit"),
    ):
        directory = output / name
        changed = name == "changed_snapshot"
        if changed:
            with application(base, product, "product", strategy, directory, phase="history") as api:
                historical_id = submit(api)
                historical = terminal(api, historical_id)
                assert historical["status"] == "COMPLETED"
        marker, options = probe_options(directory, stage)
        owned = []
        with application(
            base,
            product,
            "product",
            strategy,
            directory,
            phase="before",
            on_ready=owned.append,
            **options,
        ) as api:
            task_id = submit(api, headers=headers)
            assert wait_marker(api, marker, task_id)["pid"] == owned[0].pid
            assert submit(api, headers=headers) == task_id
            assert (
                api.post(
                    "/tasks", json={**PROBE_BODY, "as_of": "2026-11-06"}, headers=headers
                ).status_code
                == 409
            )
            before = api.get(f"/tasks/{task_id}").raise_for_status().json()
            owned[0].kill()
            owned[0].wait(timeout=5)
        snapshot = snapshots / "versions.json" if changed else product
        with application(
            base, snapshot, "test" if changed else "product", strategy, directory, phase="after"
        ) as api:
            assert submit(api, headers=headers) == task_id
            after = terminal(api, task_id)
            trace = events(api, task_id)
            if changed:
                assert after["status"] == "FAILED" and after["error"]["code"] == "CONFIG_CHANGED"
                assert api.get(f"/tasks/{historical_id}").json()["report"] == historical["report"]
            else:
                assert after["status"] == "COMPLETED" and after["report"]["model_calls"] == 1
                assert after["report"]["knowledge"]["as_of"] == PROBE_BODY["as_of"]
                assert after["report"]["knowledge"]["outcome"] == "answered"
            assert after["deadline_at"] == before["deadline_at"]
            kinds = [e["type"] for e in trace]
            assert kinds.count("KNOWLEDGE_CONTEXT_REUSED") == int(name == "after_context")
            assert kinds.count("KNOWLEDGE_CONTEXT_COMMITTED") == 1
            assert "SQL_STARTED" not in kinds
        checks[name] = {
            "passed": True,
            "task_id": task_id,
            "status": after["status"],
            "context_reused": name == "after_context",
        }
        write(directory / "evidence.json", {"before": before, "after": after, "trace": trace})

    for name, stage in (("cancel", "after_commit"), ("timeout", "before_execute")):
        directory = output / name
        marker, options = probe_options(directory, stage)
        settings = base.model_copy(deep=True)
        if name == "timeout":
            settings.task_timeout_seconds = 2
        with application(settings, product, "product", strategy, directory, **options) as api:
            task_id = submit(api)
            wait_marker(api, marker, task_id)
            if name == "cancel":
                assert api.post(f"/tasks/{task_id}/cancel").raise_for_status().json()["changed"]
            row = terminal(api, task_id)
            trace = events(api, task_id)
            if name == "cancel":
                assert row["status"] == "CANCELLED"
                assert not api.post(f"/tasks/{task_id}/cancel").json()["changed"]
            else:
                assert row["status"] == "FAILED" and row["error"]["code"] == "TASK_TIMEOUT"
            assert not any(e["type"] in {"MODEL_STARTED", "SQL_STARTED"} for e in trace)
        checks[name] = {"passed": True, "task_id": task_id, "status": row["status"]}
        write(directory / "evidence.json", {"task": row, "trace": trace})

    config = output / "after_context/after-app.json"
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentserve",
            "--config",
            str(config),
            "--mode",
            "knowledge",
            "--database",
            DATABASE,
            "--as-of",
            PROBE_BODY["as_of"],
            "--clause-id",
            PROBE_BODY["clause_id"],
            "--question",
            PROBE_BODY["question"],
        ],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=60,
    )
    report = json.loads(cli.stdout)
    assert report["status"] == "COMPLETED" and report["sql"] is None
    assert report["knowledge"]["as_of"] == PROBE_BODY["as_of"]
    assert report["knowledge"]["outcome"] == "answered" and report["knowledge"]["citations"]
    checks["cli"] = {"passed": True, "task_id": report["task_id"]}
    write(output / "cli.json", report)
    return checks
