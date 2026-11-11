"""Exercise the report contract and persistence through real API process restarts."""

import json
import subprocess
import sys

from run_v21 import write
from run_v31 import application
from v31_cases import DATABASE
from v31_recovery import events, probe_options, submit, terminal, wait_marker
from v32_cases import test_registry
from v32_controls import settings_for
from v4_cases import body, download

WINDOW = {"start": "2026-01-01", "end": "2026-04-01", "city": "Shanghai"}
REQUEST = body(WINDOW)


def boundaries(base, snapshot, output):
    checks = {}
    registry = test_registry(output / "version-fixture", "versions")
    fixtures = settings_for(base, registry, "test")
    with application(fixtures, snapshot, "product", "bm25", output / "versions") as api:
        for key, window in (
            ("cross_version", {"start": "2026-03-01", "end": "2026-05-01"}),
            ("before_approved", {"start": "2025-12-01", "end": "2026-01-01"}),
        ):
            task_id = submit(api, body(window))
            row = terminal(api, task_id)
            trace = events(api, task_id)
            checks[key] = (
                row["status"] == "FAILED" and row["error"]["code"] == "METRIC_WINDOW_UNCOVERED"
                and not any(e["type"] in {"ANALYSIS_BINDING_COMMITTED", "SQL_STARTED"} for e in trace)
            )
            write(output / f"{key}.json", {"task": row, "trace": trace})
        for key, window in (
            ("not_month_start", {"start": "2026-02-02", "end": "2026-04-01"}),
            ("over_12_months", {"start": "2026-01-01", "end": "2027-02-01"}),
            ("empty_window", {"start": "2026-01-01", "end": "2026-01-01"}),
            ("unexpected_sql", {**WINDOW, "sql": "DELETE FROM orders"}),
        ):
            checks[key] = api.post("/tasks", json=body(window)).status_code == 422
        checks["session_rejected"] = (
            api.post("/tasks", json={**REQUEST, "session_id": "s"}).status_code == 422
        )
        wrong = terminal(api, submit(api, {**REQUEST, "database_id": "demo_orders"}))
        checks["wrong_database"] = wrong["error"]["code"] == "ANALYSIS_DATABASE_UNSUPPORTED"
    truncated = base.model_copy(update={"row_limit": 1})
    with application(truncated, snapshot, "product", "bm25", output / "truncation") as api:
        task_id = submit(api, REQUEST)
        row = terminal(api, task_id)
        checks["truncation"] = (
            row["error"]["code"] == "ANALYSIS_RESULT_TRUNCATED"
            and row["report"]["query_result"]["truncated"]
            and "analysis" not in row["report"]
            and api.get(f"/tasks/{task_id}/artifacts/report.md").status_code == 404
        )
        write(output / "truncation.json", row)
    return checks


def recovery(base, snapshot, output):
    checks = {}
    downloads = None
    headers = {"Idempotency-Key": "v4-brief-recovery"}
    for key, phase, kind, number in (
        ("before_binding", "before_execute", "analysis", -1),
        ("after_binding", "after_commit", "analysis", -1),
        ("after_sql", "after_commit", "sql", 0),
        ("changed_identity", "after_commit", "analysis", -1),
    ):
        directory = output / key
        marker, options = probe_options(directory, phase, kind=kind, step_no=number)
        owned = []
        with application(
            base, snapshot, "product", "bm25", directory, phase="before", on_ready=owned.append,
            **options,
        ) as api:
            task_id = submit(api, REQUEST, headers=headers)
            assert wait_marker(api, marker, task_id)["pid"] == owned[0].pid
            idem = submit(api, REQUEST, headers=headers) == task_id
            conflict = api.post(
                "/tasks", json=body({**WINDOW, "city": "Beijing"}), headers=headers
            ).status_code == 409
            before = api.get(f"/tasks/{task_id}").raise_for_status().json()
            owned[0].kill()
            owned[0].wait(timeout=5)
        settings = (
            base.model_copy(update={"analysis_enabled": False}) if key == "changed_identity" else base
        )
        with application(settings, snapshot, "product", "bm25", directory, phase="after") as api:
            idem &= submit(api, REQUEST, headers=headers) == task_id
            after = terminal(api, task_id)
            trace = events(api, task_id)
            kinds = [e["type"] for e in trace]
            common = (
                idem and conflict and before["deadline_at"] == after["deadline_at"]
                and kinds.count("ANALYSIS_BINDING_COMMITTED") == 1
            )
            if key == "changed_identity":
                checks[key] = common and after["error"]["code"] == "CONFIG_CHANGED" and "SQL_STARTED" not in kinds
            else:
                checks[key] = (
                    common and after["status"] == "COMPLETED" and kinds.count("SQL_STARTED") == 1
                    and kinds.count("ANALYSIS_BINDING_REUSED") == int(key != "before_binding")
                    and sum(e["type"] == "STEP_REUSED" and e["payload"]["kind"] == "sql" for e in trace)
                    == int(key == "after_sql")
                )
            if key == "after_sql" and checks[key]:
                downloads = download(api, after, directory)
                newer = events(api, task_id)
                checks["download_after_restart"] = (
                    sum(e["type"] == "ARTIFACT_PREPARED" for e in newer) == 3
                    and sum(e["type"] == "SQL_STARTED" for e in newer) == 1
                )
        write(directory / "evidence.json", {"before": before, "after": after, "trace": trace})
    for key, phase in (("cancel", "after_commit"), ("timeout", "before_execute")):
        directory = output / key
        marker, options = probe_options(directory, phase, kind="analysis")
        settings = base.model_copy(update={"task_timeout_seconds": 2}) if key == "timeout" else base
        with application(settings, snapshot, "product", "bm25", directory, **options) as api:
            task_id = submit(api, REQUEST)
            wait_marker(api, marker, task_id)
            if key == "cancel":
                assert api.post(f"/tasks/{task_id}/cancel").raise_for_status().json()["changed"]
            row, trace = terminal(api, task_id), events(api, task_id)
            expected = (
                row["status"] == "CANCELLED" if key == "cancel" else
                row["status"] == "FAILED" and row["error"]["code"] == "TASK_TIMEOUT"
            )
            checks[key] = (
                expected and not any(e["type"] == "SQL_STARTED" for e in trace)
                and api.get(f"/tasks/{task_id}/artifacts/report.md").status_code == 404
            )
        write(directory / "evidence.json", {"task": row, "trace": trace})
    config = output / "after_sql/after-app.json"
    recovered = json.loads((config.parent / "evidence.json").read_bytes())["after"]["report"]
    cli = subprocess.run(
        [
            sys.executable, "-m", "agentserve", "--config", str(config), "--mode", "analysis",
            "--database", DATABASE, "--start", WINDOW["start"], "--end", WINDOW["end"],
            "--city", WINDOW["city"],
        ],
        text=True, encoding="utf-8", capture_output=True, check=True, timeout=60,
    )
    direct = json.loads(cli.stdout)
    checks["cli"] = (
        direct["status"] == "COMPLETED" and direct["model_calls"] == 0
        and direct["analysis"] == recovered["analysis"]
    )
    write(output / "cli.json", direct)
    return checks, downloads
