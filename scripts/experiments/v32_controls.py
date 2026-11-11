"""V3.2 zero-SQL boundaries and real API/CLI recovery probes."""

import asyncio
import json
import shutil
import subprocess
import sys

from run_v21 import write
from run_v31 import application, read, sha
from v31_cases import DATABASE, run_requests
from v31_recovery import events, probe_options, submit, terminal, wait_marker
from v32_cases import PRODUCT, body, evaluate, probe_cases, test_registry

from agentserve.application.bootstrap import open_task_service
from agentserve.config import MetricSettings
from agentserve.domain.models import MetricInput
from agentserve.knowledge.metrics import MetricCatalog, MetricRule, citation_for, compile_metric_sql
from agentserve.knowledge.snapshot import assemble, digest

REQUEST = {"name": "成交营收", "start": "2026-01-01", "end": "2026-04-01"}
PROBE = body(REQUEST)


def settings_for(base, registry, purpose="product"):
    settings = base.model_copy(deep=True)
    settings.metric = MetricSettings(registry_path=registry, purpose=purpose)
    return settings


async def rejected_on_start(base, registry, purpose, message):
    settings = settings_for(base, registry, purpose)
    settings.memory = None
    settings.knowledge = None
    try:
        async with open_task_service(settings):
            return False
    except ValueError as error:
        return message in str(error)


def counterfactual(base, snapshot, output):
    results, checks = {}, {}
    for variant in ("versions", "object-a", "object-b"):
        directory = output / variant
        registry = test_registry(directory / "sources", variant)
        settings = settings_for(base, registry, "test")
        with application(settings, snapshot, "product", "bm25", directory) as api:
            results[variant] = evaluate(api, probe_cases(variant), settings, registry, directory)
            if variant == "versions":
                checks.update(boundaries(api, directory))
    checks["old_and_new_results"] = all(r["passed"] for r in results["versions"])
    for old, new in zip(results["object-a"], results["object-b"], strict=True):
        checks[f"intervention_{old['id']}"] = (
            old["passed"] and new["passed"] and old["rows"] != new["rows"]
            and old["rule_sha256"] != new["rule_sha256"]
            and old["citation"] != new["citation"] and old["parameters"] != new["parameters"]
        )
    a = output / "object-a/sources/registry.json"
    checks["test_registry_rejected_by_product"] = asyncio.run(
        rejected_on_start(base, a, "product", "purpose")
    )
    registry = shutil.copytree(a.parent, output / "text-only") / a.name
    source = registry.with_name("settled-revenue-A.md")
    manifest = read(registry.with_name("manifest.json"))
    catalog = MetricCatalog(registry, purpose="test", database_ids={DATABASE})
    request = MetricInput.model_validate(REQUEST)
    bound = MetricRule.model_construct(**catalog.bind(DATABASE, request)["rule"])
    original_sql = compile_metric_sql(bound, request)
    source.write_bytes(source.read_bytes().replace(b"completed", b"cancelled"))
    manifest["documents"][0]["source_sha256"] = sha(source)
    write(registry.with_name("manifest.json"), manifest)
    checks["document_only_compilation"] = compile_metric_sql(bound, request) == original_sql
    checks["document_only_rejected_by_startup"] = asyncio.run(
        rejected_on_start(base, registry, "test", "source binding changed")
    )
    ambiguous = shutil.copytree(a.parent, output / "ambiguous") / a.name
    value = read(ambiguous)
    conflicting_rule = value["rules"][1]
    conflicting_rule["aliases"].append("成交营收")
    conflicting_source = next(
        c for c in assemble(ambiguous.with_name("manifest.json"))["chunks"]
        if c["chunk_id"] == conflicting_rule["source"]
    )
    conflicting_rule["source_binding_sha256"] = digest({
        "rule": {k: v for k, v in conflicting_rule.items() if k != "source_binding_sha256"},
        "citation": citation_for(conflicting_source),
    })
    write(ambiguous, value)
    checks["ambiguous_alias_rejected_by_startup"] = asyncio.run(
        rejected_on_start(base, ambiguous, "test", "Ambiguous metric alias")
    )
    draft = output / "versions/sources/registry.json"
    unapproved = draft.with_name("unapproved.json")
    value = read(draft)
    value["rules"][0]["source"] = "settled-revenue@draft#1"
    write(unapproved, value)
    checks["draft_rule_rejected_by_startup"] = asyncio.run(
        rejected_on_start(base, unapproved, "test", "ineligible metric source")
    )
    write(output / "evidence.json", {"checks": checks, "results": results})
    return checks


def boundaries(api, output):
    invalid = {
        "reversed_window": {**REQUEST, "start": "2026-04-02"},
        "invalid_date": {**REQUEST, "start": "2026-02-30"},
        "disallowed_group": {**REQUEST, "group_by": "status"},
        "free_sql_parameter": {**REQUEST, "sql": "DELETE FROM orders"},
    }
    checks = {
        key: api.post("/tasks", json=body(request)).status_code == 422
        for key, request in invalid.items()
    }
    checks["metric_session_rejected"] = (
        api.post("/tasks", json={**PROBE, "session_id": "not-a-metric-session"}).status_code == 422
    )
    cases = {}
    for name, prefix in (("成交营收", "revenue"), ("完成订单量", "orders")):
        for boundary, start, end in (
            ("crosses_version", "2026-03-31", "2026-04-02"),
            ("gap", "2026-07-01", "2026-07-02"),
            ("only_draft", "2026-08-01", "2026-08-02"),
            ("before_first", "2025-12-31", "2026-01-02"),
        ):
            cases[f"{prefix}_{boundary}"] = (
                body({"name": name, "start": start, "end": end}), "METRIC_WINDOW_UNCOVERED"
            )
    cases["unapproved_alias"] = (body({**REQUEST, "name": "成交营收; DROP TABLE orders"}), "METRIC_UNKNOWN")
    cases["wrong_database"] = ({**PROBE, "database_id": "demo_orders"}, "METRIC_UNKNOWN")
    rows = asyncio.run(run_requests(str(api.base_url), [(k, v[0]) for k, v in cases.items()], 4))
    for key, (_, code) in cases.items():
        row = rows[key]
        checks[key] = (
            row["status"] == "FAILED" and row["error"]["code"] == code
            and row["report"]["model_calls"] == 0 and row["report"]["sql"] is None
            and not any(e["type"].startswith(("SQL_", "MODEL_", "KNOWLEDGE_")) for e in row["trace"])
        )
    write(output / "boundaries.json", {"checks": checks, "tasks": rows})
    return checks


def recovery(base, snapshot, output):
    checks = {}
    headers = {"Idempotency-Key": "metric-recovery"}
    for name, stage, kind, step_no in (
        ("before_binding", "before_execute", "metric", -1),
        ("after_binding", "after_commit", "metric", -1),
        ("after_sql_commit", "after_commit", "sql", 0),
        ("changed_identity", "after_commit", "metric", -1),
    ):
        directory = output / name
        changed = name == "changed_identity"
        if changed:
            with application(base, snapshot, "product", "bm25", directory, phase="history") as api:
                history_id = submit(api, PROBE)
                historical = terminal(api, history_id)
                assert historical["status"] == "COMPLETED"
        marker, options = probe_options(directory, stage, kind=kind, step_no=step_no)
        owned = []
        with application(
            base, snapshot, "product", "bm25", directory, phase="before",
            on_ready=owned.append, **options,
        ) as api:
            task_id = submit(api, PROBE, headers=headers)
            assert wait_marker(api, marker, task_id)["pid"] == owned[0].pid
            same = submit(api, PROBE, headers=headers) == task_id
            different = api.post(
                "/tasks", json=body({**REQUEST, "end": "2026-03-01"}), headers=headers
            ).status_code == 409
            before = api.get(f"/tasks/{task_id}").raise_for_status().json()
            owned[0].kill()
            owned[0].wait(timeout=5)
        settings = base
        if changed:
            settings = settings_for(base, test_registry(directory / "changed", "object-b"), "test")
        with application(settings, snapshot, "product", "bm25", directory, phase="after") as api:
            same &= submit(api, PROBE, headers=headers) == task_id
            after = terminal(api, task_id)
            trace = events(api, task_id)
            kinds = [event["type"] for event in trace]
            common = same and different and before["deadline_at"] == after["deadline_at"]
            common &= kinds.count("METRIC_BINDING_COMMITTED") == 1
            if changed:
                checks[name] = (
                    common and after["status"] == "FAILED"
                    and after["error"]["code"] == "CONFIG_CHANGED" and "SQL_STARTED" not in kinds
                    and api.get(f"/tasks/{history_id}").json()["report"] == historical["report"]
                )
            else:
                checks[name] = (
                    common and after["status"] == "COMPLETED"
                    and after["report"]["model_calls"] == 0 and kinds.count("SQL_STARTED") == 1
                    and kinds.count("METRIC_BINDING_REUSED") == int(name != "before_binding")
                    and sum(e["type"] == "STEP_REUSED" and e["payload"]["kind"] == "sql" for e in trace)
                    == int(name == "after_sql_commit")
                )
        write(directory / "evidence.json", {"before": before, "after": after, "trace": trace})
    for name, stage in (("cancel", "after_commit"), ("timeout", "before_execute")):
        directory = output / name
        marker, options = probe_options(directory, stage, kind="metric")
        settings = base.model_copy(deep=True)
        if name == "timeout":
            settings.task_timeout_seconds = 2
        with application(settings, snapshot, "product", "bm25", directory, **options) as api:
            task_id = submit(api, PROBE)
            wait_marker(api, marker, task_id)
            if name == "cancel":
                assert api.post(f"/tasks/{task_id}/cancel").raise_for_status().json()["changed"]
            row = terminal(api, task_id)
            trace = events(api, task_id)
            expected = (
                row["status"] == "CANCELLED"
                and not api.post(f"/tasks/{task_id}/cancel").json()["changed"]
                if name == "cancel" else
                row["status"] == "FAILED" and row["error"]["code"] == "TASK_TIMEOUT"
            )
            checks[name] = expected and not any(e["type"].startswith(("SQL_", "MODEL_")) for e in trace)
        write(directory / "evidence.json", {"task": row, "trace": trace})

    config = output / "after_binding/after-app.json"
    cli = subprocess.run(
        [
            sys.executable, "-m", "agentserve", "--config", str(config), "--mode", "metric",
            "--database", DATABASE, "--metric", REQUEST["name"], "--start", REQUEST["start"],
            "--end", REQUEST["end"],
        ],
        text=True, encoding="utf-8", capture_output=True, check=True, timeout=60,
    )
    report = json.loads(cli.stdout)
    recovered = read(output / "after_binding/evidence.json")["after"]["report"]
    checks["cli"] = (
        report["status"] == "COMPLETED" and report["model_calls"] == 0
        and report["metric"] == recovered["metric"]
        and report["query_result"] == recovered["query_result"]
    )
    write(output / "cli.json", report)
    return checks
