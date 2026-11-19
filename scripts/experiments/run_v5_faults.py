"""Run v5 P0-2 against real Qwen and an isolated business task ledger."""

import argparse
import asyncio
import json
import signal
import sqlite3
import time
import traceback
from contextlib import closing
from pathlib import Path

import httpx

import v5_engine
import v5_fault_cases as cases
import v6_phase_benchmark as contract
from agentserve.tools.sqlite import SQLTool
from run_v5_phase import interrupt


async def permissions(database, healthy, attempts):
    run = healthy["revenue_rule"]
    assert not any(step["kind"] == "sql" for step in run["journal"]["steps"])
    ids = set(cases.model_ids(run))
    assert len(ids) == 1
    sequences = {row["sequence"] for row in attempts if row.get("request_id") in ids}
    wires = [row for row in attempts if row["type"] == "UPSTREAM_STARTED" and row["sequence"] in sequences]
    assert len(wires) == 1 and wires[0]["tool_names"] == [] and wires[0]["response_format"] == "json_schema"
    tool = SQLTool({contract.DATABASE: database})
    try:
        write = await tool.execute(contract.DATABASE, "UPDATE orders SET status=status WHERE 0",
                                   deadline=time.monotonic() + 2)
    finally:
        tool.close()
    assert write.status == "POLICY_REJECTED"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    return {"sql_write": write.status, "query_integrity": "ok", "knowledge_request_id": next(iter(ids)),
            "knowledge_tools": wires[0]["tool_names"], "knowledge_sql_steps": 0,
            "citations": run["task"]["report"]["knowledge"]["citations"]}


def run(args, base, frozen):
    base.model = contract.MODEL
    base.session_sam = base.session_sam_all_phases = False
    directory = args.output.resolve()
    directory.mkdir()
    database = base.databases[contract.DATABASE].path
    before = contract.file_sha(database)
    env = v5_engine.environment(args.vllm_python)
    if env["serving"]["vllm"] != frozen["vllm"]:
        raise ValueError("vLLM version differs from the frozen performance comparison")
    contract.dump(directory / "run.json", {
        "freeze_sha256": contract.file_sha(args.manifest), "environment": env,
        "gpu_uuid": args.gpu_uuid, "model_snapshot": v5_engine.verify_model(args.model_dir),
        "experiment_sha256": {name: contract.file_sha(contract.ROOT / "scripts/experiments" / name)
                              for name in ("run_v5_faults.py", "v5_fault_cases.py", "v5_engine.py",
                                           "fault_proxy.py", "recovery_app.py", "http_server.py")},
    })
    summary = {}
    with (
        v5_engine.resources(directory, args.vllm_python, args.model_dir, args.embedding_dir) as resource_url,
        v5_engine.engine(directory, "ar", args.model_dir, args.vllm_python, args.gpu_uuid) as model_url,
    ):
        healthy = cases.FaultCase(directory / "healthy", base, model_url, resource_url)
        baseline = cases.healthy(healthy, frozen["gold"])
        assert healthy.quick_check() == "ok"
        summary["healthy"] = {"passed": True, "correct": len(baseline), "tasks": len(baseline)}
        print(json.dumps({"fault_case": "healthy", **summary["healthy"]}), flush=True)
        for name, check in (("timeout", cases.timeout), ("restart", cases.restart), ("cancel", cases.cancel)):
            if args.case not in {"all", name}:
                continue
            case = cases.FaultCase(directory / name, base, model_url, resource_url)
            try:
                evidence = check(case) if name == "cancel" else check(case, baseline, frozen["gold"])
                assert case.quick_check() == "ok"
                summary[name] = {"passed": True, "state_integrity": "ok", **evidence}
            except (AssertionError, TimeoutError, httpx.HTTPError) as err:
                summary[name] = {"passed": False, "error": repr(err)}
                (case.output / "assertion.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(json.dumps({"fault_case": name, **summary[name]}), flush=True)
        if args.case in {"all", "permissions"}:
            try:
                evidence = asyncio.run(permissions(database, baseline, cases.read_lines(healthy.proxy_log)))
                summary["permissions"] = {"passed": True, **evidence}
            except AssertionError as err:
                summary["permissions"] = {"passed": False, "error": repr(err)}
    unchanged = before == contract.file_sha(database)
    contract.dump(directory / "result.json", {"cases": summary, "query_database_unchanged": unchanged,
                                             "query_database_sha256": before})
    if not unchanged or not all(row["passed"] for row in summary.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--case", choices=("all", "timeout", "restart", "cancel", "permissions"), default="all")
    args = parser.parse_args()
    config = contract.ROOT / "configs/app.v5.json"
    base = contract.Settings.load(config)
    frozen = json.loads(args.manifest.read_bytes())
    if contract.plan(config, base) != frozen:
        raise ValueError("The application, data and budgets must match the frozen manifest")
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    run(args, base, frozen)


if __name__ == "__main__":
    main()
