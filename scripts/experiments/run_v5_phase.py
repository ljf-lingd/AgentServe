"""Run the v5 pilot or the frozen P0-1 matrix on one owned GPU."""

import argparse
import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v5_engine
import v6_phase_benchmark as workload
import httpx

from v5_phase_evidence import ModelTrace, audit, engine_timing, paired_paths, task_timing

PILOT_GROUPS = [["chengdu"], ["direct"], ["revenue_rule"]]
PAIRS = (("ar", "answer"), ("ar", "all"), ("answer", "all"))


def audit_pilot(result):
    cases = result["cases"]
    trace = result["trace"]
    steps = [step for task in trace["cases"].values() for step in task]
    if not all(row["correct"] for row in cases.values()) or trace["unlinked_attempts"] or (
        trace["linked_steps"] != trace["committed_model_steps"] or trace["sam_mismatches"]
    ):
        raise ValueError("Pilot quality or model request linkage failed")
    if trace["cases"]["direct"] or not any(
        step["phase"] == "sql" and step["call"]["finish_reason"] == "tool_calls" for step in steps
    ) or not any(step["phase"] == "answer" for step in steps):
        raise ValueError("Pilot did not cover direct, tool, and answer paths")
    if any(not step["call"]["usage"] or not step["call"]["metrics"] for step in steps):
        raise ValueError("Pilot is missing vLLM usage or per-request timing")


async def run_arm(args, base, frozen, arm, block, directory, engine_dir, model_url, resources):
    traces = []
    original_create_app = workload.create_app

    def traced_app(*app_args, **kwargs):
        transport = ModelTrace()
        traces.append(transport.calls)
        return original_create_app(*app_args, model_transport=transport, **kwargs)

    settings = base.model_copy(deep=True)
    run = SimpleNamespace(arm=arm, block=block, model_url=model_url, resources_url=resources,
                          output=directory, manifest=args.manifest)
    with patch.object(workload, "create_app", traced_app):
        if args.stage == "pilot":
            settings.base_url, settings.memory.resources_url = model_url, resources
            settings.model = workload.MODEL
            settings.task_concurrency = settings.llm_concurrency = 1
            settings.state_database = engine_dir / "pilot.sqlite"
            for key, value in frozen["arms"][arm].items():
                setattr(settings, key, value)
            with patch.object(workload, "GROUPS", PILOT_GROUPS):
                rows, summary = await workload.run_mix(settings, 1, frozen["gold"])
            result = {"arm": arm, "summary": summary, "cases": rows, "trace": audit(rows, traces[0], arm)}
            workload.dump(engine_dir / "pilot.json", result)
            async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                metrics = (await client.get(model_url.removesuffix("/v1") + "/metrics")).raise_for_status()
                (engine_dir / "vllm-pilot.prom").write_text(metrics.text, encoding="utf-8")
            audit_pilot(result)
            print(json.dumps({"pilot": arm, "correct": summary["correct"],
                              "model_requests": result["trace"]["linked_steps"]}), flush=True)
            return
        try:
            await workload.live(run, settings, frozen)
        finally:
            workload.dump(engine_dir / "model-requests.json", traces)
    result = json.loads((directory / "result.json").read_bytes())
    rows = result["cases"]
    result["trace"] = audit(rows, traces[1], arm)
    result["stage_timing"] = task_timing(rows, len(rows) / result["summary"]["tasks_per_second"])
    result["engine_timing"] = engine_timing(directory / "vllm-before.prom", directory / "vllm-after.prom")
    workload.dump(directory / "evidence.json", result)


def compare(output, manifest):
    workload.compare(output, manifest)
    compact = []
    for index, block in enumerate(workload.BLOCKS, 1):
        results = {arm: json.loads((output / f"b{index}-{arm}/evidence.json").read_bytes())
                   for arm in block["order"]}
        compact.append({"block": index, **block, "arms": {
            arm: {"summary": result["summary"], "stage_timing": result["stage_timing"],
                  "engine_timing": result["engine_timing"],
                  "trace": {k: v for k, v in result["trace"].items() if k != "cases"}}
            for arm, result in results.items()
        }, "pairs": {f"{a}_to_{b}": paired_paths(results[a], results[b]) for a, b in PAIRS}})
    workload.dump(output / "paired-paths.json", compact)


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "matrix"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    config = workload.ROOT / "configs/app.v5.json"
    base = workload.Settings.load(config)
    frozen = json.loads(args.manifest.read_bytes())
    if workload.plan(config, base) != frozen:
        raise ValueError("Source, data and launch contract must match the frozen manifest")
    environment = v5_engine.environment(args.vllm_python)
    if environment["serving"]["vllm"] != frozen["vllm"]:
        raise ValueError("vLLM version differs from the frozen manifest")
    args.output = args.output.resolve()
    args.output.mkdir()
    workload.dump(args.output / "run.json", {"stage": args.stage, "freeze_sha256": workload.file_sha(args.manifest),
        "environment": environment, "gpu_uuid": args.gpu_uuid,
        "experiment_sha256": {path: workload.file_sha(workload.ROOT / path) for path in (
            "scripts/experiments/run_v5_phase.py", "scripts/experiments/v5_engine.py",
            "scripts/experiments/v5_phase_evidence.py", "scripts/experiments/http_server.py",
            "scripts/experiments/gpu_monitor.py", "scripts/experiments/memory_resources.py",
        )}, "model_snapshot": v5_engine.verify_model(args.model_dir)})
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    blocks = frozen["blocks"] if args.stage == "matrix" else [frozen["blocks"][0]]
    for block, spec in enumerate(blocks, 1):
        for arm in spec["order"]:
            directory = args.output / f"b{block}-{arm}"
            receipts = directory.with_name(directory.name + "-engine")
            receipts.mkdir()
            print(json.dumps({"stage": args.stage, "block": block, "starting": arm}), flush=True)
            with (
                v5_engine.resources(receipts, args.vllm_python, args.model_dir, args.embedding_dir) as resources,
                v5_engine.engine(receipts, arm, args.model_dir, args.vllm_python, args.gpu_uuid) as model_url,
            ):
                asyncio.run(run_arm(args, base, frozen, arm, block, directory, receipts, model_url, resources))
    if args.stage == "matrix":
        compare(args.output, args.manifest)


if __name__ == "__main__":
    main()
