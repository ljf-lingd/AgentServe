"""Smoke-test and run the complete three-arm v5 expanded benchmark."""

import argparse
import asyncio
import json
import signal
from pathlib import Path

import httpx
from filelock import FileLock

import v5_engine
import v5_expansion_live as live
import v6_phase_benchmark as contract
from agentserve.knowledge.snapshot import KnowledgeSnapshot
from run_v5_phase import interrupt
from v5_expansion_eval import summarize
from v5_phase_evidence import engine_timing


def read(path):
    return json.loads(path.read_bytes())


async def run_arm(base, frozen, manifest, workload, snapshot, block, arm, directory, urls):
    settings = base.model_copy(deep=True)
    settings.base_url, settings.memory.resources_url = urls
    settings.model = contract.MODEL
    settings.task_concurrency = settings.llm_concurrency = block["concurrency"]
    for key, enabled in frozen["arms"][arm].items():
        setattr(settings, key, enabled)
    settings.state_database = directory / "warmup.sqlite"
    warm, summary = await contract.run_mix(settings, block["concurrency"], frozen["gold"])
    contract.dump(directory / "warmup.json", {"summary": summary, "cases": warm})
    if any(row["task"]["status"] != "COMPLETED" for row in warm.values()):
        raise RuntimeError("A warmup task failed; see warmup.json")
    settings.state_database = directory / "tasks.sqlite"
    metrics_url = urls[0].removesuffix("/v1") + "/metrics"
    before, after = (directory / f"vllm-{which}.prom" for which in ("before", "after"))
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        before.write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
        rows, elapsed, calls = await live.run(settings, workload, manifest["business_time"], directory)
        after.write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
    result, trace = summarize(rows, live.index_cases(workload), snapshot, calls, elapsed, arm)
    contract.dump(directory / "joined-trace.json", trace)
    result["engine"] = engine_timing(before, after)
    contract.dump(directory / "result.json", result)
    coverage = result["trace"]
    if coverage["linked_steps"] != coverage["committed_model_steps"] or coverage["sam_mismatches"]:
        raise RuntimeError("Committed model request linkage or SAM routing failed; see joined-trace.json")
    if coverage["linked_steps"] == 0 or not any(row["model_steps"] == 0 for row in result["cases"].values()):
        raise RuntimeError("Expected both a real model request and a direct task")
    if result["engine"].get("e2e_request_latency_seconds", {}).get("n", 0) == 0:
        raise RuntimeError("Missing real vLLM latency samples")
    return result


def execute(args, base, frozen, manifest, identity):
    is_smoke = args.stage == "smoke"
    workload = live.groups(manifest, smoke=is_smoke)
    case_count = len(live.index_cases(workload))
    blocks = ([{"concurrency": 4, "order": ["ar", "answer", "all"]}]
              if is_smoke else frozen["blocks"])
    if not is_smoke:
        previous = read(args.smoke / "completed.json")
        if previous["stage"] != "smoke" or previous["identity"] != identity or case_count != manifest["total"]:
            raise ValueError("Matrix requires the completed smoke with the same benchmark identity")
    plan = {"stage": args.stage, "identity": identity, "blocks": blocks, "cases_per_arm": case_count,
            "groups": [[kind, group, [case["id"] for case in items]] for kind, group, items in workload],
            "warmup": "Independent original ten-task mix per arm, then a fresh task database"}
    snapshot = KnowledgeSnapshot(base.knowledge.snapshot_path, "static", None)
    plan_path = args.output / "run.json"
    if plan_path.exists():
        if read(plan_path) != plan:
            raise ValueError("Output was initialized with a different benchmark identity or workload")
    else:
        contract.dump(plan_path, plan)
    receipts = []
    for index, block in enumerate(blocks, 1):
        for arm in block["order"]:
            parent = args.output / f"b{index}-{arm}"
            parent.mkdir(exist_ok=True)
            path = parent / "complete.json"
            if path.exists():
                receipts.append(read(path))
                continue
            attempt = parent / f"attempt-{len(list(parent.glob('attempt-*'))) + 1:02d}"
            attempt.mkdir()
            print(json.dumps({"stage": args.stage, "block": index, "arm": arm,
                              "planned": case_count, "attempt": str(attempt)}), flush=True)
            with (
                v5_engine.resources(attempt, args.vllm_python, args.model_dir, args.embedding_dir) as resources,
                v5_engine.engine(attempt, arm, args.model_dir, args.vllm_python, args.gpu_uuid) as url,
            ):
                result = asyncio.run(run_arm(base, frozen, manifest, workload, snapshot, block,
                                             arm, attempt, (url, resources)))
            receipt = {"block": index, "arm": arm, "concurrency": block["concurrency"],
                       "attempt": attempt.relative_to(args.output).as_posix(),
                       "correct": result["timing"]["quality"].get("correct", 0),
                       "submitted": result["timing"]["submitted"], "elapsed": result["timing"]["window_seconds"],
                       "jct": result["jct"], "trace": result["trace"]}
            contract.dump(path, receipt)
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    if not (args.output / "completed.json").exists():
        contract.dump(args.output / "completed.json", {"stage": args.stage, "identity": identity, "arms": receipts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "matrix"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if args.stage == "matrix" and args.smoke is None:
        parser.error("matrix requires --smoke pointing to a completed smoke directory")
    config = contract.ROOT / "configs/app.v5.json"
    base = contract.Settings.load(config)
    frozen, manifest = read(args.parent), read(args.manifest)
    if contract.plan(config, base) != frozen or manifest["parent_freeze_sha256"] != contract.file_sha(args.parent):
        raise ValueError("Parent application, data and launch contract must match the frozen manifests")
    if contract.file_sha(args.manifest) != args.manifest_sha:
        raise ValueError("Expanded workload hash differs from its registered SHA-256")
    environment = v5_engine.environment(args.vllm_python)
    if environment["serving"]["vllm"] != frozen["vllm"]:
        raise ValueError("vLLM version differs from the frozen manifest")
    identity = {"parent_sha256": manifest["parent_freeze_sha256"], "manifest_sha256": args.manifest_sha,
                "environment": environment, "gpu_uuid": args.gpu_uuid,
                "model_snapshot": v5_engine.verify_model(args.model_dir),
                "runtime_sha256": {name: contract.file_sha(Path(__file__).with_name(name)) for name in (
                    "run_v5_expansion.py", "v5_expansion_live.py", "v5_expansion_eval.py", "v31_cases.py",
                    "v5_engine.py", "v5_phase_evidence.py", "http_server.py", "gpu_monitor.py", "memory_resources.py",
                )}}
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=True)
    with FileLock(args.output / "run.lock", timeout=0):
        execute(args, base, frozen, manifest, identity)


if __name__ == "__main__":
    main()
