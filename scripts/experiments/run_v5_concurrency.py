"""Locate the expanded-load throughput crossover with app concurrency eight."""

import argparse
import asyncio
import json
import signal
from pathlib import Path

from filelock import FileLock

import v5_engine
import v5_expansion_live as live
import v6_phase_benchmark as contract
from agentserve.knowledge.snapshot import KnowledgeSnapshot
from run_v5_expansion import read, run_arm
from run_v5_phase import interrupt
from summarize_v5_expansion import compare


def execute(args, settings, parent, manifest, identity):
    database = settings.databases[contract.DATABASE].path
    initial_hash = contract.file_sha(database)
    snapshot = KnowledgeSnapshot(settings.knowledge.snapshot_path, "static", None)
    scopes = [("smoke", [("ar", "all")]), ("matrix", [("ar", "all"), ("all", "ar")])]
    registration = {"identity": identity, "application_concurrency": 8, "engine_max_seqs": 4,
                    "source_sha256": contract.file_sha(Path(__file__))}
    plan = args.output / "run.json"
    if plan.exists() and read(plan) != registration:
        raise ValueError("Existing C8 registration differs from this run")
    if not plan.exists():
        contract.dump(plan, registration)
    for stage, orders in scopes:
        workload = live.groups(manifest, smoke=stage == "smoke")
        blocks = []
        for block, order in enumerate(orders, 1):
            arms, full = {}, {}
            for arm in order:
                group = args.output / f"{stage}-b{block}-{arm}"
                group.mkdir(exist_ok=True)
                complete = group / "complete.json"
                if complete.exists():
                    directory = args.output / read(complete)["attempt"]
                    result = read(directory / "result.json")
                else:
                    directory = group / f"attempt-{len(list(group.glob('attempt-*'))) + 1:02d}"
                    directory.mkdir()
                    print(json.dumps({"stage": stage, "block": block, "arm": arm, "attempt": str(directory)}), flush=True)
                    with (
                        v5_engine.resources(directory, args.vllm_python, args.model_dir, args.embedding_dir) as resources,
                        v5_engine.engine(directory, arm, args.model_dir, args.vllm_python, args.gpu_uuid) as url,
                    ):
                        result = asyncio.run(run_arm(settings, parent, manifest, workload, snapshot,
                                                     {"concurrency": 8}, arm, directory, (url, resources)))
                    contract.dump(complete, {"attempt": directory.relative_to(args.output).as_posix()})
                arms[arm] = {**{k: v for k, v in result.items() if k != "cases"},
                             "issues": {k: v for k, v in result["cases"].items() if not v["correct"]},
                             "attempt": directory.relative_to(args.output).as_posix()}
                full[arm] = {**result, "trace": read(directory / "joined-trace.json")}
                print(json.dumps({"stage": stage, "block": block, "arm": arm, "n": result["timing"]["submitted"],
                                  "window_seconds": result["timing"]["window_seconds"], "jct": result["jct"]}), flush=True)
            blocks.append({"block": block, "order": list(order), "arms": arms, "ar_to_all": compare(full["ar"], full["all"])})
        if contract.file_sha(database) != initial_hash:
            raise RuntimeError("Query database changed during C8")
        complete_path = args.output / f"{stage}-complete.json"
        if not complete_path.exists():
            contract.dump(complete_path, {"registration_sha256": contract.file_sha(plan), "blocks": blocks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_matrix", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    identity = read(args.prior_matrix / "completed.json")["identity"]
    parent, manifest = read(args.parent), read(args.manifest)
    config = contract.ROOT / "configs/app.v5.json"
    settings = contract.Settings.load(config)
    if (identity["manifest_sha256"] != contract.file_sha(args.manifest) or
            identity["parent_sha256"] != contract.file_sha(args.parent) or parent != contract.plan(config, settings) or
            v5_engine.environment(args.vllm_python) != identity["environment"] or args.gpu_uuid != identity["gpu_uuid"] or
            v5_engine.verify_model(args.model_dir) != identity["model_snapshot"] or
            any(contract.file_sha(Path(__file__).with_name(name)) != sha for name, sha in identity["runtime_sha256"].items())):
        raise ValueError("C8 requires the same captured environment and experiment sources as the completed matrix")
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=True)
    with FileLock(args.output / "run.lock", timeout=0):
        execute(args, settings, parent, manifest, identity)


if __name__ == "__main__":
    main()
