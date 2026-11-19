"""Compare AR, Session-SAM and native Suffix on a new four-session UCI workload."""

import argparse
import json
import signal
from pathlib import Path

import httpx
import v5_engine
from filelock import FileLock
from retail_c4 import CONCURRENCY, cases, replay
from retail_replay import DATABASE, gold, workloads
from run_v5_phase import interrupt
from run_v5_suffix import ARMS, arguments, execute, load, load_json
from v5_phase_evidence import engine_timing
from v5_suffix_analysis import summarize
from v6_phase_benchmark import MODEL, ROOT, Settings, dump, file_sha


def workload(base):
    database = base.databases[DATABASE]
    development = workloads("development")
    matrix = cases()
    if len(matrix) != 24 or sum(map(len, matrix)) != 72:
        raise ValueError("C4 requires all 24 three-turn sessions")
    code = Path(__file__).resolve().parent
    audit_sources = (
        "run_v5_suffix_c4.py", "retail_c4.py", "run_v5_suffix.py", "retail_replay.py",
        "retail_history.py", "retail_scale.py", "v5_engine.py", "v5_suffix_analysis.py",
        "v5_phase_evidence.py", "v6_phase_benchmark.py", "gpu_monitor.py", "http_server.py",
    )
    sources = [code / name for name in audit_sources]
    sources.extend(sorted((ROOT / "src/agentserve").rglob("*.py")))
    sources.extend(sorted((ROOT / "serving/vllm").glob("*.py")))
    assets = {
        "config": ROOT / "configs/app.v7.retail.json", "database": database.path,
        "schema": database.schema_path, "knowledge": base.knowledge.snapshot_path,
        "metric": base.metric.registry_path,
    }
    return load_json({
        "concurrency": CONCURRENCY, "smoke": gold(database.path, development[:4]),
        "matrix": gold(database.path, matrix), "warmup": gold(database.path, [development[-1][:1]])[0][0],
        "assets": {name: file_sha(path) for name, path in assets.items()},
        "source_sha256": {path.relative_to(ROOT).as_posix(): file_sha(path) for path in sources},
        "arguments": {arm: arguments(arm) for arm in ARMS},
    })


async def measure_c4(base, url, resources, sequences, warmup, output, arm, pilot):
    configuration = base.model_copy(deep=True)
    configuration.base_url, configuration.memory.resources_url, configuration.model = url, resources, MODEL
    before, after = (output / f"vllm-{phase}.prom" for phase in ("before", "after"))
    metrics_url = url.removesuffix("/v1") + "/metrics"
    async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
        before.write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
        sessions, elapsed, http_requests = await replay(configuration, arm, sequences, warmup, output)
        after.write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
    result = summarize(sessions)
    result.update(
        elapsed=elapsed,
        throughput_tasks_per_second=result["n"] / elapsed,
        goodput_tasks_per_second=result["correct"] / elapsed,
        http_requests=http_requests,
        engine_including_one_warmup=engine_timing(before, after),
        protocol={"passed": 0, "n": 0},
    )
    dump(output / "result.json", result)
    if pilot and result["n"] != result["correct"]:
        raise RuntimeError("The known retail smoke queries must match independent gold")
    return result


def check_smoke(output):
    smoke = load(output / "smoke-complete.json")
    if smoke["run_sha256"] != file_sha(output / "run.json"):
        raise ValueError("C4 formal execution requires the registered same-stack smoke")
    pairs = smoke["blocks"][0]["pairs"]
    for arm, pair in pairs.items():
        if pair["changed_sql_result_or_status"] or pair["strict_jct_delta"]["n"] != CONCURRENCY:
            raise ValueError(f"C4 smoke did not match AR token-for-token: {arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "smoke", "matrix"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--vllm-python")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--arctic-site", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    base = Settings.load(ROOT / "configs/app.v7.retail.json")
    registered = workload(base)
    if args.stage == "freeze":
        args.output.mkdir()
        dump(args.output / "workload.json", registered)
        print(json.dumps({"smoke": sum(map(len, registered["smoke"])), "matrix": 72,
                          "workload_sha256": file_sha(args.output / "workload.json")}))
        return
    if not all((args.model_dir, args.embedding_dir, args.vllm_python, args.gpu_uuid, args.arctic_site)):
        parser.error("GPU stages require model, embedding, GPU, Python and Arctic paths")
    if registered != load(args.output / "workload.json"):
        raise ValueError("C4 inputs differ from the frozen workload")
    args.arctic_site = args.arctic_site.resolve(strict=True)
    environment = v5_engine.environment(args.vllm_python)
    if environment["serving"]["vllm"] != "0.32.0+cu129":
        raise ValueError("C4 requires the validated vLLM version")
    extension, = args.arctic_site.glob("arctic_inference/suffix_decoding/_C.cpython-311-*.so")
    plan = {
        **registered, "environment": environment, "gpu_uuid": args.gpu_uuid,
        "model_snapshot": v5_engine.verify_model(args.model_dir),
        "arctic_extension_sha256": file_sha(extension),
    }
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    with FileLock(args.output / "run.lock", timeout=0):
        if args.stage == "matrix":
            check_smoke(args.output)
        execute(args, base, plan, arm_measure=measure_c4)
        if args.stage == "smoke":
            check_smoke(args.output)


if __name__ == "__main__":
    main()
