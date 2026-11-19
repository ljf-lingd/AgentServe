"""Run a Qwen protocol pilot and two full retail blocks on one GPU."""

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path

import httpx
import v5_engine
from filelock import FileLock
from retail_history import run_arm
from retail_replay import DATABASE, gold, workloads
from run_v5_phase import interrupt
from v5_phase_evidence import engine_timing
from v5_suffix_analysis import compare, summarize
from v6_phase_benchmark import MODEL, ROOT, SPEC, Settings, dump, engine_args, file_sha
from vllm_suffix_probe import probe_protocol

ARMS = ("ar", "suffix-local", "suffix-global", "session")
SMOKE_ORDER = ("suffix-local", "suffix-global", "ar", "session")
FEEDBACK = "suffix-feedback"


def load(path):
    return json.loads(path.read_bytes())


def arguments(arm):
    if arm in {"ar", "session"}:
        return engine_args("ar" if arm == "ar" else "all")
    suffix = {"method": "suffix", "num_speculative_tokens": SPEC["num_speculative_tokens"],
              "suffix_decoding_max_cached_requests": 0 if arm == "suffix-local" else 256}
    result = [*engine_args("ar"), "--speculative-config", json.dumps(suffix, separators=(",", ":")),
              "--per-request-spec-decode-metrics", "detailed"]
    if arm == FEEDBACK:
        result.extend(("--worker-cls", "suffix_feedback_worker.SuffixFeedbackWorker"))
    return result


async def measure(base, url, resources, cases, warmup, output, arm, pilot):
    config = base.model_copy(deep=True)
    config.base_url, config.memory.resources_url, config.model = url, resources, MODEL
    config.task_concurrency = config.llm_concurrency = 1
    before, after = (output / f"vllm-{phase}.prom" for phase in ("before", "after"))
    metrics_url = url.removesuffix("/v1") + "/metrics"
    async with httpx.AsyncClient(base_url=url + "/", trust_env=False, timeout=60) as client:
        before.write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
        sessions = await run_arm(config, "retail", arm, cases, warmup, output)
        after.write_text((await client.get(metrics_url)).raise_for_status().text, encoding="utf-8")
        metrics = engine_timing(before, after)
        protocols = await probe_protocol(client, MODEL) if pilot else []
    if pilot:
        dump(output / "protocol.json", protocols)
    result = {**summarize(sessions), "engine_including_one_warmup": metrics,
              "protocol": {"passed": sum(row["passed"] for row in protocols), "n": len(protocols)}}
    dump(output / "result.json", result)
    if pilot and (result["correct"] != result["n"] or not all(row["passed"] and row.get("tokens") for row in protocols)):
        raise RuntimeError("P1 protocol or retail pilot gate failed; see result and protocol JSON")
    return result


def execute(args, base, plan, *, arm_measure=measure):
    registration = args.output / "run.json"
    if registration.exists():
        if load(registration) != plan:
            raise ValueError("Existing run was initialized with a different frozen contract")
    else:
        dump(registration, plan)
    scopes = ["smoke", "matrix"] if args.stage == "both" else [args.stage]
    env = {"VLLM_PLUGINS": "", "PYTHONPATH": os.pathsep.join((str(args.arctic_site), str(ROOT / "serving/vllm")))}
    for scope in scopes:
        if scope == "matrix" and not (args.output / "smoke-complete.json").is_file():
            raise ValueError("The completed same-stack smoke is required")
        arms_order = tuple(plan["arguments"])
        smoke_order = SMOKE_ORDER + ((FEEDBACK,) if FEEDBACK in arms_order else ())
        blocks = (smoke_order,) if scope == "smoke" else (arms_order, arms_order[::-1])
        report = []
        for number, order in enumerate(blocks, 1):
            arms, sessions = {}, {}
            for arm in order:
                group = args.output / f"{scope}-b{number}-{arm}"
                group.mkdir(exist_ok=True)
                complete = group / "complete.json"
                if complete.exists():
                    directory = args.output / load(complete)["attempt"]
                    result = load(directory / "result.json")
                else:
                    directory = group / f"attempt-{len(list(group.glob('attempt-*'))) + 1:02d}"
                    directory.mkdir()
                    print(json.dumps({"stage": scope, "block": number, "arm": arm, "attempt": str(directory)}), flush=True)
                    with (
                        v5_engine.resources(directory, args.vllm_python, args.model_dir, args.embedding_dir) as resources,
                        v5_engine.engine(directory, arm, args.model_dir, args.vllm_python, args.gpu_uuid,
                                         server_args=arguments(arm), extra_env=env) as url,
                    ):
                        result = asyncio.run(arm_measure(base, url, resources, plan[scope], plan["warmup"],
                                                         directory, arm, scope == "smoke"))
                    dump(complete, {"attempt": directory.relative_to(args.output).as_posix()})
                arms[arm] = {**result, "attempt": directory.relative_to(args.output).as_posix()}
                sessions[arm] = load(directory / "retail.json")["sessions"]
                print(json.dumps({"stage": scope, "block": number, "arm": arm,
                                  **{k: result[k] for k in ("correct", "n", "jct", "model_requests", "accepted_tokens", "protocol")}}), flush=True)
            report.append({"block": number, "order": list(order), "arms": arms,
                           "pairs": {arm: compare(sessions["ar"], sessions[arm])
                                     for arm in arms_order if arm != "ar"},
                           **({"feedback_vs_suffix_global": compare(sessions["suffix-global"], sessions[FEEDBACK])}
                              if FEEDBACK in arms_order else {})})
        completed = args.output / f"{scope}-complete.json"
        if not completed.exists():
            dump(completed, {"run_sha256": file_sha(registration), "blocks": report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "matrix", "both"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--arctic-site", type=Path, required=True)
    parser.add_argument("--feedback", action="store_true", help="Include native Suffix with rejection feedback")
    args = parser.parse_args()
    config = ROOT / "configs/app.v7.retail.json"
    base = Settings.load(config)
    database = base.databases[DATABASE].path
    development = workloads("development")
    formal = gold(database, workloads("formal"))
    if sum(map(len, formal)) != 48:
        raise ValueError("The complete registered retail suite requires 48 turns")
    env = v5_engine.environment(args.vllm_python)
    if env["serving"]["vllm"] != "0.32.0+cu129":
        raise ValueError("P1 requires the validated vLLM version")
    extension, = args.arctic_site.glob("arctic_inference/suffix_decoding/_C.cpython-311-*.so")
    paths = [config, database, base.knowledge.snapshot_path, base.metric.registry_path,
             *sorted((ROOT / "src/agentserve").rglob("*.py")),
             *sorted((ROOT / "serving/vllm").glob("*.py"))]
    paths.extend(Path(__file__).with_name(name) for name in (
        "run_v5_suffix.py", "v5_engine.py", "retail_history.py", "retail_replay.py", "retail_scale.py",
        "v5_suffix_analysis.py", "vllm_suffix_probe.py", "retail_sam_pair.py", "model_protocol_probe.py",
        "v5_phase_evidence.py", "v6_phase_benchmark.py", "memory_resources.py", "http_server.py", "gpu_monitor.py",
    ))
    plan = {"environment": env, "gpu_uuid": args.gpu_uuid, "model_snapshot": v5_engine.verify_model(args.model_dir),
            "arctic_extension_sha256": file_sha(extension), "source_sha256": {p.resolve().relative_to(ROOT).as_posix(): file_sha(p) for p in paths},
            "arguments": {arm: arguments(arm) for arm in ARMS + ((FEEDBACK,) if args.feedback else ())},
            "smoke": gold(database, development[-2:]),
            "matrix": formal, "warmup": gold(database, [development[0]])[0][0]}
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    args.arctic_site, args.output = args.arctic_site.resolve(), args.output.resolve()
    args.output.mkdir(exist_ok=True)
    with FileLock(args.output / "run.lock", timeout=0):
        execute(args, base, load_json(plan))


def load_json(value):
    # Normalize the SQLite tuples before comparing a previous registration with this run.
    return json.loads(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
