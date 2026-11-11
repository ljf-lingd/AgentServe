"""Configuration-driven SQL quality and workload runs on a verified real GPU model."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from gpu_monitor import monitor, snapshot
from http_server import http_server
from model_server import temporary_model

from agentserve.analysis.metrics import summarize
from agentserve.analysis.plots import workload_plots
from agentserve.analysis.report import read
from agentserve.config import DatabaseConfig, Settings
from agentserve.evaluation.batch import evaluate_batch
from agentserve.evaluation.sql import EVALUATION_PROTOCOL
from agentserve.workloads.experiment import ExperimentPlan
from agentserve.workloads.runner import run_closed_loop

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_case(
    base,
    output,
    jobs,
    gold,
    *,
    concurrency,
    wait=0,
    spider=False,
    warmup=True,
    warmup_jobs=None,
    llm_concurrency=None,
    sql_workers=None,
    condition_id=None,
    kind=None,
):
    output.mkdir()
    settings = base.model_copy(deep=True)
    settings.state_database = output / "tasks.sqlite"
    settings.task_concurrency = concurrency
    if llm_concurrency is not None:
        settings.llm_concurrency = llm_concurrency
    if sql_workers is not None:
        settings.sql_workers = sql_workers
    config = output / "app.json"
    config.write_text(settings.model_dump_json(indent=2))
    probe = output / "workload-probe.json"
    write(probe, {"sql_wait_seconds": wait})
    write(output / "workload.json", jobs)
    if warmup_jobs is None:
        warmup_jobs = jobs[:2] if warmup else []

    def progress(done, total):
        if done % 10 == 0 or done == total:
            print(
                json.dumps(
                    {
                        "state": "task_progress",
                        "case": output.name,
                        "finished": done,
                        "total": total,
                    }
                ),
                flush=True,
            )

    with http_server(
        "workload_app:create_workload_app",
        output,
        "api",
        env={"AGENTSERVE_CONFIG": str(config), "AGENTSERVE_WORKLOAD_PROBE": str(probe)},
        app_dir=ROOT / "scripts/experiments",
    ) as client:
        if warmup_jobs:
            batch = asyncio.run(
                run_closed_loop(
                    str(client.base_url), warmup_jobs, concurrency=1, output=output / "warmup"
                )
            )
            write(output / "warmup-summary.json", summarize(batch))
        write(output / "gpu-start.json", snapshot(output.name))
        batch = asyncio.run(
            run_closed_loop(
                str(client.base_url),
                jobs,
                concurrency=concurrency,
                output=output / "tasks",
                on_progress=progress,
            )
        )
        write(output / "gpu-end.json", snapshot(output.name + "_end"))
    labels = asyncio.run(
        evaluate_batch(batch, gold, {key: db.path for key, db in settings.databases.items()})
    )
    write(output / "evaluations.json", labels)
    result = {
        "case": output.name,
        "concurrency": concurrency,
        "llm_concurrency": settings.llm_concurrency,
        "sql_workers": settings.sql_workers,
        "sql_wait_seconds": wait,
        "spider": spider,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        **summarize(batch),
    }
    if condition_id is not None:
        result.update(condition_id=condition_id, kind=kind)
    write(output / "metrics.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "case",
                    "submitted",
                    "completed",
                    "quality",
                    "completed_tasks_per_second",
                    "correct_tasks_per_second",
                    "mean_jct_seconds",
                    "failures",
                ]
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g5.experiment.json")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate planned cases without starting services"
    )
    args = parser.parse_args()
    plan = ExperimentPlan.load(args.config)
    cases = plan.prepare(smoke_only=args.smoke_only)
    counts = {
        "measured_tasks": sum(len(c["jobs"]) for c in cases if c["condition"]["kind"] != "prewarm"),
        "warmup_tasks": sum(
            len(c["warmup_jobs"]) + (len(c["jobs"]) if c["condition"]["kind"] == "prewarm" else 0)
            for c in cases
        ),
    }
    if args.dry_run:
        print(json.dumps({**counts, "condition_order": [c["condition_id"] for c in cases]}))
        return
    if not plan.output_root.is_relative_to(ROOT):
        raise ValueError("Experiment outputs must stay inside the project")
    base = Settings.load(plan.application_config)
    for key, db in read(plan.database_manifest).items():
        base.databases[key] = DatabaseConfig(
            path=ROOT / db["path"], schema_path=ROOT / db["schema_path"], context=db["context"]
        )
    output = plan.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    paths = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted(Path(__file__).parent.glob("*.py"))
        + [
            ROOT / "pyproject.toml",
            plan.application_config,
            ROOT / "serving/vllm/launch-qwen.sh",
            args.config.resolve(),
        ]
    )
    report = {
        "run_id": output.name,
        "state": "prepared",
        "purpose": "real_GPU_Qwen_configured_SQL_workloads",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "effective_experiment": plan.model_dump(mode="json"),
        "planned_cases": cases,
        "input_sha256": plan.input_hashes(),
        "planned_counts": counts,
        "frozen_spider_manifest": read(plan.dataset_manifest),
        "quality_runs": [],
        "performance_runs": [],
        "tool_wait_runs": [],
        "prewarm_runs": [],
        "engine": {
            "max_model_len": plan.model_max_length,
            "max_num_seqs": 2,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.45,
            "enforce_eager": True,
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["fastapi", "httpx", "pydantic", "aiosqlite", "uvicorn"]
        },
        "model_manifest": read(ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json"),
        "cache_condition": "one_engine_shared_prefix_cache_no_resets_configured_prewarm_and_case_warmups",
    }
    write(output / "summary.json", report)
    hashes = {
        key: hashlib.sha256(db.path.read_bytes()).hexdigest() for key, db in base.databases.items()
    }
    try:
        with (
            monitor(output),
            temporary_model(
                output, max_model_len=plan.model_max_length, verify_gpu=True
            ) as endpoint,
        ):
            base.base_url = endpoint
            for case in cases:
                condition = case["condition"]
                dataset = plan.datasets[condition["dataset"]]
                settings = base.model_copy(deep=True)
                if not dataset.spider:
                    ids = {row["database_id"] for row in case["jobs"]}
                    settings.databases = {
                        key: db for key, db in base.databases.items() if key in ids
                    }
                result = run_case(
                    settings,
                    output / case["case"],
                    case["jobs"],
                    {q["id"]: q for q in read(dataset.gold)},
                    concurrency=condition["task_concurrency"],
                    wait=condition["sql_wait_seconds"],
                    llm_concurrency=condition["llm_concurrency"],
                    sql_workers=condition["sql_workers"],
                    spider=dataset.spider,
                    warmup_jobs=case["warmup_jobs"],
                    condition_id=case["condition_id"],
                    kind=condition["kind"],
                )
                key = {
                    "quality": "quality_runs",
                    "load": "performance_runs",
                    "tool_wait": "tool_wait_runs",
                    "prewarm": "prewarm_runs",
                }[condition["kind"]]
                report[key].append(result)
                write(output / "summary.json", report)
        report.update(
            state="completed",
            all_query_databases_unchanged=all(
                hashes[key] == hashlib.sha256(db.path.read_bytes()).hexdigest()
                for key, db in base.databases.items()
            ),
        )
        workload_plots(report, output, output)
    except Exception as error:
        report.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(output / "summary.json", report)
    print(json.dumps({"run_id": output.name, "state": report["state"], **counts}), flush=True)


if __name__ == "__main__":
    main()
