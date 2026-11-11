"""One-factor quality comparisons, followed by separately gated fresh acceptance."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from gpu_monitor import monitor
from model_server import temporary_model
from run_g5 import read, run_case, write

from agentserve.analysis.quality import select_profile
from agentserve.config import DatabaseConfig, Settings
from agentserve.evaluation.sql import EVALUATION_PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "baseline": {
        "answer_style": "baseline",
        "sql_function_profile": "baseline",
        "sql_prompt_version": "baseline",
    },
    "concise": {
        "answer_style": "concise_v1",
        "sql_function_profile": "baseline",
        "sql_prompt_version": "baseline",
    },
    "replace": {
        "answer_style": "concise_v1",
        "sql_function_profile": "with_replace_v1",
        "sql_prompt_version": "baseline",
    },
    "projection": {
        "answer_style": "concise_v1",
        "sql_function_profile": "with_replace_v1",
        "sql_prompt_version": "projection_v1",
    },
    "order_context": {
        "answer_style": "concise_v1",
        "sql_function_profile": "with_replace_v1",
        "sql_prompt_version": "baseline",
    },
    "bounded": {
        "answer_style": "bounded_v1",
        "sql_function_profile": "with_replace_v1",
        "sql_prompt_version": "baseline",
    },
}
DATASETS = {
    "demo": ("data/processed/demo_orders/questions.json", "data/gold/demo_orders/answers.json"),
    "smoke": ("data/splits/spider_g5_v1/smoke.json", "data/gold/spider_g5_v1/smoke.json"),
    "development": (
        "data/splits/spider_g5_v1/development.json",
        "data/gold/spider_g5_v1/development.json",
    ),
    "business": (
        "data/processed/business_orders_v1/questions.json",
        "data/gold/business_orders_v1/answers.json",
    ),
    "acceptance": (
        "data/splits/v1_quality_v1/acceptance.json",
        "data/gold/v1_quality_v1/acceptance.json",
    ),
}


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    paths.append(ROOT / "configs/app.demo.json")
    paths.append(ROOT / "configs/order-quality-context.txt")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def profile_settings(base, name):
    settings = base.model_copy(deep=True, update=PROFILES[name])
    if name in {"order_context", "bounded"}:
        note = (ROOT / "configs/order-quality-context.txt").read_text(encoding="utf-8")
        for key in ["demo_orders", "business_orders_v1"]:
            if key in settings.databases:
                settings.databases[key].context += "\n" + note
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "acceptance"], required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES))
    args = parser.parse_args()
    if args.phase == "acceptance" and args.selection is None:
        parser.error("acceptance requires a frozen --selection from development")
    selection = read(args.selection) if args.selection else None
    sources = source_hashes()
    if selection and (selection["source_sha256"] != sources or selection["profiles"] != PROFILES):
        raise ValueError(
            "Code/profiles changed after development selection; run a new development version"
        )
    if args.profiles and (
        args.phase != "development"
        or args.profiles[0] != "baseline"
        or len(set(args.profiles)) != len(args.profiles)
    ):
        parser.error(
            "--profiles is development-only and must start with baseline, without duplicates"
        )
    names = (
        (args.profiles or list(PROFILES))
        if not selection
        else list(dict.fromkeys(["baseline", selection["selected_profile"]]))
    )
    datasets = ["demo", "smoke", "development", "business"] if not selection else ["acceptance"]
    output = (
        ROOT
        / "results/v1-quality"
        / (args.phase + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    base = Settings.load(ROOT / "configs/app.demo.json")
    for key, db in read(ROOT / "data/splits/v1_quality_v1/databases.json").items():
        base.databases[key] = DatabaseConfig(
            path=ROOT / db["path"], schema_path=ROOT / db["schema_path"], context=db["context"]
        )
    hashes = {
        key: hashlib.sha256(db.path.read_bytes()).hexdigest() for key, db in base.databases.items()
    }
    report = {
        "state": "prepared",
        "phase": args.phase,
        "source_sha256": sources,
        "profiles": PROFILES,
        "evaluated_profiles": names,
        "dataset_manifest": read(ROOT / "data/splits/v1_quality_v1/manifest.json"),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "runs": [],
        "base_configuration": base.model_dump(mode="json"),
        "model_manifest": read(ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json"),
        "input_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for pair in DATASETS.values()
            for p in pair
        },
        "selection": selection,
        "cache_condition": "one_engine_shared_prefix_cache_no_resets_no_warmup",
        "engine": {
            "context": 8192,
            "max_num_seqs": 2,
            "dtype": "BF16",
            "TP": 1,
            "gpu_memory_utilization": 0.45,
            "eager": True,
        },
    }
    write(output / "summary.json", report)
    try:
        with (
            monitor(output),
            temporary_model(output, max_model_len=8192, verify_gpu=True) as endpoint,
        ):
            base.base_url = endpoint
            for name in names:
                for dataset in datasets:
                    settings = profile_settings(base, name)
                    jobs = read(ROOT / DATASETS[dataset][0])
                    ids = {q["database_id"] for q in jobs}
                    settings.databases = {
                        key: db for key, db in settings.databases.items() if key in ids
                    }
                    result = run_case(
                        settings,
                        output / (name + "-" + dataset),
                        jobs,
                        {q["id"]: q for q in read(ROOT / DATASETS[dataset][1])},
                        concurrency=2 if dataset == "demo" else 4,
                        warmup=False,
                        spider=dataset in {"smoke", "development", "acceptance"},
                    )
                    result.update(profile=name, dataset=dataset)
                    report["runs"].append(result)
                    write(output / "summary.json", report)
        report.update(
            state="completed",
            all_query_databases_unchanged=all(
                hashes[key] == hashlib.sha256(db.path.read_bytes()).hexdigest()
                for key, db in base.databases.items()
            ),
        )
        if args.phase == "development":
            chosen = select_profile(report["runs"], names)
            write(
                output / "selection.json",
                {
                    "development_run": output.name,
                    "selected_profile": chosen,
                    "profiles": PROFILES,
                    "evaluated_profiles": names,
                    "source_sha256": sources,
                    "rule": "no SQL correctness drop on demo/development/business; maximize completed_correct across four sets; simpler profile wins ties",
                },
            )
    except Exception as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write(output / "summary.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "state": report["state"],
                "selected_profile": chosen
                if args.phase == "development"
                else selection["selected_profile"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
