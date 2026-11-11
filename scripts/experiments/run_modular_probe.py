"""Before/after structural regression: real GPU Qwen, fixed 16 demo + 20 smoke tasks."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from gpu_monitor import monitor
from model_server import temporary_model
from run_g5 import read, run_case, write

from agentserve.config import DatabaseConfig, Settings

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["before", "after"], required=True)
    args = parser.parse_args()
    output = (
        ROOT
        / "results/modularization"
        / (args.label + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings.load(ROOT / "configs/app.demo.json")
    for key, db in read(ROOT / "data/splits/spider_g5_v1/databases.json").items():
        settings.databases[key] = DatabaseConfig(
            path=ROOT / db["path"], schema_path=ROOT / db["schema_path"], context=db["context"]
        )
    hashes = {
        key: hashlib.sha256(db.path.read_bytes()).hexdigest()
        for key, db in settings.databases.items()
    }
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    report = {
        "label": args.label,
        "state": "prepared",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "cases": [],
    }
    write(output / "summary.json", report)
    try:
        with (
            monitor(output),
            temporary_model(output, max_model_len=8192, verify_gpu=True) as endpoint,
        ):
            settings.base_url = endpoint
            demo = settings.model_copy(deep=True)
            demo.databases = {"demo_orders": settings.databases["demo_orders"]}
            report["cases"].append(
                run_case(
                    demo,
                    output / "demo-regression",
                    read(ROOT / "data/processed/demo_orders/questions.json"),
                    {q["id"]: q for q in read(ROOT / "data/gold/demo_orders/answers.json")},
                    concurrency=2,
                    warmup=False,
                )
            )
            report["cases"].append(
                run_case(
                    settings,
                    output / "spider-smoke",
                    read(ROOT / "data/splits/spider_g5_v1/smoke.json"),
                    {q["id"]: q for q in read(ROOT / "data/gold/spider_g5_v1/smoke.json")},
                    concurrency=4,
                    warmup=False,
                    spider=True,
                )
            )
        report.update(
            state="completed",
            all_query_databases_unchanged=all(
                hashes[key] == hashlib.sha256(db.path.read_bytes()).hexdigest()
                for key, db in settings.databases.items()
            ),
        )
    finally:
        write(output / "summary.json", report)
    print(json.dumps({"output": str(output), "state": report["state"]}), flush=True)


if __name__ == "__main__":
    main()
