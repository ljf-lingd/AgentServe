"""Real GPU check of the deployable V1 config: 40 API tasks and one direct CLI task."""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentserve.config import Settings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from gpu_monitor import monitor
from model_server import temporary_model
from run_g5 import read, run_case, write


def main():
    output = (
        ROOT / "results/v1-quality" / ("product-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings.load(ROOT / "configs/app.v1.json")
    hashes = {
        key: hashlib.sha256(db.path.read_bytes()).hexdigest()
        for key, db in settings.databases.items()
    }
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT / "configs/app.v1.json"]
    report = {
        "state": "prepared",
        "purpose": "bounded_answer_business_regression_and_product_entrypoints",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    write(output / "summary.json", report)
    try:
        with (
            monitor(output),
            temporary_model(output, max_model_len=8192, verify_gpu=True) as endpoint,
        ):
            settings.base_url = endpoint
            report["business"] = run_case(
                settings,
                output / "business",
                read(ROOT / "data/processed/business_orders_v1/questions.json"),
                {q["id"]: q for q in read(ROOT / "data/gold/business_orders_v1/answers.json")},
                concurrency=4,
                warmup=False,
            )
            config = output / "cli-app.json"
            config.write_text(settings.model_dump_json(indent=2))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentserve",
                    "--config",
                    str(config),
                    "--database",
                    "business_orders_v1",
                    "--question",
                    "共有多少位客户？",
                    "--output",
                    str(output / "cli-report.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=ROOT,
            )
            (output / "cli.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError("Product CLI failed; see cli.log")
            cli = read(output / "cli-report.json")
            if cli["status"] != "COMPLETED" or cli["query_result"]["rows"] != [[120]]:
                raise RuntimeError("Product CLI returned an unexpected result")
            report["cli"] = {
                "status": cli["status"],
                "rows": cli["query_result"]["rows"],
                "model_calls": cli["model_calls"],
                "answer_source": cli["answer_source"],
            }
        report.update(
            state="completed",
            all_query_databases_unchanged=all(
                hashes[key] == hashlib.sha256(db.path.read_bytes()).hexdigest()
                for key, db in settings.databases.items()
            ),
        )
    except Exception as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write(output / "summary.json", report)
    print(json.dumps({"output": str(output), "state": report["state"]}), flush=True)


if __name__ == "__main__":
    main()
