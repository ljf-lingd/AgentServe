"""Run real functional probes; gold stays outside the Agent. Not a performance benchmark."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import httpx
from model_server import temporary_model

from agentserve.agents.sql import SQLAgent
from agentserve.application.service import TaskService
from agentserve.config import Settings
from agentserve.domain.models import TaskInput
from agentserve.llm.client import VLLMClient
from agentserve.tools.sqlite import SQLTool

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def probes(settings: Settings, output: Path) -> dict:
    questions_path = ROOT / "data/processed/demo_orders/questions.json"
    gold_path = ROOT / "data/gold/demo_orders/answers.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    gold = {item["id"]: item for item in json.loads(gold_path.read_text(encoding="utf-8"))}
    database = settings.databases["demo_orders"].path
    source = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__),
        Path(__file__).with_name("model_server.py"),
        ROOT / "configs/app.demo.json",
    ]
    before = sha256(database)
    report = {
        "purpose": "functional_regression_not_performance_benchmark",
        "run_id": output.name,
        "started_at": datetime.now(UTC).isoformat(),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source},
        "dataset_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                questions_path,
                gold_path,
                database,
                settings.databases["demo_orders"].schema_path,
            ]
        },
        "config": settings.model_dump(mode="json"),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ["httpx", "pydantic", "pytest"]
        },
        "model_manifest": json.loads(
            (ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json").read_text()
        ),
        "records": [],
        "final_answers_automatically_graded": False,
    }
    tool = SQLTool(
        {key: db.path for key, db in settings.databases.items()},
        workers=settings.sql_workers,
        timeout_seconds=settings.sql_timeout_seconds,
        row_limit=settings.row_limit,
        byte_limit=settings.result_byte_limit,
    )
    try:
        async with httpx.AsyncClient(base_url=settings.base_url + "/", trust_env=False) as client:
            catalog = await client.get("models", timeout=10)
            catalog.raise_for_status()
            report["model_catalog"] = catalog.json()
            service = TaskService(SQLAgent(VLLMClient(client, settings.model), tool, settings))
            for question in questions:
                result = await service.run(
                    TaskInput(database_id=question["database_id"], question=question["question"])
                )
                with (output / (question["id"] + ".json")).open("x", encoding="utf-8") as handle:
                    json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
                expected, actual = gold[question["id"]], result.query_result
                if actual is None or actual.status != "ok":
                    matched = False
                elif actual.truncated:
                    matched = None
                elif expected["ordered"]:
                    matched = actual.rows == expected["rows"]
                else:
                    matched = Counter(map(json.dumps, actual.rows)) == Counter(
                        map(json.dumps, expected["rows"])
                    )
                record = {
                    "id": question["id"],
                    "status": result.status,
                    "strict_match": matched,
                    "sql_repairs": result.sql_repairs,
                    "format_repairs": result.format_repairs,
                    "model_calls": result.model_calls,
                    "elapsed_seconds": result.elapsed_seconds,
                    "error": result.error,
                }
                report["records"].append(record)
                (output / "summary.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)
                )
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        tool.close()
    report.update(
        total=len(questions),
        completed=sum(r["status"] == "COMPLETED" for r in report["records"]),
        strict_matches=sum(r["strict_match"] is True for r in report["records"]),
        database_unchanged=before == sha256(database),
        ended_at=datetime.now(UTC).isoformat(),
    )
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/app.demo.json")
    parser.add_argument("--start-model", action="store_true")
    args = parser.parse_args()
    settings = Settings.load(args.config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / "slice1" / stamp
    output.mkdir(parents=True, exist_ok=False)
    lifecycle = temporary_model(output) if args.start_model else nullcontext(settings.base_url)
    with lifecycle as base_url:
        settings.base_url = base_url
        report = asyncio.run(probes(settings, output))
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ["total", "completed", "strict_matches", "database_unchanged"]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
