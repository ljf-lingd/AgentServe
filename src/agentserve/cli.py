"""CLI composition root: construct dependencies, run one task, close resources."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from pydantic import ValidationError

from agentserve.application.bootstrap import open_task_service
from agentserve.config import DEFAULT_APP_CONFIG, Settings
from agentserve.domain.models import TaskInput


async def run(args) -> int:
    task = TaskInput(
        database_id=args.database,
        question=args.question,
        session_id=args.session_id,
        mode=getattr(args, "mode", None),
        answer_mode=getattr(args, "answer_mode", None),
        as_of=getattr(args, "as_of", None),
        clause_id=getattr(args, "clause_id", None),
        metric=getattr(args, "metric_input", None),
        analysis=getattr(args, "analysis_input", None),
    )
    if args.session_id:
        async with httpx.AsyncClient(base_url=args.api_url, trust_env=False, timeout=30) as client:
            response = await client.post("/tasks", json=task.model_dump(exclude_none=True))
            response.raise_for_status()
            task_id = response.json()["task_id"]
            while True:
                response = await client.get(f"/tasks/{task_id}")
                response.raise_for_status()
                row = response.json()
                if row["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    result = row["report"] or row
                    status = row["status"]
                    break
                await asyncio.sleep(0.2)
    else:
        settings = Settings.load(args.config)
        async with open_task_service(settings) as service:
            report = await service.run(task)
        result, status = report.to_dict(), report.status
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output)
    return 0 if status == "COMPLETED" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a read-only AgentServe task")
    parser.add_argument("--config", type=Path, default=DEFAULT_APP_CONFIG)
    parser.add_argument("--database", default="demo_orders")
    parser.add_argument("--question")
    parser.add_argument("--mode", choices=["sql", "knowledge", "metric", "analysis"], default="sql")
    parser.add_argument("--answer-mode", choices=["explain"], help="Explain the SQL result")
    parser.add_argument("--metric")
    parser.add_argument("--start", help="Inclusive window start (YYYY-MM-DD)")
    parser.add_argument("--end", help="Exclusive window end (YYYY-MM-DD)")
    parser.add_argument("--city")
    parser.add_argument("--group-by", choices=["city", "month"])
    parser.add_argument("--as-of", help="Knowledge rule date (YYYY-MM-DD)")
    parser.add_argument("--clause-id", help="Limit knowledge lookup to a registered clause")
    parser.add_argument("--session-id", help="Use a persistent session through the Task API")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--output", type=Path, help="Write a new report; existing files are preserved"
    )
    args = parser.parse_args()
    if args.mode == "analysis":
        if args.metric is not None or args.group_by is not None:
            parser.error("--metric and --group-by are unavailable in analysis mode")
        args.analysis_input = {"start": args.start, "end": args.end, "city": args.city}
    elif args.mode == "metric" or any((args.metric, args.start, args.end, args.city, args.group_by)):
        args.metric_input = {
            "name": args.metric, "start": args.start, "end": args.end,
            "city": args.city, "group_by": args.group_by,
        }
    try:
        status = asyncio.run(run(args))
    except ValidationError as error:
        parser.error(str(error))
    raise SystemExit(status)
