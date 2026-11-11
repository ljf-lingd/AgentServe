"""Registered fixed-input C0 stability diagnosis; all attempts retained, not acceptance."""

import asyncio
import json
import time
from contextlib import ExitStack
from pathlib import Path

import httpx
from gpu_monitor import monitor
from model_server import temporary_model

from agentserve.domain.models import AgentError
from agentserve.evaluation.sql import SQLEvaluator
from agentserve.llm.client import VLLMClient

ROOT = Path(__file__).resolve().parents[2]


async def replay(endpoint, cases, output):
    evaluator = SQLEvaluator(
        {"business_orders_v1": ROOT / "data/databases/business_orders_v1/business_orders_v1.sqlite"}
    )
    rows = []
    try:
        async with httpx.AsyncClient(base_url=endpoint + "/", trust_env=False) as client:
            backend = VLLMClient(client, "agent-qwen")
            # Three complete, preselected repetitions; never stop after a desired score.
            for repeat in range(3):
                for case in cases:
                    started = time.monotonic()
                    inputs = case["input"]
                    row = {"id": case["id"], "repeat": repeat, "input": inputs}
                    try:
                        reply = await backend.generate(
                            inputs["messages"],
                            query=True,
                            max_tokens=inputs["max_tokens"],
                            deadline=time.monotonic() + 30,
                            allow_clarify=inputs.get("allow_clarify", False),
                        )
                        row["reply"] = reply.__dict__
                        row["evaluation"] = await evaluator.evaluate(
                            "business_orders_v1", reply.sql, case["gold"], ordered=True
                        )
                    except AgentError as error:
                        row.update(
                            error={"code": error.code, "detail": str(error)},
                            evaluation={"status": "incorrect"},
                        )
                    row["seconds"] = time.monotonic() - started
                    rows.append(row)
                    with (output / "replays.jsonl").open("a") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows
    finally:
        evaluator.close()


def main():
    output = ROOT / "results/v2.1-memory/c0-stability-20261104"
    output.mkdir(parents=True, exist_ok=False)
    old = ROOT / "results/v2-memory/acceptance-20261103T085452Z"
    records = {r["id"]: r for r in map(json.loads, (old / "sql_episode/turns.jsonl").open())}
    labels = json.loads((ROOT / "data/gold/memory_v2/acceptance.json").read_text())
    selected = []
    for tag, want, n in [
        ("inheritance", False, 3),
        ("inheritance", True, 2),
        ("long_reference", False, 2),
        ("correction", False, 2),
    ]:
        pool = [
            r
            for r in records.values()
            if tag in r["tags"] and (r["evaluation"]["status"] == "correct") == want
        ]
        for r in sorted(pool, key=lambda r: r["id"])[:n]:
            steps = [
                s for s in r["steps"]["steps"] if s["kind"] == "model" and s["input"].get("query")
            ]
            if steps and r["id"] not in [c["id"] for c in selected]:
                selected.append(
                    {"id": r["id"], "input": steps[0]["input"], "gold": labels[r["id"]]["sql"]}
                )
    # Published category regression input comes from its actual stored task trace.
    product = ROOT / "results/v2-memory/product-20261103T124830Z/v1-business"
    for path in product.rglob("*.json"):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "messages" in value:
            pass
    # Use frozen category trace from run_g5 raw task reports (same initial prompt hash).
    import sqlite3

    db = sqlite3.connect(f"file:{product / 'tasks.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    for row in db.execute("SELECT task_id,input_json FROM tasks"):
        if "category" in json.loads(row["input_json"])["question"]:
            question = json.loads(row["input_json"])["question"]
            if "营收" not in question:
                continue
            step = db.execute(
                "SELECT input_json FROM steps WHERE task_id=? AND kind='model' ORDER BY step_no LIMIT 1",
                (row["task_id"],),
            ).fetchone()
            gold = json.loads((ROOT / "data/gold/business_orders_v1/answers.json").read_text())
            label = next(x for x in gold if x["id"] == "business.category_revenue")
            selected.insert(
                0,
                {
                    "id": "business.category_revenue",
                    "input": json.loads(step[0]),
                    "gold": label["sql"],
                },
            )
            break
    db.close()
    registration = {
        "kind": "observed-input-development-diagnosis",
        "repeats": 3,
        "order": "repeat-major, fixed case order",
        "serving": "C0 4/4/2/2/0.45; sequential fixed-input calls",
        "cases": selected,
        "stop_rule": "all registered attempts, or infrastructure failure; no quality-based retry",
    }
    (output / "registration.json").write_text(
        json.dumps(registration, ensure_ascii=False, indent=2)
    )
    with ExitStack() as stack:
        stack.enter_context(monitor(output))
        endpoint = stack.enter_context(temporary_model(output, max_model_len=8192, verify_gpu=True))
        rows = asyncio.run(replay(endpoint, selected, output))
    summary = {
        case["id"]: [r["evaluation"]["status"] for r in rows if r["id"] == case["id"]]
        for case in selected
    }
    (output / "summary.json").write_text(
        json.dumps({"state": "completed", "results": summary}, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
