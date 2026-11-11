"""Toggle AgentServe SAM on the same loaded Qwen for paired retail measurements."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from retail_replay import DATABASE, gold, workloads
from retail_scale import percentile, query

from agentserve.api.app import create_app
from agentserve.config import Settings


async def run_arm(base, label, block, concurrency, sequences, output):
    config = base.model_copy(deep=True)
    enabled = label == "sam"
    config.session_sam = enabled
    config.session_sam_all_phases = enabled
    config.task_concurrency = concurrency
    config.state_database = output / f"{block}-{label}.sqlite"
    app = create_app(config, web_mode="")
    gate = asyncio.Semaphore(concurrency)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
    ) as client:
        async def conversation(sequence):
            result = (await client.post("/sessions", json={"database_id": DATABASE})).raise_for_status()
            session_id = result.json()["session_id"]
            results = []
            for turn in sequence:
                row = await query(client, turn, gate, session_id=session_id)
                steps = (await client.get(f"/tasks/{row['task_id']}/steps")).raise_for_status().json()
                if isinstance(steps, dict):
                    steps = steps["steps"]
                metrics = [
                    step["output"]["reply"].get("metrics") or {}
                    for step in steps if step.get("output") and "reply" in step["output"]
                ]
                row["drafts"] = sum(item.get("num_draft_tokens", 0) for item in metrics)
                row["accepted"] = sum(item.get("num_accepted_draft_tokens", 0) for item in metrics)
                row["requests_with_drafts"] = sum(item.get("num_draft_tokens", 0) > 0 for item in metrics)
                results.append(row)
            return results

        await conversation(sequences[0])
        results = await asyncio.gather(*(conversation(sequence) for sequence in sequences))
    rows = [row for sequence in results for row in sequence]
    return {
        "arm": label, "block": block, "concurrency": concurrency, "n": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "jct_p95_seconds": percentile([row["jct_seconds"] for row in rows], .95),
        "draft_tokens": sum(row["drafts"] for row in rows),
        "accepted_tokens": sum(row["accepted"] for row in rows),
        "requests_with_drafts": sum(row["requests_with_drafts"] for row in rows),
        "sequences": results,
    }


async def main(args):
    args.output.mkdir()
    settings = Settings.load(args.config)
    settings.base_url = args.model_url
    settings.memory.resources_url = args.resources_url
    formal = workloads("formal")
    sequences = gold(settings.databases[DATABASE].path, [formal[i] for i in (0, 8, 16, 20, 24, 25, 26, 27)])
    report = []
    for block, concurrency, order in ((1, 1, ("off", "sam")), (2, 4, ("sam", "off"))):
        for arm in order:
            result = await run_arm(settings, arm, block, concurrency, sequences, args.output)
            report.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "sequences"}), flush=True)
            (args.output / "results.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--resources-url", required=True)
    asyncio.run(main(parser.parse_args()))

