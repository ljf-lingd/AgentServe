"""Measure native vLLM Suffix with AgentServe's retail and model protocol workloads."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
from model_protocol_probe import SCHEMA
from retail_replay import DATABASE, gold, workloads
from retail_sam_pair import run_arm

from agentserve.config import Settings
from agentserve.llm.client import SQL_TOOLS

SQL = "select 1 as value"
TOOL = {"type": "function", "function": {"name": "execute_sql"}}
CALL = {
    "messages": [{"role": "user", "content": "Call execute_sql exactly once with SELECT 1 AS value."}],
    "tools": SQL_TOOLS,
}
PROTOCOL = {
    "named_tool": {**CALL, "tool_choice": TOOL},
    "auto_tool": {**CALL, "tool_choice": "auto"},
    "json_schema": {
        "messages": [{"role": "user", "content": "Return an object with operation set to keep."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "operation", "strict": True, "schema": SCHEMA},
        },
    },
    "text": {
        "messages": [{"role": "user", "content": (
            "Explain the difference between database transactions, checkpoints and backups "
            "to a junior engineer. Give three short paragraphs, one per concept."
        )}],
        "max_tokens": 512,
    },
}
COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def protocol_passed(case, choice):
    message = choice["message"]
    try:
        if case in {"named_tool", "auto_tool"}:
            calls = message.get("tool_calls") or []
            if len(calls) != 1:
                return False
            function = calls[0]["function"]
            sql = json.loads(function["arguments"])["sql"]
            return function["name"] == "execute_sql" and sql.strip().rstrip(";").lower() == SQL
        content = message.get("content") or ""
        if case == "json_schema":
            return json.loads(content) == {"operation": "keep"}
        return choice["finish_reason"] == "stop" and len(content.split()) >= 40
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


async def probe_protocol(client, model):
    rows = []
    for repeat in range(3):
        for case, payload in PROTOCOL.items():
            started = time.perf_counter()
            response = await client.post("chat/completions", json={
                "model": model, "max_tokens": 128, "temperature": 0,
                "return_token_ids": True, **payload,
            })
            row = {"case": case, "repeat": repeat, "seconds": time.perf_counter() - started,
                   "http_status": response.status_code, "passed": False}
            if response.is_success:
                body = response.json()
                choice = body["choices"][0]
                row.update(passed=protocol_passed(case, choice), request_id=body["id"],
                           finish_reason=choice["finish_reason"], usage=body.get("usage"),
                           metrics=body.get("metrics"), tokens=choice.get("token_ids"),
                           message=choice["message"])
            else:
                row["detail"] = response.text
            rows.append(row)
    return rows


async def counters(client):
    response = (await client.get(client.base_url.copy_with(path="/metrics"))).raise_for_status()
    values = {}
    for line in response.text.splitlines():
        metric, _, value = line.rpartition(" ")
        name = metric.partition("{")[0]
        if name in COUNTERS:
            values[name] = values.get(name, 0) + int(float(value))
    return values


async def main(args):
    settings = Settings.load(args.config)
    settings.base_url = args.model_url.rstrip("/")
    settings.model = args.model
    settings.memory.resources_url = args.resources_url
    # The existing replay always creates one warmup session before these eight sessions.
    formal = workloads("formal")
    reference = gold(settings.databases[DATABASE].path, [formal[i] for i in (0, 8, 16, 20, 24, 25, 26, 27)])
    sha256 = hashlib.sha256(json.dumps(reference, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=settings.base_url + "/", trust_env=False, timeout=60) as client:
        before = await counters(client)
        retail = await run_arm(settings, args.arm, args.block, 1, reference, args.output)
        after = await counters(client)
        protocol = await probe_protocol(client, args.model)
    # Step-ledger metrics exclude the warmup. The endpoint counters include it.
    report = {
        "retail": {key: retail[key] for key in (
            "arm", "block", "n", "correct", "jct_p95_seconds",
            "draft_tokens", "accepted_tokens", "requests_with_drafts",
            "model_replies", "replies_with_draft_metrics",
        )},
        "gold_sha256": sha256,
        "suffix_counters_including_warmup": {key: value - before.get(key, 0) for key, value in after.items()},
        "sessions": [[{key: row[key] for key in (
            "correct", "status", "task_id", "result", "error", "jct_seconds", "sql_seconds",
            "drafts", "accepted", "requests_with_drafts", "model_replies", "replies_with_draft_metrics",
        )} for row in sequence] for sequence in retail["sequences"]],
        "protocol": protocol,
    }
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["retail"], "protocol_correct": sum(row["passed"] for row in protocol),
                      "protocol_n": len(protocol), "counters": report["suffix_counters_including_warmup"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--resources-url", required=True)
    parser.add_argument("--model", default="agent-qwen")
    parser.add_argument("--arm", choices=("ar", "suffix-local", "suffix-global"), required=True)
    parser.add_argument("--block", type=int, required=True)
    asyncio.run(main(parser.parse_args()))
