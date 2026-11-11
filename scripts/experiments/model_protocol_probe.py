"""Probe the AgentServe model contract with a configurable OpenAI-compatible endpoint."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from agentserve.config import Settings
from agentserve.domain.models import AgentError
from agentserve.llm.client import VLLMClient
from agentserve.memory.resources import MemoryResources

SCHEMA = {
    "type": "object",
    "properties": {"operation": {"type": "string", "enum": ["keep"]}},
    "required": ["operation"],
    "additionalProperties": False,
}
CASES = (
    ("named_sql", "Call execute_sql exactly once with SELECT 1 AS value.", {"query": True}),
    ("clarification", "Do not call the SQL tool. Reply exactly CLARIFY: Which month?", {
        "query": True, "allow_clarify": True,
    }),
    ("json_schema", "Return the JSON object with operation set to keep.", {
        "query": False, "json_mode": True, "json_schema": SCHEMA,
    }),
    ("answer", "Reply with the word ready.", {"query": False}),
)


async def run(args):
    settings = Settings.load(args.config)
    async with (
        httpx.AsyncClient(base_url=settings.base_url + "/", trust_env=False) as endpoint,
        httpx.AsyncClient(base_url=settings.memory.resources_url, trust_env=False) as cpu,
    ):
        resources = MemoryResources(cpu)
        await resources.connect(settings.memory.resources_identity)
        client = VLLMClient(endpoint, settings.model)
        rows = []
        for name, prompt, options in CASES:
            try:
                reply = await client.generate(
                    [{"role": "user", "content": prompt}], max_tokens=128,
                    deadline=time.monotonic() + 30, **options,
                )
            except AgentError as error:
                if name == "clarification" and error.code == "MEMORY_CONTEXT_REQUIRED":
                    rows.append({"case": name, "passed": True, "text": str(error)})
                    continue
                rows.append({"case": name, "passed": False, "error": error.code, "detail": str(error)})
                continue
            content = reply.message.get("content") or ""
            if name == "named_sql":
                passed = bool(reply.sql and reply.sql.strip().rstrip(";").lower() == "select 1 as value")
            elif name == "clarification":
                passed = reply.sql is None and content.startswith("CLARIFY:")
            elif name == "json_schema":
                passed = json.loads(content) == {"operation": "keep"}
            else:
                passed = content.strip().lower().rstrip(".") == "ready"
            rows.append({
                "case": name, "passed": passed, "request_id": reply.request_id,
                "sql": reply.sql, "text": content, "finish_reason": reply.finish_reason,
            })
        report = {"model": settings.model, "tokenizer_resources": resources.info.identity, "cases": rows}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not all(row["passed"] for row in rows):
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    asyncio.run(run(parser.parse_args()))

