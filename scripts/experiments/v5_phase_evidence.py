"""Observe actual vLLM payloads and join them to AgentServe step receipts."""

import hashlib
import json
import time
from collections import defaultdict

import httpx

from agentserve.analysis.metrics import summarize as summarize_tasks
from v6_phase_benchmark import distribution

HISTOGRAMS = (
    "e2e_request_latency_seconds",
    "time_to_first_token_seconds",
    "inter_token_latency_seconds",
    "request_time_per_output_token_seconds",
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ModelTrace(httpx.AsyncHTTPTransport):
    def __init__(self):
        super().__init__(trust_env=False)
        self.calls = []

    async def handle_async_request(self, request):
        payload = json.loads(request.content)
        hints = payload.pop("vllm_xargs", {}).get("agentserve_v5_sam")
        call = {"model_input_sha256": digest_json(payload), "sam_enabled": hints is not None,
                "history_entries": len(json.loads(hints)) if hints is not None else 0,
                "tool_choice": payload.get("tool_choice"),
                "response_format": (payload.get("response_format") or {}).get("type")}
        self.calls.append(call)
        sent = time.monotonic()
        try:
            response = await super().handle_async_request(request)
            await response.aread()
        finally:
            call["http_seconds"] = time.monotonic() - sent
        call["status"] = response.status_code
        if response.is_success:
            raw = response.json()
            choice = raw["choices"][0]
            message = choice["message"]
            call.update(request_id=raw["id"], usage=raw.get("usage"), metrics=raw.get("metrics"),
                        finish_reason=choice["finish_reason"],
                        model_output_sha256=digest_json({
                            "content": message.get("content"),
                            "tools": [tool["function"] for tool in (message.get("tool_calls") or [])],
                        }))
        return response


def audit(rows, calls, arm):
    ids = {call["request_id"]: call for call in calls if "request_id" in call}
    joined = {}
    for name, row in rows.items():
        joined[name] = []
        for step in row["journal"]["steps"]:
            reply = (step["output"] or {}).get("reply")
            if step["kind"] != "model" or reply is None:
                continue
            inputs = step["input"]
            phase = inputs.get("purpose") or (
                "sql" if inputs["query"] else "knowledge" if inputs.get("json_schema") else "answer"
            )
            call = ids.get(reply["request_id"])
            expected = arm == "all" or (arm == "answer" and phase == "answer")
            joined[name].append({"step_no": step["step_no"], "phase": phase,
                                 "request_id": reply["request_id"], "expected_sam": expected,
                                 "call": call})
    steps = [step for task in joined.values() for step in task]
    linked = {step["request_id"] for step in steps if step["call"] is not None}
    return {"cases": joined, "model_attempts": len(calls), "committed_model_steps": len(steps),
            "linked_steps": sum(step["call"] is not None for step in steps),
            "unlinked_attempts": len(calls) - len(linked),
            "sam_mismatches": [step["request_id"] for step in steps if step["call"] is not None
                               and step["call"]["sam_enabled"] != step["expected_sam"]]}


def task_timing(rows, elapsed):
    records = [{"accepted": True, "task": row["task"], "status": row["task"]["status"],
                "events": row["events"], "history": row["journal"],
                "evaluation": {"status": "correct" if row["correct"] else "incorrect"}}
               for row in rows.values()]
    return summarize_tasks({"records": records, "window_seconds": elapsed})


def engine_timing(before, after):
    def samples(path):
        known = {f"vllm:{name}_{suffix}" for name in HISTOGRAMS for suffix in ("sum", "count")}
        values = defaultdict(float)
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.partition("{")[0].split(" ", 1)[0]
            if name in known:
                values[name] += float(line.rsplit(None, 1)[1])
        return values

    start, stop = samples(before), samples(after)
    result = {}
    for name in HISTOGRAMS:
        count, total = f"vllm:{name}_count", f"vllm:{name}_sum"
        if count in stop and total in stop:
            n = int(stop[count] - start[count])
            result[name] = {"n": n, "mean_seconds": (stop[total] - start[total]) / n if n else None}
    return result


def paired_paths(left, right):
    paths = {}
    for name in left["cases"]:
        lhs, rhs = left["cases"][name], right["cases"][name]
        left_calls, right_calls = left["trace"]["cases"][name], right["trace"]["cases"][name]
        left_report, right_report = lhs["task"].get("report") or {}, rhs["task"].get("report") or {}
        exact_calls = len(left_calls) == len(right_calls) and all(
            a["phase"] == b["phase"] and a["call"] is not None and b["call"] is not None
            and all(a["call"][key] == b["call"][key] for key in (
                "model_input_sha256", "model_output_sha256", "usage", "finish_reason",
            )) for a, b in zip(left_calls, right_calls)
        )
        paths[name] = {"both_correct": lhs["correct"] and rhs["correct"],
                       "both_model": bool(left_calls and right_calls),
                       "both_direct": not (left_calls or right_calls),
                       "same_model_steps": exact_calls,
                       "same_sql": left_report.get("sql") == right_report.get("sql"),
                       "delta_seconds": rhs["jct"] - lhs["jct"]}
    return {"cases": paths, "matched_model_jct_delta_seconds": distribution(
        row["delta_seconds"] for row in paths.values() if row["both_correct"] and row["both_model"]
        and row["same_model_steps"] and row["same_sql"]
    )}
