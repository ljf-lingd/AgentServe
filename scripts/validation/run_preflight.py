"""Check SQL boundaries and a real Qwen -> SQL -> tool result -> answer chain."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlite_tool import execute_query

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data/databases/demo_orders/demo_orders.sqlite"
OUTPUT = ROOT / "results/preflight"
STATUS = ROOT / "logs/preflight-status.json"
TOOLS = [{"type": "function", "function": {
    "name": "execute_sql", "description": "Execute one read-only SQLite query on demo_orders.",
    "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                   "required": ["sql"], "additionalProperties": False},
}}]


class SQLArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sql: str = Field(min_length=1, max_length=16000)


def check_sql_tool():
    cases = [
        ("read", "SELECT COUNT(*) FROM customers", {}, "ok"),
        ("valid_cte", "WITH counts AS (SELECT COUNT(*) AS n FROM orders) SELECT n FROM counts", {}, "ok"),
        ("syntax_error", "SELEC name FROM customers", {}, "SQL_EXECUTION_ERROR"),
        ("unknown_column", "SELECT missing_column FROM customers", {}, "SQL_EXECUTION_ERROR"),
        ("deny_update", "UPDATE customers SET name='changed'", {}, "POLICY_REJECTED"),
        ("deny_cte_delete", "WITH ids AS (SELECT customer_id FROM customers) DELETE FROM customers WHERE customer_id IN ids", {}, "POLICY_REJECTED"),
        ("deny_attach", "ATTACH DATABASE ':memory:' AS other", {}, "POLICY_REJECTED"),
        ("deny_pragma", "PRAGMA query_only=OFF", {}, "POLICY_REJECTED"),
        ("deny_extension", "SELECT load_extension('not_allowed')", {}, "POLICY_REJECTED"),
        ("deny_multi_statement", "SELECT 1; SELECT 2", {}, "SQL_EXECUTION_ERROR"),
        ("row_limit", "SELECT customer_id FROM customers ORDER BY customer_id", {"row_limit": 2}, "ok"),
        ("byte_limit", "SELECT 'abcdefghijklmnopqrstuvwxyz' || 'abcdefghijklmnopqrstuvwxyz' AS value FROM customers", {"byte_limit": 240}, "ok"),
        ("oversize_cell", "SELECT '" + "x" * 1000 + "' AS value", {"byte_limit": 240}, "RESULT_TOO_LARGE"),
        ("oversize_columns", 'SELECT 1 AS "' + "x" * 400 + '" WHERE 0', {"byte_limit": 240}, "RESULT_TOO_LARGE"),
        ("real_timeout", "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT SUM(x) FROM n", {"timeout_seconds": 0.02}, "SQL_TIMEOUT"),
        ("read_after_timeout", "SELECT COUNT(*) FROM customers", {}, "ok"),
    ]
    before = hashlib.sha256(DATABASE.read_bytes()).hexdigest()
    records = []
    for name, sql, options, expected in cases:
        started = time.monotonic()
        result = execute_query(DATABASE, sql, **options)
        elapsed = time.monotonic() - started
        passed = result["status"] == expected
        if name == "read" or name == "read_after_timeout":
            passed &= result["rows"] == [[6]]
        elif name == "valid_cte":
            passed &= result["rows"] == [[12]]
        elif name == "row_limit":
            passed &= result["truncated"] and result["rows"] == [[1], [2]]
        elif name == "byte_limit":
            passed &= result["truncated"] and result["limit_reason"] == "bytes"
            passed &= len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) <= 240
        elif name == "real_timeout":
            passed &= elapsed < 2
        records.append({"id": name, "passed": bool(passed), "elapsed_seconds": elapsed, "result": result})
    after = hashlib.sha256(DATABASE.read_bytes()).hexdigest()
    report = {"passed": all(case["passed"] for case in records) and before == after,
              "database_unchanged": before == after, "database_sha256": after, "cases": records}
    (OUTPUT / "sql-tool-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("SQL boundary checks failed; see results/preflight/sql-tool-checks.json")
    return report


def verify_model(client, question, schema, gold):
    messages = [{"role": "system", "content": (
        "You answer questions about a fixed SQLite database using execute_sql. "
        "Use exactly one read-only SQL query. Never invent database values. "
        "All monetary amounts are integer cents; do not convert their units. "
        "statuses are completed/pending/cancelled; city values are Beijing/Shanghai/Shenzhen. "
        "After the tool result, answer briefly in Chinese using only that result. Schema:\n" + schema)},
        {"role": "user", "content": question["question"]}]
    started = time.monotonic()
    response = client.post("/v1/chat/completions", json={
        "model": "agent-qwen", "messages": messages, "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "execute_sql"}},
        "temperature": 0, "max_tokens": 256,
    })
    response.raise_for_status()
    generation = response.json()
    message = generation["choices"][0]["message"]
    calls = message.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "execute_sql":
        raise ValueError("Expected exactly one execute_sql call")
    call = calls[0]
    arguments = SQLArguments.model_validate_json(call["function"]["arguments"])
    tool = execute_query(DATABASE, arguments.sql)
    # Gold is used only here for result comparison; it is never put in messages.
    correct = tool["status"] == "ok" and not tool["truncated"] and tool["rows"] == gold["rows"]
    messages.extend([
        {"role": "assistant", "content": message["content"], "tool_calls": calls},
        {"role": "tool", "tool_call_id": call["id"],
         "content": json.dumps(tool, ensure_ascii=False, separators=(",", ":"))},
    ])
    response = client.post("/v1/chat/completions", json={
        "model": "agent-qwen", "messages": messages, "tools": TOOLS,
        "tool_choice": "none", "temperature": 0, "max_tokens": 192,
    })
    response.raise_for_status()
    final = response.json()
    final_message = final["choices"][0]["message"]
    answer = final_message["content"]
    answered = isinstance(answer, str) and bool(answer.strip()) and not final_message.get("tool_calls")
    return {"id": question["id"], "question": question["question"], "sql": arguments.sql,
            "tool_result": tool, "expected_rows": gold["rows"], "sql_correct": correct,
            "final_answer": answer, "final_answer_nonempty": answered,
            "pipeline_completed": answered and tool["status"] == "ok",
            "elapsed_seconds": time.monotonic() - started,
            "generation_response": generation, "answer_response": final}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    report = {"state": "checking_tools", "purpose": "preflight_only_not_benchmark",
              "formal_experiments_run": False, "synthetic_scenarios_executed": False, "probes": []}
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [Path(__file__), ROOT / "scripts/validation/sqlite_tool.py", ROOT / "scripts/setup/prepare_demo.py"]}
    STATUS.write_text(json.dumps(report))
    checks = check_sql_tool()
    report["sql_tool_checks_passed"] = len(checks["cases"])
    questions = json.loads((ROOT / "data/processed/demo_orders/questions.json").read_text())
    gold = {row["id"]: row for row in json.loads((ROOT / "data/gold/demo_orders/answers.json").read_text())}
    schema = (ROOT / "data/processed/demo_orders/schema.sql").read_text()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", PORT=str(port))
    log = (ROOT / "logs/vllm/preflight-server.log").open("ab", buffering=0)
    server = subprocess.Popen(["bash", str(ROOT / "serving/vllm/launch-qwen.sh")],
                              env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report.update(state="starting_model", port=port, server_pid=server.pid)
    STATUS.write_text(json.dumps(report))
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(90, connect=5)) as client:
            deadline = time.monotonic() + 480
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Temporary vLLM exited: {server.returncode}")
                try:
                    ready = client.get("/v1/models", timeout=2)
                    ready.raise_for_status()
                    report["model_catalog"] = ready.json()
                    break
                except httpx.HTTPError:
                    time.sleep(2)
            else:
                raise TimeoutError("Temporary vLLM did not become ready")
            report["state"] = "running_probes"
            for question in questions:
                try:
                    record = verify_model(client, question, schema, gold[question["id"]])
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
                    # Probe-level reporting preserves failures without hiding their traceback.
                    record = {"id": question["id"], "pipeline_completed": False,
                              "sql_correct": False, "final_answer_nonempty": False,
                              "error": repr(error), "traceback": traceback.format_exc()}
                report["probes"].append(record)
                STATUS.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(json.dumps({key: record[key] for key in ["id", "pipeline_completed", "sql_correct"]}), flush=True)
        report["pipeline_completed"] = sum(row["pipeline_completed"] for row in report["probes"])
        report["sql_correct"] = sum(row["sql_correct"] for row in report["probes"])
        report["total"] = len(questions)
        report["final_answers_automatically_graded"] = False
        report["state"] = "passed" if report["pipeline_completed"] == report["sql_correct"] == len(questions) else "completed_with_failures"
    except Exception as error:
        report.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        log.close()
        report["temporary_server_closed"] = server.poll() is not None
        report["database_unchanged_after_model"] = checks["database_sha256"] == hashlib.sha256(DATABASE.read_bytes()).hexdigest()
        (OUTPUT / "model-probes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        STATUS.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ["state", "total", "pipeline_completed", "sql_correct", "temporary_server_closed"]}), flush=True)


if __name__ == "__main__":
    main()
