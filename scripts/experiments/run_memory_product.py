"""Actual CLI/session product check; run only after the full prediction pipeline releases GPU."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy",
        choices=["simplemem", "sql_episode", "state", "state_on_demand"],
        default="simplemem",
    )
    parser.add_argument("--serving", choices=["C0", "C1"], default="C0")
    parser.add_argument("--relative-probe", action="store_true")
    parser.add_argument(
        "--regression", action="store_true", help="Also check all 40 V1 business cases"
    )
    args = parser.parse_args()
    if args.relative_probe and args.policy not in {"state", "state_on_demand"}:
        parser.error("Relative-date probe requires a V2.1 state policy")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "scripts/experiments"))
    from http_server import http_server
    from model_server import temporary_model

    from agentserve.config import Settings
    from agentserve.evaluation.sql import SQLEvaluator
    from agentserve.workloads.memory import reference

    output = (
        root
        / (
            "results/v2.1-memory"
            if args.policy in {"state", "state_on_demand"}
            else "results/v2-memory"
        )
        / ("product-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir()
    result = {"state": "started", "policy": args.policy}
    settings = Settings.load(root / "configs/app.v2.json")
    settings.memory.policy = args.policy
    if args.policy in {"state", "state_on_demand"}:
        settings.sql_prompt_version = "evidence_v21"
        settings.answer_style = "evidence_v21"
    settings.llm_concurrency = 2 if args.serving == "C0" else 4
    settings.state_database = output / "tasks.sqlite"
    before = hashlib.sha256(settings.databases["business_orders_v1"].path.read_bytes()).hexdigest()
    processes, handles = [], []
    questions = [
        "查询2026年1月Beijing客户的completed订单营收总计，单位分，只返回总计。",
        "继续刚才的营收分析，城市只改为Shanghai，其余条件保持不变。",
        "继续上次会话的营收分析，月份改为2026年2月，其他条件保持不变。",
    ]
    if args.relative_probe:
        questions[2] = "继续上次会话的营收分析，月份改为上个月，其他条件保持不变。"
    try:
        with ExitStack() as stack:
            resources = stack.enter_context(
                http_server(
                    "memory_resources:create_app",
                    output,
                    "cpu",
                    env={
                        "AGENT_EMBEDDING_PATH": str(root / "models/Qwen3-Embedding-0.6B"),
                        "AGENT_TOKENIZER_PATH": str(root / "models/Qwen2.5-7B-Instruct"),
                        "CUDA_VISIBLE_DEVICES": "",
                    },
                    app_dir=root / "scripts/experiments",
                    python_executable=os.environ.get("VLLM_PYTHON", str(Path.home() / "miniconda3/envs/agent-vllm/bin/python")),
                    readiness_seconds=120,
                )
            )
            settings.memory.resources_url = str(resources.base_url).rstrip("/")
            settings.memory.resources_identity = resources.get("/info").json()["identity"]
            settings.base_url = stack.enter_context(
                temporary_model(
                    output,
                    max_model_len=8192,
                    verify_gpu=True,
                    max_num_seqs=settings.llm_concurrency,
                )
            )
            config = output / "app.json"
            config.write_text(settings.model_dump_json(indent=2))
            env = {**os.environ, "PYTHONPATH": str(root / "src")}

            def launch(client, session_id, index):
                report = output / f"cli-{index + 1}.json"
                handle = (output / f"cli-{index + 1}.log").open("wb")
                handles.append(handle)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "agentserve",
                        "--database",
                        "business_orders_v1",
                        "--session-id",
                        session_id,
                        "--api-url",
                        str(client.base_url).rstrip("/"),
                        "--question",
                        questions[index],
                        "--output",
                        str(report),
                    ],
                    cwd=root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                processes.append(process)
                return process

            with http_server(
                "agentserve.api.app:create_app",
                output,
                "api-first",
                env={"AGENTSERVE_CONFIG": str(config)},
            ) as client:
                session = client.post("/sessions", json={"database_id": "business_orders_v1"})
                session.raise_for_status()
                sid = session.json()["session_id"]
                first = launch(client, sid, 0)
                deadline = time.monotonic() + 30
                while True:
                    history = client.get(f"/sessions/{sid}/history").json()["turns"]
                    if history:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("CLI first acceptance")
                    time.sleep(0.05)
                result["first_status_when_second_launched"] = history[0]["status"]
                second = launch(client, sid, 1)
                result["cli_exit_codes"] = [first.wait(timeout=120), second.wait(timeout=120)]
                assert all((output / f"cli-{i}.json").exists() for i in [1, 2]), (
                    "CLI integration failed before producing a report"
                )
                before_restart = client.get(f"/sessions/{sid}/history").json()
                mismatch = client.post(
                    "/tasks",
                    json={"session_id": sid, "database_id": "demo_orders", "question": "客户数量"},
                )
                assert mismatch.status_code == 409
                result["cross_database_status"] = mismatch.status_code
            with http_server(
                "agentserve.api.app:create_app",
                output,
                "api-restarted",
                env={"AGENTSERVE_CONFIG": str(config)},
            ) as client:
                assert client.get(f"/sessions/{sid}/history").json() == before_restart
                third = launch(client, sid, 2)
                result["cli_exit_codes"].append(third.wait(timeout=120))
                assert (output / "cli-3.json").exists(), "Resumed CLI produced no report"
                turns = client.get(f"/sessions/{sid}/history").json()["turns"]
                assert [t["turn_no"] for t in turns] == [1, 2, 3]
                rows = [client.get(f"/tasks/{t['task_id']}").json() for t in turns]
                assert rows[1]["started_at"] >= rows[0]["ended_at"]
                context = client.get(f"/tasks/{turns[2]['task_id']}/context").json()["context"]
                result["third_context_has_previous_source"] = bool(
                    context
                    and context["cutoff_turn"] == 2
                    and (
                        turns[1]["task_id"] in context["source_ids"]
                        or any(e["task_id"] == turns[1]["task_id"] for e in context["selected"])
                    )
                )
                result["history"] = turns
                result["third_context"] = context
            evaluator = SQLEvaluator({k: v.path for k, v in settings.databases.items()})
            try:
                labels = [
                    asyncio.run(
                        evaluator.evaluate(
                            "business_orders_v1",
                            (row["report"] or {}).get("sql"),
                            reference(city, month, "completed"),
                            ordered=True,
                        )
                    )
                    for row, city, month in zip(
                        rows, ["Beijing", "Shanghai", "Shanghai"], [1, 1, 2], strict=True
                    )
                ]
                if args.relative_probe:
                    from datetime import timedelta
                    from zoneinfo import ZoneInfo

                    anchor = datetime.fromtimestamp(
                        rows[2]["created_at"], ZoneInfo("Asia/Shanghai")
                    )
                    end = anchor.replace(day=1).strftime("%Y-%m-%d")
                    start = (
                        (anchor.replace(day=1) - timedelta(days=1))
                        .replace(day=1)
                        .strftime("%Y-%m-%d")
                    )
                    expected = (
                        "SELECT SUM(i.quantity*i.unit_price_cents) FROM orders o JOIN order_items i ON i.order_id=o.order_id JOIN customers c ON c.customer_id=o.customer_id WHERE c.city='Shanghai' AND o.status='completed' AND o.order_date>='"
                        + start
                        + "' AND o.order_date<'"
                        + end
                        + "'"
                    )
                    labels[2] = asyncio.run(
                        evaluator.evaluate(
                            "business_orders_v1",
                            (rows[2]["report"] or {}).get("sql"),
                            expected,
                            ordered=True,
                        )
                    )
                    result["relative_anchor"] = anchor.isoformat()
                    result["relative_expected_range"] = [start, end]
                    result["relative_frame_correct"] = context["plan"]["state"]["fields"]["date"][
                        "value"
                    ] == [start, end]
            finally:
                evaluator.close()
            result["evaluations"] = labels
            result["product_sql_correct"] = sum(l["status"] == "correct" for l in labels)
            if args.regression:
                from run_g5 import run_case

                regression = settings.model_copy(deep=True)
                regression.memory = None
                result["single_turn_regression"] = run_case(
                    regression,
                    output / "v1-business",
                    json.loads(
                        (root / "data/processed/business_orders_v1/questions.json").read_text()
                    ),
                    {
                        q["id"]: q
                        for q in json.loads(
                            (root / "data/gold/business_orders_v1/answers.json").read_text()
                        )
                    },
                    concurrency=4,
                    warmup=False,
                )
            passed = (
                result["product_sql_correct"] == 3
                and result["cli_exit_codes"] == [0, 0, 0]
                and result["third_context_has_previous_source"]
                and (not args.relative_probe or result["relative_frame_correct"])
            )
            result["state"] = "passed" if passed else "completed_with_model_failures"
    except BaseException as error:
        result.update(state="failed", error=repr(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for handle in handles:
            handle.close()
        result["query_database_unchanged"] = (
            hashlib.sha256(settings.databases["business_orders_v1"].path.read_bytes()).hexdigest()
            == before
        )
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({"output": str(output), "state": result["state"]}), flush=True)


if __name__ == "__main__":
    main()
