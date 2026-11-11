"""G3 real-model capacity comparison and explicitly injected HTTP fault probes.

The small comparison is exploratory: three repetitions, eight measured tasks per
condition, fixed model and data. It is not a production load or stable p95 claim.
"""

import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import httpx
from http_server import http_server
from model_server import temporary_model

from agentserve.config import Settings

ROOT = Path(__file__).resolve().parents[2]
IDS = [
    "customers_count",
    "orders_count",
    "completed_count",
    "products_count",
    "shanghai_count",
    "beijing_names",
    "completed_revenue",
    "orders_per_customer",
]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def terminal(client, task_id):
    async with asyncio.timeout(110):
        while True:
            response = await client.get(f"/tasks/{task_id}")
            response.raise_for_status()
            task = response.json()
            if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return task
            await asyncio.sleep(0.01)


async def batch(url, questions, gold, output):
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=10) as client:
        warmups = []
        for question in questions[:2]:
            response = await client.post(
                "/tasks", json={k: question[k] for k in ("database_id", "question")}
            )
            response.raise_for_status()
            warmups.append(await terminal(client, response.json()["task_id"]))
        write(output / "warmups.json", warmups)
        started = time.monotonic()
        responses = await asyncio.gather(
            *(
                client.post("/tasks", json={k: q[k] for k in ("database_id", "question")})
                for q in questions
            )
        )
        for response in responses:
            assert response.status_code == 202, response.text
        tasks = await asyncio.gather(*(terminal(client, r.json()["task_id"]) for r in responses))
        window = time.monotonic() - started
        records, all_events = [], []
        for question, task in zip(questions, tasks, strict=True):
            events_response = await client.get(f"/tasks/{task['task_id']}/events?limit=1000")
            events_response.raise_for_status()
            events = events_response.json()["events"]
            all_events.extend(events)
            write(output / f"{question['id']}.json", {"task": task, "events": events})
            result = task["report"]["query_result"] if task["report"] else None
            match = bool(
                result
                and result["status"] == "ok"
                and not result["truncated"]
                and result["rows"] == gold[question["id"]]["rows"]
            )
            records.append(
                {
                    "id": question["id"],
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "strict_match": match,
                    "jct_seconds": task["ended_at"] - task["created_at"],
                    "application_queue_seconds": task["started_at"] - task["created_at"],
                    "error": task["error"],
                }
            )
        active, peak = 0, 0
        waits = []
        for event in sorted(all_events, key=lambda e: e["event_id"]):
            if event["type"] == "MODEL_STARTED":
                active += 1
                peak = max(peak, active)
                waits.append(event["payload"]["capacity_wait_seconds"])
            elif event["type"] in {"MODEL_COMPLETED", "MODEL_FAILED"}:
                active -= 1
            assert active >= 0
        assert active == 0
        completed = sum(r["status"] == "COMPLETED" for r in records)
        correct = sum(r["strict_match"] for r in records)
        jcts = [r["jct_seconds"] for r in records]
        result = {
            "submitted": len(questions),
            "accepted": len(tasks),
            "completed": completed,
            "correct": correct,
            "window_seconds": window,
            "completed_tasks_per_second": completed / window,
            "correct_tasks_per_second": correct / window,
            "mean_jct_seconds": statistics.mean(jcts),
            "p95_jct_seconds_descriptive": statistics.quantiles(jcts, n=100, method="inclusive")[
                94
            ],
            "mean_model_capacity_wait_seconds": statistics.mean(waits),
            "observed_model_peak": peak,
            "warmup_count": len(warmups),
            "records": records,
        }
        write(output / "batch.json", result)
        return result


def fault_probes(endpoint, output):
    proxy_config = output / "proxy.json"
    write(proxy_config, {"target_url": endpoint, "audit_path": str(output / "proxy-audit.jsonl")})
    settings = Settings.load(ROOT / "configs/app.demo.json")
    settings.task_concurrency, settings.llm_concurrency = 1, 1
    settings.state_database = output / "tasks.sqlite"
    with http_server(
        "fault_proxy:create_proxy",
        output,
        "proxy",
        env={"AGENTSERVE_PROXY_CONFIG": str(proxy_config)},
        app_dir=ROOT / "scripts/experiments",
    ) as proxy:
        settings.base_url = str(proxy.base_url).rstrip("/") + "/v1"
        config = output / "app.json"
        config.write_text(settings.model_dump_json(indent=2))
        with http_server(
            "agentserve.api.app:create_app", output, "api", env={"AGENTSERVE_CONFIG": str(config)}
        ) as client:

            def submit(question):
                response = client.post(
                    "/tasks", json={"database_id": "demo_orders", "question": question}
                )
                assert response.status_code == 202, response.text
                return response.json()["task_id"]

            def wait_task(task_id):
                deadline = time.monotonic() + 100
                while time.monotonic() < deadline:
                    task = client.get(f"/tasks/{task_id}").json()
                    if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                        return task
                    time.sleep(0.01)
                raise TimeoutError("Fault probe task did not reach terminal state")

            proxy.post("/control/fail_once").raise_for_status()
            retried_id = submit("一共有多少位客户？")
            retried = wait_task(retried_id)
            retry_events = client.get(f"/tasks/{retried_id}/events").json()["events"]
            write(output / "injected-503-task.json", {"task": retried, "events": retry_events})
            assert retried["status"] == "COMPLETED"
            assert retried["report"]["query_result"]["rows"] == [[6]]
            assert retried["report"]["model_retries"] == 1 and retried["report"]["model_calls"] == 3

            proxy.post("/control/hold_response").raise_for_status()
            running_id = submit("一共有多少笔订单？")
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if proxy.get("/healthz").json()["held"] == 1:
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("Real upstream response was not held")
            queued_id = submit("状态为 completed 的订单有多少笔？")
            assert client.get(f"/tasks/{queued_id}").json()["status"] == "QUEUED"
            assert client.post(f"/tasks/{queued_id}/cancel").status_code == 202
            assert client.post(f"/tasks/{running_id}/cancel").status_code == 202
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                events = client.get(f"/tasks/{running_id}/events").json()["events"]
                if any(e["type"] == "APPLICATION_EXECUTION_STOPPED" for e in events):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("Application execution did not stop after cancellation")
            proxy.post("/control/release").raise_for_status()
            proxy.post("/control/normal").raise_for_status()
            followup_id = submit("一共有多少位客户？")
            followup = wait_task(followup_id)
            assert followup["status"] == "COMPLETED" and followup["report"]["query_result"][
                "rows"
            ] == [[6]]
            for task_id, name in ((running_id, "cancel-running"), (queued_id, "cancel-queued")):
                task = client.get(f"/tasks/{task_id}").json()
                events = client.get(f"/tasks/{task_id}/events").json()["events"]
                write(output / f"{name}.json", {"task": task, "events": events})
                assert task["status"] == "CANCELLED" and task["report"] is None
                assert not any(e["type"] == "TASK_COMPLETED" for e in events)
                if name == "cancel-queued":
                    assert not any(e["type"] == "MODEL_STARTED" for e in events)
            write(output / "followup.json", followup)
            return {
                "injected_503_recovered": True,
                "model_requests_in_retry_task": 3,
                "queued_cancel_no_model_start": True,
                "running_cancel_stable_after_release": True,
                "followup_succeeded": True,
                "cancellation_window": "real_upstream_completed_response_held_before_application_receive",
                "gpu_generation_stopped_by_cancel": "not_measured_upstream_already_completed",
            }


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, metric, label in zip(
        axes,
        ["completed_tasks_per_second", "mean_jct_seconds"],
        ["Completed task/s", "Mean task JCT (s)"],
        strict=True,
    ):
        for capacity in (1, 2):
            values = [
                r[metric] for r in report["capacity_runs"] if r["llm_concurrency"] == capacity
            ]
            axis.scatter([capacity] * len(values), values, color="#2563eb", s=28)
            axis.plot(
                [capacity - 0.15, capacity + 0.15],
                [statistics.mean(values)] * 2,
                color="#111827",
                linewidth=2,
            )
        axis.set_xticks([1, 2], ["LLM limit 1", "LLM limit 2"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Real Qwen exploratory comparison: 3 runs x 8 tasks; task limit 4")
    figure.tight_layout()
    figure.savefig(output / "capacity-comparison.png", dpi=180)
    plt.close(figure)


def main():
    output = ROOT / "results/slice3" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    questions_path = ROOT / "data/processed/demo_orders/questions.json"
    gold_path = ROOT / "data/gold/demo_orders/answers.json"
    by_id = {q["id"]: q for q in json.loads(questions_path.read_text())}
    questions = [by_id[key] for key in IDS]
    gold = {q["id"]: q for q in json.loads(gold_path.read_text())}
    paths = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted(Path(__file__).parent.glob("*.py"))
        + [ROOT / "pyproject.toml", ROOT / "configs/app.demo.json"]
    )
    report = {
        "purpose": "g3_real_model_exploratory_comparison_and_controlled_faults",
        "run_id": output.name,
        "state": "prepared",
        "question_ids": IDS,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "data_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [questions_path, gold_path]
        },
        "model_manifest": json.loads(
            (ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json").read_text()
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["fastapi", "httpx", "pydantic", "aiosqlite", "uvicorn"]
        },
        "capacity_runs": [],
        "cache_condition": "one_model_process_balanced_order_two_warmup_tasks_per_run_no_cache_reset",
    }
    write(output / "summary.json", report)
    settings = Settings.load(ROOT / "configs/app.demo.json")
    database = settings.databases["demo_orders"].path
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    try:
        with temporary_model(output) as endpoint:
            for sequence, capacity in enumerate([1, 2, 2, 1, 1, 2], start=1):
                case = output / f"capacity-{sequence}-limit-{capacity}"
                case.mkdir()
                config = settings.model_copy(deep=True)
                config.base_url, config.state_database = endpoint, case / "tasks.sqlite"
                config.task_concurrency, config.llm_concurrency = 4, capacity
                config_path = case / "app.json"
                config_path.write_text(config.model_dump_json(indent=2))
                with http_server(
                    "agentserve.api.app:create_app",
                    case,
                    "api",
                    env={"AGENTSERVE_CONFIG": str(config_path)},
                ) as client:
                    result = asyncio.run(batch(str(client.base_url), questions, gold, case))
                assert result["observed_model_peak"] <= capacity
                result.update(llm_concurrency=capacity, repetition_sequence=sequence)
                report["capacity_runs"].append(result)
                report["state"] = "running"
                write(output / "summary.json", report)
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in [
                                "llm_concurrency",
                                "repetition_sequence",
                                "completed",
                                "correct",
                                "completed_tasks_per_second",
                                "mean_jct_seconds",
                                "observed_model_peak",
                            ]
                        }
                    ),
                    flush=True,
                )
            fault_output = output / "controlled-faults"
            fault_output.mkdir()
            report["fault_probes"] = fault_probes(endpoint, fault_output)
        report["database_unchanged"] = before == hashlib.sha256(database.read_bytes()).hexdigest()
        report["state"] = "completed"
        plot(report, output)
    except Exception as error:
        report.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(output / "summary.json", report)
    print(
        json.dumps(
            {
                "run_id": output.name,
                "state": report["state"],
                "fault_probes": report["fault_probes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
