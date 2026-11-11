"""A real HTTP + Qwen task, application restart and persistent-result check."""

import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from model_server import temporary_model

from agentserve.config import Settings

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def api_server(config: Path, output: Path, phase: str):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env["AGENTSERVE_CONFIG"] = str(config)
    with (output / f"api-{phase}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "agentserve.api.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base_url, trust_env=False, timeout=5) as client:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"API exited during {phase}; see API log")
                    try:
                        ready = client.get("/healthz")
                        ready.raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError("API readiness deadline exceeded")
                yield client
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            (output / f"api-{phase}-lifecycle.json").write_text(
                json.dumps(
                    {
                        "pid": process.pid,
                        "port": port,
                        "closed": process.poll() is not None,
                        "returncode": process.returncode,
                    },
                    indent=2,
                )
            )


def main():
    output = ROOT / "results/slice2" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings.load(ROOT / "configs/app.demo.json")
    source = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__),
        Path(__file__).with_name("model_server.py"),
        ROOT / "pyproject.toml",
        ROOT / "configs/app.demo.json",
    ]
    report = {
        "purpose": "real_http_functional_probe_not_performance_benchmark",
        "run_id": output.name,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["fastapi", "uvicorn", "aiosqlite", "httpx", "pydantic"]
        },
    }
    database = settings.databases["demo_orders"].path
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with temporary_model(output) as endpoint:
        settings.base_url = endpoint
        settings.state_database = output / "tasks.sqlite"
        config = output / "app.json"
        config.write_text(settings.model_dump_json(indent=2))
        body = {"database_id": "demo_orders", "question": "一共有多少位客户？"}
        headers = {"Idempotency-Key": "slice2-real-customer-count"}
        with api_server(config, output, "first") as client:
            accepted = client.post("/tasks", json=body, headers=headers)
            assert accepted.status_code == 202, accepted.text
            task_id = accepted.json()["task_id"]
            report["acceptance"] = accepted.json()
            deadline = time.monotonic() + 100
            while time.monotonic() < deadline:
                task = client.get(f"/tasks/{task_id}").json()
                if task["status"] in {"COMPLETED", "FAILED"}:
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Task polling deadline exceeded")
            report["task"] = task
            events = client.get(f"/tasks/{task_id}/events").json()
            report["events"] = events
            (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            assert task["status"] == "COMPLETED", task
            assert task["report"]["query_result"]["rows"] == [[6]]
            assert task["report"]["answer"].strip()
        with api_server(config, output, "restarted") as client:
            saved = client.get(f"/tasks/{task_id}").json()
            assert saved == task
            replay = client.post("/tasks", json=body, headers=headers)
            assert replay.status_code == 200 and replay.json()["task_id"] == task_id
            assert client.get(f"/tasks/{task_id}/events").json() == events
            report.update(
                restart_preserved_result=True,
                idempotent_replay_after_restart=True,
                completed_model_calls=sum(e["type"] == "MODEL_COMPLETED" for e in events["events"]),
            )
    report["database_unchanged"] = before == hashlib.sha256(database.read_bytes()).hexdigest()
    report["passed"] = report["database_unchanged"]
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "run_id",
                    "passed",
                    "restart_preserved_result",
                    "idempotent_replay_after_restart",
                    "completed_model_calls",
                    "database_unchanged",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
