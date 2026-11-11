"""Owned loopback HTTP processes for real API and controlled proxy experiments."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx


@contextmanager
def http_server(
    factory: str,
    output: Path,
    phase: str,
    *,
    env: dict,
    app_dir: Path | None = None,
    on_ready=None,
    python_executable: str | None = None,
    readiness_seconds: float = 30,
):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process_env = os.environ.copy()
    process_env.update(env)
    command = [
        python_executable or sys.executable,
        "-m",
        "uvicorn",
        factory,
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    if app_dir is not None:
        command.extend(["--app-dir", str(app_dir)])
    with (output / f"{phase}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, env=process_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5
            ) as client:
                deadline = time.monotonic() + readiness_seconds
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"HTTP process exited during {phase}; see log")
                    try:
                        response = client.get("/healthz")
                        response.raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError("HTTP process readiness deadline exceeded")
                if on_ready is not None:
                    on_ready(process)
                yield client
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            (output / f"{phase}-lifecycle.json").write_text(
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
