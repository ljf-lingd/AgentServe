"""Lifecycle and identity receipts for the frozen v5 serving configurations."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import version

import httpx

from gpu_monitor import snapshot
from http_server import http_server
from v6_phase_benchmark import ENGINE_ENV, MODEL, ROOT, dump, engine_args, file_sha


def verify_model(model_dir):
    manifest = ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json"
    files = json.loads(manifest.read_bytes())["Data"]["Files"]
    hashes = {entry["Path"]: file_sha(model_dir / entry["Path"]) for entry in files}
    if hashes != {entry["Path"]: entry["Sha256"] for entry in files}:
        raise ValueError("Local Qwen files differ from the pinned snapshot")
    return {"manifest_sha256": file_sha(manifest), "files": hashes}


@contextmanager
def resources(output, vllm_python, model_dir, embedding_dir):
    env = {
        **ENGINE_ENV,
        "CUDA_VISIBLE_DEVICES": "",
        "AGENT_TOKENIZER_PATH": str(model_dir),
        "AGENT_EMBEDDING_PATH": str(embedding_dir),
    }
    with http_server(
        "memory_resources:create_app", output, "resources", env=env,
        app_dir=ROOT / "scripts/experiments", python_executable=vllm_python,
        readiness_seconds=90,
    ) as client:
        dump(output / "resources-identity.json", client.get("/info").raise_for_status().json())
        yield str(client.base_url).rstrip("/")


@contextmanager
def engine(output, arm, model_dir, vllm_python, gpu_uuid, *, server_args=None, extra_env=None):
    before = snapshot("before")
    gpu = next(row for row in before["gpus"] if row["uuid"] == gpu_uuid)
    if "RTX 4090" not in gpu["name"] or any(
        process["gpu_uuid"] == gpu_uuid for process in before["processes"]
    ):
        raise RuntimeError("The selected RTX 4090 must be free before starting each arm")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    arguments = engine_args(arm) if server_args is None else server_args
    cmd = [vllm_python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_dir),
           "--port", str(port), *arguments]
    controlled = {**ENGINE_ENV, "CUDA_VISIBLE_DEVICES": gpu_uuid,
                  "PYTHONPATH": str(ROOT / "serving/vllm"), **(extra_env or {})}
    dump(output / "engine-command.json", {"argv": cmd, "env": controlled, "gpu_before": before})
    url = f"http://127.0.0.1:{port}/v1"
    with (output / "vllm.log").open("xb", buffering=0) as log:
        server = subprocess.Popen(cmd, env={**os.environ, **controlled}, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(trust_env=False, timeout=2) as client:
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited with {server.returncode}; see {output}/vllm.log")
                    try:
                        response = client.get(url + "/models")
                        if response.is_success and MODEL in {m["id"] for m in response.json()["data"]}:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError(f"vLLM readiness timed out; see {output}/vllm.log")
            ready = snapshot("model_ready")
            owned = [p for p in ready["processes"] if p["gpu_uuid"] == gpu_uuid
                     and os.getpgid(p["pid"]) == server.pid]
            if not owned:
                raise RuntimeError("The model did not produce an owned compute process on the selected GPU")
            dump(output / "gpu-ready.json", {**ready, "owned": owned})
            yield url
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            dump(output / "engine-closed.json", {"pid": server.pid, "returncode": server.returncode,
                                                 "port": port, "gpu_after": snapshot("after")})


def environment(vllm_python):
    serving = subprocess.check_output(
        [vllm_python, "-c", "from importlib.metadata import version; "
         "import json; print(json.dumps({p: version(p) for p in ('vllm','torch','transformers')}))"],
        text=True,
    )
    return {"serving": json.loads(serving), "application": {"python": sys.version,
            **{p: version(p) for p in ("httpx", "pydantic", "fastapi", "aiosqlite")}}}
