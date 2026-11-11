"""Shared lifecycle for owned temporary model servers used by functional probes."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def temporary_model(
    output: Path,
    *,
    max_model_len=2048,
    verify_gpu=False,
    max_num_seqs=2,
    gpu_memory_utilization=0.45,
):
    lifecycle = {"started_temporary_model": True}
    lifecycle["gpu_before"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
        text=True,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(PORT=str(port), CUDA_VISIBLE_DEVICES="0", MAX_MODEL_LEN=str(max_model_len))
    env.update(MAX_NUM_SEQS=str(max_num_seqs), GPU_MEMORY_UTILIZATION=str(gpu_memory_utilization))
    launch = ROOT / "serving/vllm/launch-qwen.sh"
    with (output / "vllm.log").open("ab", buffering=0) as log:
        server = subprocess.Popen(
            ["bash", str(launch)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        lifecycle.update(
            server_pid=server.pid,
            port=port,
            launch_script_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(
            json.dumps({"state": "starting_model", "output": str(output), **lifecycle}), flush=True
        )
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            deadline = time.monotonic() + 480
            with httpx.Client(trust_env=False) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"Temporary model exited: {server.returncode}")
                    try:
                        response = client.get(base_url + "/models", timeout=2)
                        response.raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(2)
                else:
                    raise TimeoutError("Model readiness deadline exceeded")
            if verify_gpu:
                from gpu_monitor import snapshot

                loaded = snapshot("model_ready")
                owned = []
                for process in loaded["processes"]:
                    if os.getpgid(process["pid"]) == server.pid:
                        owned.append(process)
                loaded["owned_model_processes"] = owned
                (output / "gpu-model-ready.json").write_text(json.dumps(loaded, indent=2) + "\n")
                if not owned:
                    raise RuntimeError("No GPU compute process belongs to this model server")
                print(
                    json.dumps(
                        {"state": "model_ready_on_gpu", "processes": owned, "gpus": loaded["gpus"]}
                    ),
                    flush=True,
                )
            yield base_url
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            lifecycle["temporary_model_closed"] = server.poll() is not None
            lifecycle["gpu_after"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            (output / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")
