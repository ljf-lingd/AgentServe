"""Read-only NVIDIA samples; never infer GPU activity from mocked responses."""

import csv
import json
import subprocess
import threading
import time
from contextlib import contextmanager


def snapshot(label):
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = [
        dict(
            zip(
                ["index", "uuid", "name", "used_mib", "total_mib", "utilization_percent"],
                [v.strip() for v in row],
                strict=True,
            )
        )
        for row in csv.reader(gpu_csv.splitlines())
    ]
    processes = []
    for row in csv.reader(processes_csv.splitlines()):
        if row:
            pid, name, memory, uuid = [v.strip() for v in row]
            processes.append(
                {"pid": int(pid), "name": name, "used_mib": int(memory), "gpu_uuid": uuid}
            )
    return {"label": label, "wall_time": time.time(), "gpus": gpus, "processes": processes}


@contextmanager
def monitor(output, interval=1):
    stop = threading.Event()
    path = output / "gpu-samples.jsonl"

    def sample():
        with path.open("a", encoding="utf-8") as handle:
            while not stop.is_set():
                try:
                    record = snapshot("periodic")
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    record = {"wall_time": time.time(), "error": repr(error)}
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                stop.wait(interval)

    worker = threading.Thread(target=sample, name="g5-gpu-observer", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=10)
