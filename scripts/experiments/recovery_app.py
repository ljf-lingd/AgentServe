"""Experiment-only pause points for terminating an owned API process at an exact boundary."""

import asyncio
import json
import os
import time
from pathlib import Path

from agentserve.api.app import create_app


def create_recovery_app():
    config = json.loads(Path(os.environ["AGENTSERVE_RECOVERY_PROBE"]).read_text())
    audit = Path(config["audit_path"])

    def persist(record):
        with audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        pause = config["enabled"] and (record["stage"], record["step_no"], record["kind"]) == (
            config["stage"],
            config["step_no"],
            config["kind"],
        )
        if pause:
            Path(config["marker_path"]).write_text(json.dumps(record))
        return pause

    async def boundary(stage, task_id, step_no, kind):
        record = {
            "stage": stage,
            "task_id": task_id,
            "step_no": step_no,
            "kind": kind,
            "pid": os.getpid(),
            "wall_time": time.time(),
        }
        if await asyncio.to_thread(persist, record):
            await asyncio.Event().wait()

    return create_app(step_hook=boundary)
