"""Experiment-only tool wait; every model response and SQL execution remains real."""

import asyncio
import json
import os
from pathlib import Path

from agentserve.api.app import create_app


def create_workload_app():
    probe = json.loads(Path(os.environ["AGENTSERVE_WORKLOAD_PROBE"]).read_text())

    async def delay(stage, task_id, step_no, kind):
        if stage == "before_execute" and kind == "sql":
            await asyncio.sleep(probe["sql_wait_seconds"])

    return create_app(step_hook=delay)
