"""Bounded closed-loop load against the actual Task API, with offline evidence collection."""

import asyncio
import json
import time
from uuid import uuid4

import httpx


async def run_closed_loop(base_url, jobs, *, concurrency, output, transport=None, on_progress=None):
    output.mkdir(parents=True, exist_ok=False)
    remaining = iter(enumerate(jobs))
    records = []
    run_key = uuid4().hex
    async with httpx.AsyncClient(
        base_url=base_url, trust_env=False, timeout=10, transport=transport
    ) as client:

        async def worker():
            while (item := next(remaining, None)) is not None:
                index, job = item
                record = {
                    "index": index,
                    "job": job,
                    "accepted": False,
                    "task": None,
                    "task_id": None,
                }
                try:
                    response = await client.post(
                        "/tasks",
                        json={k: job[k] for k in ("database_id", "question")},
                        headers={"Idempotency-Key": f"{run_key}-{index}"},
                    )
                    record["http_status"] = response.status_code
                    if response.status_code not in (200, 202):
                        record.update(status="REJECTED", error=response.text[:1000])
                        records.append(record)
                        if on_progress is not None:
                            on_progress(len(records), len(jobs))
                        continue
                    task_id = response.json()["task_id"]
                    record.update(accepted=True, task_id=task_id)
                    async with asyncio.timeout(130):
                        while True:
                            current = await client.get(f"/tasks/{task_id}")
                            current.raise_for_status()
                            task = current.json()
                            if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                                record.update(task=task, status=task["status"], error=task["error"])
                                break
                            await asyncio.sleep(0.02)
                except (httpx.HTTPError, TimeoutError) as error:
                    record.update(status="CLIENT_ERROR", error=repr(error))
                records.append(record)
                if on_progress is not None:
                    on_progress(len(records), len(jobs))

        started = time.monotonic()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        window = time.monotonic() - started
        # Audit fetching and independent quality evaluation do not delay measured arrivals.
        for record in records:
            if record["task_id"]:
                path = f"/tasks/{record['task_id']}"
                events = await client.get(path + "/events?limit=1000")
                history = await client.get(path + "/steps")
                events.raise_for_status()
                history.raise_for_status()
                record["events"] = events.json()["events"]
                record["history"] = history.json()
    records.sort(key=lambda r: r["index"])
    for record in records:
        (output / f"{record['index']:04d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {"concurrency": concurrency, "window_seconds": window, "records": records}
