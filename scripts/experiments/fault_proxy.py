"""Experiment-only proxy: injected HTTP errors and a held real upstream response.

This never fabricates a successful model response. Control endpoints bind only to
the owned temporary loopback process and are absent from the product API.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def create_proxy():
    config = json.loads(Path(os.environ["AGENTSERVE_PROXY_CONFIG"]).read_text())
    audit = Path(config["audit_path"])
    release = asyncio.Event()
    state = {"mode": "normal", "requests": 0, "injections": 0, "held": 0}

    def record(kind, **payload):
        with audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": kind, "wall_time": time.time(), **payload}) + "\n")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url=config["target_url"] + "/", trust_env=False, timeout=90
        ) as client:
            app.state.upstream = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return state

    @app.post("/control/{mode}")
    async def control(mode: str):
        if mode not in {"normal", "fail_once", "hold_response", "release"}:
            return JSONResponse(status_code=400, content={"error": "Unknown experiment mode"})
        if mode == "release":
            release.set()
        else:
            state["mode"] = mode
            release.clear()
        record("CONTROL", mode=mode)
        return state

    @app.post("/v1/chat/completions")
    async def forward(request: Request):
        body = await request.json()
        state["requests"] += 1
        sequence = state["requests"]
        mode = state["mode"]
        if mode == "fail_once":
            state["mode"] = "normal"
            state["injections"] += 1
            record("INJECTED_503", sequence=sequence)
            return JSONResponse(
                status_code=503,
                content={"error": "Experiment-injected 503, not a real service outage"},
            )
        record("UPSTREAM_STARTED", sequence=sequence)
        response = await request.app.state.upstream.post("chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        record(
            "UPSTREAM_COMPLETED",
            sequence=sequence,
            request_id=payload["id"],
            usage=payload.get("usage"),
        )
        if mode == "hold_response":
            state["held"] += 1
            record("RESPONSE_HELD", sequence=sequence)
            try:
                await asyncio.wait_for(release.wait(), timeout=30)
            finally:
                state["held"] -= 1
            record("RESPONSE_RELEASED", sequence=sequence)
        return JSONResponse(content=payload)

    return app
