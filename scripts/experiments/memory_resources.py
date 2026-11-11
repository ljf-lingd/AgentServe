"""Loopback CPU-only resource process. Run in the existing serving environment."""

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class Request(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=32)
    query: bool = False
    messages: list[dict] | None = None
    tools: list[dict] | None = None
    bypass_cache: bool = False


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        embedding = Path(os.environ["AGENT_EMBEDDING_PATH"])
        generation = Path(os.environ["AGENT_TOKENIZER_PATH"])
        manifest = json.loads((embedding / "agent-manifest.json").read_text())
        for name, expected in manifest["files"].items():
            digest = hashlib.sha256()
            with (embedding / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(2**20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError("Pinned embedding file changed: " + name)
        app.state.tokenizer = AutoTokenizer.from_pretrained(generation, local_files_only=True)
        app.state.embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding, padding_side="left", local_files_only=True
        )
        app.state.model = (
            AutoModel.from_pretrained(embedding, local_files_only=True, dtype=torch.float32)
            .cpu()
            .eval()
        )
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in generation.glob("*")
            if p.name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "chat_template.jinja",
            }
        }
        identity = {
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": manifest["revision"],
            "tokenizer_sha256": hashes,
        }
        material = {
            **identity,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_files": manifest["files"],
        }
        app.state.info = {
            **identity,
            "identity": hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest(),
        }
        app.state.count_slot = asyncio.Semaphore(1)
        app.state.embed_slot = asyncio.Semaphore(1)
        app.state.count_cache = OrderedDict()
        app.state.embed_cache = OrderedDict()
        app.state.metrics = {
            name: {"hits": 0, "misses": 0, "wait_seconds": 0.0, "compute_seconds": 0.0}
            for name in ["count", "embed"]
        }
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/info")
    async def info():
        return app.state.info

    @app.get("/metrics")
    async def metrics():
        return app.state.metrics

    async def cached(route, key, calculate, capacity, bypass=False):
        if bypass:
            async with getattr(app.state, route + "_slot"):
                return await asyncio.to_thread(calculate)
        cache = getattr(app.state, route + "_cache")
        metric = app.state.metrics[route]
        if key in cache:
            metric["hits"] += 1
            cache.move_to_end(key)
            return cache[key]
        queued = time.monotonic()
        async with getattr(app.state, route + "_slot"):
            metric["wait_seconds"] += time.monotonic() - queued
            if key in cache:
                metric["hits"] += 1
                cache.move_to_end(key)
                return cache[key]
            metric["misses"] += 1
            started = time.monotonic()
            value = await asyncio.to_thread(calculate)
            metric["compute_seconds"] += time.monotonic() - started
            cache[key] = value
            if len(cache) > capacity:
                cache.popitem(last=False)
            return value

    @app.post("/count")
    async def count(request: Request):
        def calculate():
            tokenizer = app.state.tokenizer
            if request.messages is not None:
                kwargs = {"tools": request.tools} if request.tools else {}
                ids = tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    **kwargs,
                )
                return [len(ids)]
            return [len(tokenizer.encode(t, add_special_tokens=False)) for t in request.texts]

        # Process identity pins tokenizer/template implementation; full request pins all options.
        return {
            "counts": await cached(
                "count", request.model_dump_json(), calculate, 512, request.bypass_cache
            )
        }

    @app.post("/embed")
    async def embed(request: Request):
        def calculate():
            import torch

            texts = [
                "Instruct: Given a conversational SQL question, retrieve relevant historical "
                "analysis conditions\nQuery: " + t
                if request.query
                else t
                for t in request.texts
            ]
            tokenizer = app.state.embedding_tokenizer
            if any(len(tokenizer.encode(t)) > 8192 for t in texts):
                raise HTTPException(422, "Embedding input exceeds limit; truncation disabled")
            batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            with torch.inference_mode():
                hidden = app.state.model(**batch).last_hidden_state
                # Official Qwen recipe: left padding makes the final position the last token.
                vectors = torch.nn.functional.normalize(hidden[:, -1], p=2, dim=1)
            return vectors.tolist()

        if not request.texts:
            raise HTTPException(422, "At least one text required")
        return {
            "vectors": await cached(
                "embed", request.model_dump_json(), calculate, 128, request.bypass_cache
            )
        }

    return app
