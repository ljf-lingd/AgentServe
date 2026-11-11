"""CPU embeddings and exact generation tokenizer HTTP boundary."""

import asyncio

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agentserve.domain.models import AgentError


class ResourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identity: str = Field(min_length=1)
    embedding_model: str
    embedding_revision: str
    tokenizer_sha256: dict[str, str]


class MemoryResources:
    def __init__(self, client):
        self.client = client

    async def connect(self, expected):
        try:
            response = await self.client.get("info", timeout=30)
            response.raise_for_status()
            self.info = ResourceInfo.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as error:
            raise AgentError("MEMORY_RESOURCE_ERROR", str(error)) from error
        if expected and self.info.identity != expected:
            raise AgentError("MEMORY_RESOURCE_CHANGED", "CPU resource identity differs from config")

    async def _post(self, route, data, deadline):
        try:
            async with asyncio.timeout_at(deadline):
                response = await self.client.post(route, json=data, timeout=None)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise AgentError("MEMORY_RESOURCE_ERROR", str(error)) from error

    async def embed(self, texts, *, query, deadline):
        result = await self._post("embed", {"texts": texts, "query": query}, deadline)
        vectors = result.get("vectors", [])
        if len(vectors) != len(texts) or any(len(v) != 1024 for v in vectors):
            raise AgentError("MEMORY_RESOURCE_ERROR", "Invalid embedding dimensions")
        return vectors

    async def count(self, text, deadline):
        return (await self._post("count", {"texts": [text]}, deadline))["counts"][0]

    async def guard(self, messages, tools, max_tokens, limit, deadline):
        result = await self._post("count", {"messages": messages, "tools": tools}, deadline)
        if result["counts"][0] + max_tokens > limit:
            raise AgentError(
                "MODEL_CONTEXT_LIMIT", "Required input and reserved output exceed context"
            )
