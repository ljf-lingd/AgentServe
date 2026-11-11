"""The validated vLLM chat protocol, with one external-response validation boundary."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentserve.domain.models import AgentError, ModelReply, ModelRequestError


class SQLArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    sql: str = Field(min_length=1, max_length=16000)


SQL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute one read-only SQLite query.",
            "parameters": SQLArguments.model_json_schema(),
        },
    }
]


class FunctionCall(BaseModel):
    name: Literal["execute_sql"]
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: FunctionCall


class Message(BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    message: Message
    finish_reason: str


class Completion(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    choices: list[Choice] = Field(min_length=1, max_length=1)
    usage: dict | None = None
    metrics: object = None


def request_metrics(raw: object, sam: bool) -> dict | None:
    if not isinstance(raw, dict):
        return None
    values = {}
    for key in ("time_to_first_token_ms", "tokens_per_second", "mean_itl_ms"):
        value = raw.get(key)
        if type(value) in (int, float) and 0 <= value < float("inf"):
            values[key] = value
    if sam and isinstance(spec := raw.get("speculative_decoding"), dict):
        for key in ("num_draft_tokens", "num_accepted_draft_tokens"):
            value = spec.get(key)
            if type(value) is int and value >= 0:
                values[key] = value
    return values or None


class LLMBackend(Protocol):
    async def generate(
        self,
        messages: list[dict],
        *,
        query: bool,
        max_tokens: int,
        deadline: float,
        json_mode: bool = False,
        allow_clarify: bool = False,
        json_schema: dict | None = None,
        draft_history: list[str] | None = None,
    ) -> ModelReply: ...


phase_draft_scope: ContextVar[Callable[[bool, str | None], Awaitable[list[str]]] | None] = (
    ContextVar("agentserve_phase_draft_scope", default=None)
)


class VLLMClient:
    """The caller owns the reusable HTTP client and its lifecycle. No implicit retries."""

    def __init__(
        self, client: httpx.AsyncClient, model: str, *,
        draft_all_phases: bool = False, model_tool_choice: Literal["auto", "named"] = "auto",
    ):
        self.client = client
        self.model = model
        self.draft_all_phases = draft_all_phases
        self.model_tool_choice = model_tool_choice

    async def generate(
        self,
        messages: list[dict],
        *,
        query: bool,
        max_tokens: int,
        deadline: float,
        json_mode: bool = False,
        allow_clarify: bool = False,
        json_schema: dict | None = None,
        draft_history: list[str] | None = None,
    ) -> ModelReply:
        allow_clarify = allow_clarify and self.model_tool_choice == "auto"
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": SQL_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "execute_sql"}}
            if query
            else "none",
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload.pop("tools")
            payload.pop("tool_choice")
            payload["response_format"] = {"type": "json_object"}
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "memory", "strict": True, "schema": json_schema},
                }
        elif query and allow_clarify:
            payload["tool_choice"] = "auto"
        if draft_history is None and self.draft_all_phases:
            provider = phase_draft_scope.get()
            draft_history = (
                await provider(query, json_schema.get("title") if json_schema else None)
                if provider and (not json_mode or json_schema) else []
            )
        if draft_history is not None:
            payload["vllm_xargs"] = {
                "agentserve_v5_sam": json.dumps(draft_history, ensure_ascii=False)
            }
        try:
            async with asyncio.timeout_at(deadline):
                response = await self.client.post(
                    "chat/completions",
                    json=payload,
                    timeout=max(0.001, deadline - time.monotonic()),
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as error:
            raise ModelRequestError(
                "MODEL_CONNECTION_ERROR", str(error), retryable=True, outcome_unknown=False
            ) from error
        except (httpx.TimeoutException, TimeoutError) as error:
            raise ModelRequestError(
                "MODEL_TIMEOUT",
                "Model request exceeded its deadline",
                retryable=True,
                outcome_unknown=True,
            ) from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            raise ModelRequestError(
                "MODEL_HTTP_ERROR",
                f"Model HTTP status {status}",
                retryable=status in {429, 502, 503, 504},
                outcome_unknown=status in {502, 503, 504},
            ) from error
        except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as error:
            raise ModelRequestError(
                "MODEL_TRANSPORT_ERROR", str(error), retryable=True, outcome_unknown=True
            ) from error
        except httpx.HTTPError as error:
            raise AgentError("MODEL_TRANSPORT_ERROR", str(error)) from error

        try:
            completion = Completion.model_validate_json(response.content)
            choice = completion.choices[0]
            if choice.finish_reason == "length":
                raise AgentError("MODEL_OUTPUT_LIMIT", "Model output reached the token limit")
            message = choice.message
            sql = None
            if query:
                if (
                    allow_clarify
                    and not message.tool_calls
                    and (message.content or "").startswith("CLARIFY:")
                ):
                    raise AgentError("MEMORY_CONTEXT_REQUIRED", message.content[8:].strip())
                if not message.tool_calls or len(message.tool_calls) != 1:
                    raise AgentError(
                        "MODEL_PROTOCOL_ERROR", "Expected exactly one execute_sql call"
                    )
                sql = SQLArguments.model_validate_json(message.tool_calls[0].function.arguments).sql
            elif message.tool_calls or not message.content or not message.content.strip():
                raise AgentError("MODEL_PROTOCOL_ERROR", "Expected a nonempty answer without tools")
        except ValidationError as error:
            raise AgentError("MODEL_PROTOCOL_ERROR", str(error)[:800]) from error
        return ModelReply(
            message.model_dump(exclude_none=True),
            sql,
            completion.id,
            completion.usage,
            choice.finish_reason,
            request_metrics(completion.metrics, draft_history is not None),
        )
