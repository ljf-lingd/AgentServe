from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["new", "refine", "resume", "clarify"]
    search_query: str = Field(min_length=1, max_length=1600)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    topic: str = ""
    depth: int = Field(default=3, ge=1, le=20)
    target_id: str | None = None
    changes: list[str] = Field(default_factory=list, max_length=12)
    clarification: str | None = None


class QueryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: str = Field(min_length=1, max_length=1600)
    conditions: list[str] = Field(default_factory=list, max_length=16)
    date_range: str | None = None
    metric: str | None = None
    group_by: list[str] = Field(default_factory=list, max_length=8)
    unit: str | None = None


class EntryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    keywords: list[str] = Field(default_factory=list, max_length=16)
    topic: str = ""
    event_time: str | None = None
    frame: QueryFrame | None = None
    state: dict | None = None


class EncodedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[EntryDraft] = Field(max_length=4)


class FactEntry(EntryDraft):
    frame: None = None


class EpisodeEntry(EntryDraft):
    frame: QueryFrame


class FactMemory(EncodedMemory):
    entries: list[FactEntry] = Field(min_length=1, max_length=4)


class EpisodeMemory(EncodedMemory):
    entries: list[EpisodeEntry] = Field(min_length=1, max_length=1)


@dataclass
class ContextPack:
    text: str
    source_ids: list[str]
    cutoff_turn: int
    revision: int
    policy_version: str
    token_count: int
    plan: dict | None = None
    selected: list[dict] = field(default_factory=list)
    planning_messages: list[dict] = field(default_factory=list)
