"""Validate external task input once; keep internal execution records explicit."""

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MetricInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    start: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    end: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    city: str | None = Field(default=None, min_length=1, max_length=100)
    group_by: Literal["city", "month"] | None = None

    @model_validator(mode="after")
    def validate_window(self):
        if date.fromisoformat(self.start) >= date.fromisoformat(self.end):
            raise ValueError("Metric start must precede end")
        if self.city is not None and "\x00" in self.city:
            raise ValueError("Metric city cannot contain NUL")
        return self


class AnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    start: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-01$")
    end: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-01$")
    city: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_window(self):
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
        months = (end.year - start.year) * 12 + end.month - start.month
        if not 1 <= months <= 12:
            raise ValueError("Analysis requires 1 to 12 complete calendar months")
        if self.city is not None and ("\x00" in self.city or not self.city.isprintable()):
            raise ValueError("Analysis city must be printable text")
        return self


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    database_id: str = Field(min_length=1, max_length=100)
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    mode: Literal["sql", "knowledge", "metric", "analysis"] | None = None
    answer_mode: Literal["explain"] | None = None
    as_of: str | None = Field(default=None, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    clause_id: str | None = Field(default=None, max_length=100, pattern=r"^[a-z0-9_-]+$")
    metric: MetricInput | None = None
    analysis: AnalysisInput | None = None
    save_to_history: bool | None = None

    @field_validator("as_of")
    @classmethod
    def validate_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_mode(self):
        if self.mode == "sql":
            self.mode = None
        if self.mode is not None and self.session_id is not None:
            raise ValueError("Only SQL tasks support sessions")
        if self.mode is not None and self.answer_mode is not None:
            raise ValueError("Only SQL tasks support answer_mode")
        if (self.metric is not None) != (self.mode == "metric"):
            raise ValueError("Metric mode and metric input must be used together")
        if (self.analysis is not None) != (self.mode == "analysis"):
            raise ValueError("Analysis mode and analysis input must be used together")
        if self.mode in {"metric", "analysis"}:
            if self.question is not None:
                raise ValueError("Structured tasks do not accept a question")
        elif self.question is None:
            raise ValueError("SQL and knowledge tasks require a question")
        if self.mode != "knowledge" and (self.as_of is not None or self.clause_id is not None):
            raise ValueError("as_of and clause_id require knowledge mode")
        if self.save_to_history is not None and self.mode not in {"metric", "analysis"}:
            raise ValueError("History preference requires metric or analysis mode")
        return self


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    database_id: str = Field(min_length=1, max_length=100)


@dataclass
class QueryResult:
    status: str = "ok"
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    limit_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelReply:
    message: dict
    sql: str | None
    request_id: str
    usage: dict | None
    finish_reason: str
    metrics: dict | None = None


class AgentError(Exception):
    """Expected boundary failure; only the task owner converts it to task failure."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


class ModelRequestError(AgentError):
    def __init__(self, code: str, detail: str, *, retryable: bool, outcome_unknown: bool):
        super().__init__(code, detail)
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class TaskStateError(AgentError):
    """Execution lost ownership, for example after a persistent cancellation."""


@dataclass
class RunReport:
    task_id: str
    task: dict
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    sql: str | None = None
    query_result: QueryResult | None = None
    answer: str | None = None
    error: dict | None = None
    model_calls: int = 0
    model_retries: int = 0
    sql_repairs: int = 0
    format_repairs: int = 0
    elapsed_seconds: float = 0
    evaluation_status: str = "not_evaluated"
    events: list[dict] = field(default_factory=list)
    answer_source: Literal[
        "model", "bounded_summary", "tool_values", "knowledge", "metric", "analysis", "workbench"
    ] | None = None
    memory_calls: int = 0
    memory_context: dict | None = None
    memory_draft: list[dict] | None = field(default=None, repr=False)
    knowledge: dict | None = None
    metric: dict | None = None
    analysis: dict | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("memory_draft")
        if self.knowledge is None:
            result.pop("knowledge")
        if self.metric is None:
            result.pop("metric")
        if self.analysis is None:
            result.pop("analysis")
        return result
