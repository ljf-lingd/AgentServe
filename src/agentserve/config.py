"""Load application configuration at the boundary; relative paths use the config directory."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_APP_CONFIG = Path("configs/app.v5.json")


class WorkbenchDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    first_date: date
    last_date: date
    currency: str
    modes: tuple[Literal["sql", "knowledge", "metric", "analysis"], ...]
    data_description: str
    rule_description: str
    query_examples: tuple[str, ...]
    rule_examples: tuple[str, ...]

    @model_validator(mode="after")
    def validate_catalog(self):
        if self.last_date < self.first_date or "sql" not in self.modes:
            raise ValueError("Workbench datasets require a valid date range and SQL mode")
        return self


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    schema_path: Path
    context: str = ""
    workbench: WorkbenchDataset | None = Field(default=None, exclude=True)


class MemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal[
        "window", "simplemem", "sql_episode", "no_binding", "no_updates", "state", "state_on_demand"
    ] = "window"
    resources_url: str = "http://127.0.0.1:8002"
    resources_identity: str = ""
    recent_turns: int = Field(default=2, ge=1, le=5)
    context_tokens: int = Field(default=2048, ge=256, le=4096)
    model_context_tokens: int = Field(default=8192, ge=2048)
    plan_max_tokens: int = Field(default=512, ge=128, le=1024)
    write_max_tokens: int = Field(default=1024, ge=256, le=2048)
    max_model_calls: int = Field(default=2, ge=2, le=2)


class KnowledgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_path: Path
    strategy: Literal["bm25", "dense", "rrf", "static", "none"] = "bm25"
    purpose: Literal["product", "test", "benchmark"] = "product"


class MetricSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_path: Path
    purpose: Literal["product", "test"] = "product"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "agent-qwen"
    model_tool_choice: Literal["auto", "named"] = Field(
        default="auto", exclude_if=lambda value: value == "auto"
    )
    session_sam: bool = False
    session_sam_all_phases: bool = False
    databases: dict[str, DatabaseConfig] = Field(min_length=1)
    task_timeout_seconds: float = Field(default=90, gt=0, le=600)
    sql_timeout_seconds: float = Field(default=1, gt=0, le=10)
    sql_workers: int = Field(default=2, ge=1, le=16)
    row_limit: int = Field(default=100, ge=1, le=1000)
    result_byte_limit: int = Field(default=4096, ge=240, le=16384)
    max_sql_repairs: int = Field(default=2, ge=0, le=2)
    max_format_repairs: int = Field(default=1, ge=0, le=1)
    max_model_calls: int = Field(default=6, ge=2, le=6)
    sql_max_tokens: int = Field(default=256, ge=1, le=512)
    answer_max_tokens: int = Field(default=192, ge=1, le=512)
    answer_style: Literal[
        "baseline", "concise_v1", "bounded_v1", "evidence_v21", "evidence_v221", "evidence_direct_v5"
    ] = "baseline"
    sql_prompt_version: Literal["baseline", "projection_v1", "evidence_v21", "evidence_v22"] = (
        "baseline"
    )
    sql_function_profile: Literal["baseline", "with_replace_v1"] = "baseline"
    state_database: Path = Path("../state/tasks.sqlite")
    task_concurrency: int = Field(default=2, ge=1, le=16)
    queue_limit: int = Field(default=100, ge=1, le=10000)
    shutdown_grace_seconds: float = Field(default=5, ge=0, le=60)
    llm_concurrency: int = Field(default=2, ge=1, le=16)
    model_request_timeout_seconds: float = Field(default=30, gt=0, le=600)
    max_model_retries: int = Field(default=1, ge=0, le=2)
    model_retry_backoff_seconds: float = Field(default=0.1, ge=0, le=2)
    memory: MemorySettings | None = None
    knowledge: KnowledgeSettings | None = None
    metric: MetricSettings | None = None
    analysis_enabled: bool = False

    @field_validator("base_url")
    @classmethod
    def http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) endpoint")
        return value.rstrip("/")

    @classmethod
    def load(cls, path: Path) -> "Settings":
        settings = cls.model_validate_json(path.read_text(encoding="utf-8"))
        for database in settings.databases.values():
            database.path = (path.parent / database.path).resolve(strict=True)
            database.schema_path = (path.parent / database.schema_path).resolve(strict=True)
        settings.state_database = (path.parent / settings.state_database).resolve()
        if settings.knowledge:
            settings.knowledge.snapshot_path = (
                path.parent / settings.knowledge.snapshot_path
            ).resolve(strict=True)
        if settings.metric:
            settings.metric.registry_path = (
                path.parent / settings.metric.registry_path
            ).resolve(strict=True)
        return settings
