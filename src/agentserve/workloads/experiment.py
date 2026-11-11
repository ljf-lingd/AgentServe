"""Validated experiment inputs and frozen question lists, separate from gold evaluation."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Name = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,79}$")]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    database_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_db_id: str | None = None


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: Path
    gold: Path
    spider: bool = False
    task_count: int | None = Field(default=None, ge=1)
    task_id_prefix: str = "demo-load-"

    def jobs(self) -> list[dict]:
        # Question files are one external boundary; gold fields are forbidden here.
        rows = [
            q.model_dump(exclude_none=True)
            for q in TypeAdapter(Annotated[list[Question], Field(min_length=1)]).validate_json(
                self.questions.read_bytes()
            )
        ]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Question IDs must be unique")
        if self.task_count is None:
            return rows
        return [
            {
                **rows[i % len(rows)],
                "id": f"{self.task_id_prefix}{i:03d}",
                "source_id": rows[i % len(rows)]["id"],
            }
            for i in range(self.task_count)
        ]


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    kind: Literal["quality", "prewarm", "load", "tool_wait"]
    task_concurrency: int = Field(ge=1, le=16)
    llm_concurrency: int = Field(default=2, ge=1, le=16)
    sql_workers: int = Field(default=2, ge=1, le=16)
    sql_wait_seconds: float = Field(default=0, ge=0)
    warmup_ids: list[str] = Field(default_factory=list)


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    application_config: Path
    database_manifest: Path
    dataset_manifest: Path
    output_root: Path
    model_max_length: Literal[2048, 8192] = 8192
    datasets: dict[Name, DatasetSpec] = Field(min_length=1)
    conditions: dict[Name, Condition] = Field(min_length=1)
    run_order: list[str] = Field(min_length=1)
    smoke_order: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self):
        if any(c.dataset not in self.datasets for c in self.conditions.values()):
            raise ValueError("Condition references an unknown dataset")
        if any(key not in self.conditions for key in self.run_order + self.smoke_order):
            raise ValueError("Run order references an unknown condition")
        return self

    @classmethod
    def load(cls, path: Path):
        plan = cls.model_validate_json(path.read_text(encoding="utf-8"))
        for name in ["application_config", "database_manifest", "dataset_manifest"]:
            setattr(plan, name, (path.parent / getattr(plan, name)).resolve(strict=True))
        plan.output_root = (path.parent / plan.output_root).resolve()
        for dataset in plan.datasets.values():
            dataset.questions = (path.parent / dataset.questions).resolve(strict=True)
            dataset.gold = (path.parent / dataset.gold).resolve(strict=True)
        return plan

    def prepare(self, *, smoke_only=False) -> list[dict]:
        order = self.smoke_order if smoke_only else self.run_order
        jobs = {
            key: self.datasets[key].jobs() for key in {self.conditions[c].dataset for c in order}
        }
        cases = []
        for index, key in enumerate(order, 1):
            condition = self.conditions[key]
            selected = jobs[condition.dataset]
            lookup = {row["id"]: row for row in selected}
            if any(task_id not in lookup for task_id in condition.warmup_ids):
                raise ValueError(f"Unknown warmup task in condition {key}")
            cases.append(
                {
                    "case": f"{index:02d}-{key}",
                    "condition_id": key,
                    "condition": condition.model_dump(),
                    "jobs": selected,
                    "warmup_jobs": [lookup[task_id] for task_id in condition.warmup_ids],
                }
            )
        return cases

    def input_hashes(self) -> dict[str, str]:
        paths = {self.application_config, self.database_manifest, self.dataset_manifest}
        paths.update(p for d in self.datasets.values() for p in [d.questions, d.gold])
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
