"""Resolve versioned, approved order metrics and compile only fixed SQL expressions."""

import time
from datetime import date
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentserve.agents.order_sql import COLUMNS, METRICS, TABLES
from agentserve.domain.models import AgentError, MetricInput
from agentserve.knowledge.snapshot import assemble, digest, normalized
from agentserve.storage.metric import MetricRepository

FORMULAS = {
    "revenue": (*METRICS["订单项营收合计"][1:], "分"),
    "order_count": (*METRICS["订单笔数"][1:], "笔"),
}
CITATION_FIELDS = (
    "chunk_id", "database_id", "clause_id", "path", "title", "anchor", "text",
    "approval_ref", "source_sha256", "content_sha256", "purpose", "version",
    "effective_from", "effective_to",
)


class MetricRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, str_strip_whitespace=True)
    metric_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    aliases: tuple[str, ...] = Field(min_length=1)
    formula: Literal["revenue", "order_count"]
    statuses: tuple[Literal["completed", "pending", "cancelled"], ...] = Field(min_length=1)
    unit: Literal["分", "笔"]
    groups: tuple[Literal["city", "month"], ...]
    source: str
    source_binding_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class MetricRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format: Literal[1]
    rules: list[MetricRule] = Field(min_length=1)


def citation_for(chunk):
    return {field: chunk[field] for field in CITATION_FIELDS}


def alias_key(name):
    return " ".join(normalized(name).split())


class MetricCatalog:
    def __init__(self, registry_path, *, purpose, database_ids):
        registry = MetricRegistration.model_validate_json(registry_path.read_bytes())
        snapshot = assemble(registry_path.with_name("manifest.json"))
        if snapshot["purpose"] != purpose:
            raise ValueError("Metric source purpose does not match configuration")
        chunks = {c["chunk_id"]: c for c in snapshot["chunks"]}
        self.aliases = {}
        self.versions = {}
        bindings = []
        for rule in registry.rules:
            chunk = chunks.get(rule.source)
            if not chunk or not chunk["approval_ref"] or chunk["conflict_group"]:
                raise ValueError(f"Missing or ineligible metric source {rule.source}")
            citation = citation_for(chunk)
            database_id = citation["database_id"]
            rule_sha = digest({
                "rule": rule.model_dump(mode="json", exclude={"source_binding_sha256"}),
                "citation": citation,
            })
            if database_id not in database_ids or rule_sha != rule.source_binding_sha256:
                raise ValueError(f"Metric source binding changed for {rule.source}")
            if (
                rule.unit != FORMULAS[rule.formula][2]
                or len(set(rule.statuses)) != len(rule.statuses)
                or len(set(rule.groups)) != len(rule.groups)
            ):
                raise ValueError(f"Invalid unit, statuses or groups for {rule.metric_id}")
            for alias in (*rule.aliases, rule.metric_id):
                name = alias_key(alias)
                key = (database_id, name)
                if not name or self.aliases.setdefault(key, rule.metric_id) != rule.metric_id:
                    raise ValueError(f"Ambiguous metric alias {alias!r} for {database_id}")
            binding = {"rule": rule.model_dump(mode="json"), "citation": citation, "rule_sha256": rule_sha}
            bindings.append(binding)
            self.versions.setdefault((database_id, rule.metric_id), []).append(binding)
        for key, versions in self.versions.items():
            ordered = sorted(versions, key=lambda b: b["citation"]["effective_from"])
            if len({b["citation"]["version"] for b in versions}) != len(versions):
                raise ValueError(f"Duplicate approved metric version for {key}")
            for previous, current in pairwise(ordered):
                previous_end = previous["citation"]["effective_to"]
                if previous_end is None or current["citation"]["effective_from"] < previous_end:
                    raise ValueError(f"Overlapping approved metric versions for {key}")
        self.identity = digest({"format": registry.format, "purpose": purpose, "bindings": bindings})

    def bind(self, database_id: str, request: MetricInput):
        metric_id = self.aliases.get((database_id, alias_key(request.name)))
        if metric_id is None:
            raise AgentError("METRIC_UNKNOWN", "该数据库未登记此指标。请提供已批准的指标名。")
        for binding in self.versions[database_id, metric_id]:
            source = binding["citation"]
            start, end = source["effective_from"], source["effective_to"]
            if start <= request.start and (end is None or request.end <= end):
                if request.group_by is not None and request.group_by not in binding["rule"]["groups"]:
                    raise AgentError("METRIC_GROUP_UNSUPPORTED", "该版本未批准此分组方式。")
                return {**binding, "registry_sha256": self.identity}
        raise AgentError(
            "METRIC_WINDOW_UNCOVERED", "请提供完全落在同一已批准口径版本内的统计窗口。"
        )


def compile_metric_sql(rule: MetricRule, request: MetricInput) -> tuple[str, tuple[str, ...]]:
    aggregate, alias, _ = FORMULAS[rule.formula]
    tables = TABLES["orders"]
    if rule.formula == "revenue":
        tables += " JOIN order_items i ON i.order_id = o.order_id"
    projection = f"{aggregate} AS {alias}"
    group = COLUMNS[request.group_by] if request.group_by else None
    if group:
        projection = f"{group} AS {request.group_by}, {projection}"
    placeholders = ", ".join("?" for _ in rule.statuses)
    sql = (
        f"SELECT {projection} FROM {tables}"
        f" WHERE {COLUMNS['date']} >= ? AND {COLUMNS['date']} < ?"
        f" AND {COLUMNS['status']} IN ({placeholders})"
    )
    parameters = (request.start, request.end, *rule.statuses)
    if request.city is not None:
        sql += f" AND {COLUMNS['city']} = ?"
        parameters += (request.city,)
    if group:
        sql += f" GROUP BY {group} ORDER BY {group}"
    return sql, parameters


class MetricService:
    def __init__(self, catalog, tool):
        self.catalog, self.tool = catalog, tool

    async def run(self, task, report, *, deadline, emit, journal):
        started = time.monotonic()
        repository = MetricRepository(journal.store.tasks) if journal else None
        binding = await repository.get(report.task_id) if repository else None
        boundary = {"kind": "metric", "step_no": -1}
        if binding is None:
            if repository:
                await journal.boundary("before_execute", boundary)
            binding = self.catalog.bind(task.database_id, task.metric)
            if repository:
                await repository.save(report.task_id, journal.version, binding)
                await journal.boundary("after_commit", boundary)
            await emit("METRIC_BOUND", {
                "rule_sha256": binding["rule_sha256"], "seconds": time.monotonic() - started,
            })
        else:
            await emit("METRIC_BINDING_REUSED", {"rule_sha256": binding["rule_sha256"]})
        rule = MetricRule.model_construct(**binding["rule"])
        sql, parameters = compile_metric_sql(rule, task.metric)
        report.sql = sql
        report.metric = {**binding, "query": task.metric.model_dump(exclude_none=True)}
        if journal:
            result = await journal.execute_sql(
                self.tool, task.database_id, sql, deadline=deadline, parameters=parameters
            )
        else:
            await emit("SQL_STARTED", {"sql": sql, "parameters": parameters})
            result = await self.tool.execute(task.database_id, sql, deadline=deadline, parameters=parameters)
            await emit("SQL_COMPLETED", {"result": result.to_dict()})
        report.query_result = result
        if result.status != "ok":
            raise AgentError(result.status, result.error)
        value = f"{len(result.rows)} 行"
        if task.metric.group_by is None and result.rows:
            cell = result.rows[0][0]
            value = "无匹配金额（NULL）" if cell is None else f"{cell} {rule.unit}"
        if result.truncated:
            value += "，结果已截断"
        citation = binding["citation"]
        report.answer = (
            f"{rule.aliases[0]}（{citation['version']} 版，单位 {rule.unit}）"
            f" [{task.metric.start}, {task.metric.end}) · {value}。"
        )
        report.answer_source = "metric"
