"""Bind both metrics atomically and read their monthly aggregates in one snapshot."""

import time
from datetime import date

from agentserve.analytics.artifacts import ARTIFACT_TYPES
from agentserve.domain.models import AgentError, MetricInput
from agentserve.knowledge.metrics import MetricRule, compile_metric_sql
from agentserve.storage.metric import MetricRepository

METRIC_IDS = ("settled_revenue", "completed_orders")


def monthly_rows(window, query_rows):
    values = {month: (revenue, count) for month, revenue, count in query_rows}
    start, end = date.fromisoformat(window.start), date.fromisoformat(window.end)
    rows = []
    for month_index in range(start.year * 12 + start.month - 1, end.year * 12 + end.month - 1):
        year, month = divmod(month_index, 12)
        label = f"{year:04d}-{month + 1:02d}"
        revenue, count = values.get(label, (None, None))
        rows.append({"month": label, "revenue_cents": revenue, "order_count": count or 0})
    return rows


class AnalysisService:
    def __init__(self, catalog, tool):
        self.catalog, self.tool = catalog, tool

    async def run(self, task, report, *, deadline, emit, journal):
        if task.database_id != "business_orders_v1":
            raise AgentError("ANALYSIS_DATABASE_UNSUPPORTED", "经营简报仅支持业务订单库。")
        started = time.monotonic()
        window = task.analysis
        requests = [
            MetricInput.model_construct(
                name=name, start=window.start, end=window.end, city=window.city, group_by="month"
            )
            for name in METRIC_IDS
        ]
        repository = MetricRepository(journal.store.tasks) if journal else None
        binding = await repository.get(report.task_id) if repository else None
        boundary = {"kind": "analysis", "step_no": -1}
        if binding is None:
            if repository:
                await journal.boundary("before_execute", boundary)
            binding = {
                "registry_sha256": self.catalog.identity,
                "metrics": [self.catalog.bind(task.database_id, request) for request in requests],
            }
            if repository:
                await repository.save(report.task_id, journal.version, binding, analysis=True)
                await journal.boundary("after_commit", boundary)
            await emit("ANALYSIS_BOUND", {"seconds": time.monotonic() - started})
        else:
            await emit("ANALYSIS_BINDING_REUSED", {"registry_sha256": binding["registry_sha256"]})

        (revenue_sql, revenue_params), (orders_sql, orders_params) = [
            compile_metric_sql(MetricRule.model_construct(**metric["rule"]), request)
            for metric, request in zip(binding["metrics"], requests, strict=True)
        ]
        sql = (
            f"WITH revenue AS ({revenue_sql}), orders_by_month AS ({orders_sql}), "
            "months AS (SELECT month FROM revenue UNION SELECT month FROM orders_by_month) "
            "SELECT m.month, r.revenue_cents, o.order_count FROM months m "
            "LEFT JOIN revenue r ON r.month = m.month "
            "LEFT JOIN orders_by_month o ON o.month = m.month ORDER BY m.month"
        )
        parameters = revenue_params + orders_params
        report.sql = sql
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
        if result.truncated:
            raise AgentError("ANALYSIS_RESULT_TRUNCATED", "查询结果截断，无法生成完整经营简报。")

        months = monthly_rows(window, result.rows)
        amounts = [row["revenue_cents"] for row in months if row["revenue_cents"] is not None]
        totals = {
            "revenue_cents": sum(amounts) if amounts else None,
            "order_count": sum(row["order_count"] for row in months),
        }
        report.analysis = {
            "window": window.model_dump(exclude_none=True),
            "totals": totals,
            "months": months,
            "units": {"revenue_cents": "分", "order_count": "笔"},
            **binding,
            "artifact_version": 1,
            "artifacts": list(ARTIFACT_TYPES),
        }
        report.answer = f"已生成 {len(months)} 个月的经营简报。"
        report.answer_source = "analysis"
