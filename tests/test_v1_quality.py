import hashlib
import json
import sqlite3
import time
from contextlib import closing

import httpx
import pytest
from conftest import completion
from pydantic import TypeAdapter, ValidationError
from test_agent import run_script
from test_recovery import backend, crash_once, resume

from agentserve.agents.answer_policy import render_answer
from agentserve.agents.sql_prompt import CONCISE_ANSWER_PROMPT
from agentserve.analysis.quality import select_profile
from agentserve.application.execution import execution_snapshot
from agentserve.domain.models import QueryResult
from agentserve.evaluation.answers import AnswerReview, summarize_answers
from agentserve.evaluation.sql import SQLEvaluator
from agentserve.tools.sqlite import SQLTool
from agentserve.workloads.business import prepare


def test_business_fixture_is_frozen_and_distinguishes_historical_prices(tmp_path):
    manifest = prepare(tmp_path)
    assert prepare(tmp_path) == manifest
    questions = json.loads(
        (tmp_path / "data/processed/business_orders_v1/questions.json").read_text()
    )
    assert len(questions) == len({q["id"] for q in questions}) == 40
    assert all(set(q) == {"id", "database_id", "question"} for q in questions)
    path = tmp_path / manifest["database"]["path"]
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone() == (1000,)
        assert conn.execute("SELECT COUNT(*) FROM customers WHERE name='Alex'").fetchone() == (2,)
        assert conn.execute(
            "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS(SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id)"
        ).fetchone() == (10,)
        correct = conn.execute(
            "SELECT SUM(quantity*unit_price_cents) FROM order_items WHERE order_id=1"
        ).fetchone()
        wrong = conn.execute(
            "SELECT SUM(i.quantity*p.price_cents) FROM order_items i JOIN products p ON p.product_id=i.product_id WHERE i.order_id=1"
        ).fetchone()
        assert correct != wrong


async def test_replace_profile_adds_only_scalar_function_and_still_blocks_writes(settings):
    path = settings.databases["demo_orders"].path
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for profile, expected in [("baseline", "POLICY_REJECTED"), ("with_replace_v1", "ok")]:
        tool = SQLTool({"db": path}, function_profile=profile)
        try:
            result = await tool.execute(
                "db", "SELECT REPLACE('a-b','-','')", deadline=time.monotonic() + 2
            )
            assert result.status == expected
            if profile != "baseline":
                assert result.rows == [["ab"]]
            write = await tool.execute(
                "db", "REPLACE INTO customers VALUES(1,'changed')", deadline=time.monotonic() + 2
            )
            assert write.status == "POLICY_REJECTED"
        finally:
            tool.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    evaluator = SQLEvaluator({"db": path})
    try:
        assert (
            await evaluator.evaluate(
                "db", "SELECT REPLACE('a-b','-','')", "SELECT 'ab'", ordered=True
            )
        )["status"] == "correct"
    finally:
        evaluator.close()


async def test_concise_style_changes_only_answer_input_without_raising_budget(settings):
    _, baseline = await run_script(settings, [completion(), completion(answer="有2位客户。")])
    settings.answer_style = "concise_v1"
    report, requests = await run_script(settings, [completion(), completion(answer="有2位客户。")])
    assert requests[0] == baseline[0]
    assert requests[1]["messages"][-1] == {"role": "user", "content": CONCISE_ANSWER_PROMPT}
    assert requests[1]["max_tokens"] == baseline[1]["max_tokens"] == 192
    assert report.model_calls == 2


async def test_concise_recovery_reuses_committed_sql(settings):
    settings.answer_style = "concise_v1"
    calls = []
    task_id, before = await crash_once(
        settings, backend(calls), stage="after_commit", step_no=1, kind="sql"
    )
    task, history, _ = await resume(settings, backend(calls), task_id)
    assert task["status"] == "COMPLETED" and len(calls) == 2
    assert len([s for s in history["steps"] if s["kind"] == "sql"]) == 1
    assert task["deadline_at"] == before["deadline_at"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer_style", "concise_v1"),
        ("sql_prompt_version", "projection_v1"),
        ("sql_function_profile", "with_replace_v1"),
    ],
)
def test_quality_profiles_have_distinct_persisted_identity(settings, field, value):
    old = execution_snapshot(settings)["execution_identity"]
    setattr(settings, field, value)
    assert execution_snapshot(settings)["execution_identity"] != old


def test_missing_review_is_unknown_and_truncated_answer_is_not_full_success():
    records = [
        {
            "task_id": str(i),
            "status": "COMPLETED",
            "task": {"report": {"query_result": {"truncated": i == 1}}},
        }
        for i in range(3)
    ]
    labels = [{"task_id": str(i), "status": "correct" if i < 2 else "incorrect"} for i in range(3)]
    assert summarize_answers(records, labels, [])["completed_unreviewed"] == 3
    reviews = [
        AnswerReview(
            task_id=str(i),
            reviewer="unit",
            source="codex_assisted",
            grounded=True,
            adequate=True,
            truncation_disclosed=True,
            evidence="checked result",
        )
        for i in range(3)
    ]
    result = summarize_answers(records, labels, reviews)
    assert result["fully_successful"] == 1
    assert result["faithful_bounded_responses"] == 2 and result["honest_partial_responses"] == 1
    assert result["review_sources"] == ["codex_assisted"]
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_answers(records, labels, reviews + reviews[:1])


def test_selection_rejects_sql_regression_and_keeps_simpler_tie():
    profiles = ["baseline", "concise", "replace", "projection"]
    runs = []
    for name in profiles:
        for dataset in ["demo", "smoke", "development", "business"]:
            correct = 8 if name == "projection" and dataset == "development" else 10
            runs.append(
                {
                    "profile": name,
                    "dataset": dataset,
                    "quality": {"correct": correct},
                    "completed_correct": 9 if name == "baseline" else 10,
                }
            )
    assert select_profile(runs, profiles) == "concise"


def test_order_context_does_not_modify_other_database_prompts(settings, monkeypatch):
    import importlib

    from conftest import ROOT

    monkeypatch.syspath_prepend(str(ROOT / "scripts/experiments"))
    runner = importlib.import_module("run_v1_quality")
    original = settings.databases["demo_orders"].model_copy(deep=True)
    settings.databases["spider_other"] = original.model_copy(
        update={"context": "other schema context"}
    )
    improved = runner.profile_settings(settings, "order_context")
    assert improved.databases["spider_other"].context == "other schema context"
    assert improved.databases["demo_orders"].context != original.context
    assert settings.databases["demo_orders"].context == original.context
    assert improved.sql_prompt_version == "baseline" and improved.answer_style == "concise_v1"


def test_bounded_summary_does_not_claim_a_total_and_preserves_normal_model_answers():
    result = QueryResult(rows=[["Alice"]], truncated=True)
    text, source = render_answer("实际返回120行", result, "bounded_v1")
    assert "120" not in text and "1行" in text and "总行数未知" in text
    assert source == "bounded_summary"
    assert render_answer("无匹配。", QueryResult(), "bounded_v1") == ("无匹配。", "model")


async def test_bounded_summary_replays_original_model_step_and_keeps_audit(settings):
    settings.answer_style = "bounded_v1"
    settings.row_limit = 1
    calls = []

    def replies(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json=completion("SELECT name FROM customers")
            if body["tool_choice"] != "none"
            else completion(answer="实际返回120行"),
        )

    transport = httpx.MockTransport(replies)
    task_id, _ = await crash_once(
        settings, transport, stage="after_commit", step_no=2, kind="model"
    )
    task, history, events = await resume(settings, transport, task_id)
    assert len(calls) == 2 and task["status"] == "COMPLETED"
    assert task["report"]["answer_source"] == "bounded_summary"
    assert "当前仅返回1行" in task["report"]["answer"]
    final_step = history["steps"][-1]
    assert final_step["output"]["reply"]["message"]["content"] != task["report"]["answer"]
    assert any(e["type"] == "ANSWER_POLICY_APPLIED" for e in events)


@pytest.mark.parametrize("change", [{"grounded": "yes"}, {"source": "unknown"}])
def test_review_json_boundary_remains_strict_with_standard_library_dto(change):
    review = {
        "task_id": "task",
        "reviewer": "unit",
        "source": "codex_assisted",
        "grounded": True,
        "adequate": True,
        "truncation_disclosed": False,
        "evidence": "checked",
        **change,
    }
    with pytest.raises(ValidationError):
        TypeAdapter(list[AnswerReview]).validate_json(json.dumps([review]))
