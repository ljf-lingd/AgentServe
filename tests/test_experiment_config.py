import json
from collections import Counter

import pytest
from conftest import ROOT
from pydantic import ValidationError

from agentserve.analysis.report import aggregate, group_conditions
from agentserve.workloads.experiment import Condition, DatasetSpec, ExperimentPlan


def test_default_plan_preserves_g5_job_sequence_and_warmup_boundaries():
    plan = ExperimentPlan.load(ROOT / "configs/g5.experiment.json")
    cases = plan.prepare()
    assert len(cases) == 28
    assert sum(len(c["jobs"]) for c in cases if c["condition"]["kind"] != "prewarm") == 796
    assert (
        sum(
            len(c["warmup_jobs"]) + (len(c["jobs"]) if c["condition"]["kind"] == "prewarm" else 0)
            for c in cases
        )
        == 64
    )
    loads = [c for c in cases if c["condition"]["kind"] == "load"]
    assert [c["condition"]["task_concurrency"] for c in loads] == [
        1,
        4,
        8,
        16,
        16,
        8,
        4,
        1,
        4,
        1,
        16,
        8,
    ]
    assert Counter(c["condition_id"] for c in loads) == {f"load-c{c}": 3 for c in [1, 4, 8, 16]}
    demo = json.loads((ROOT / "data/processed/demo_orders/questions.json").read_text("utf-8"))
    for case in loads:
        assert [q["source_id"] for q in case["jobs"]] == [demo[i % 16]["id"] for i in range(24)]
        assert [q["id"] for q in case["warmup_jobs"]] == ["demo-load-000", "demo-load-001"]
    smoke = plan.prepare(smoke_only=True)
    assert len(smoke) == 1 and len(smoke[0]["jobs"]) == 20 and not smoke[0]["warmup_jobs"]


def test_supported_new_condition_and_task_count_only_require_config():
    plan = ExperimentPlan.load(ROOT / "configs/g5.experiment.json")
    plan.datasets["demo-load"].task_count = 7
    plan.conditions = {
        "load-c2": Condition(
            dataset="demo-load",
            kind="load",
            task_concurrency=2,
            llm_concurrency=1,
            sql_workers=1,
            warmup_ids=["demo-load-002"],
        )
    }
    plan.run_order = ["load-c2", "load-c2"]
    cases = plan.prepare()
    assert len(cases) == 2 and all(len(c["jobs"]) == 7 for c in cases)
    assert all(c["condition"]["task_concurrency"] == 2 for c in cases)
    assert cases[0]["warmup_jobs"][0]["source_id"] == "completed_count"


def test_invalid_condition_and_warmup_fail_before_services_start():
    plan = ExperimentPlan.load(ROOT / "configs/g5.experiment.json")
    raw = plan.model_dump()
    raw["run_order"] = ["unknown"]
    with pytest.raises(ValidationError, match="unknown condition"):
        ExperimentPlan.model_validate(raw)
    plan.conditions["smoke"].warmup_ids = ["not-a-question"]
    with pytest.raises(ValueError, match="Unknown warmup"):
        plan.prepare(smoke_only=True)


@pytest.mark.parametrize("extra", [{"sql": "secret"}, {"query": "secret"}, {"answer": "secret"}])
def test_question_source_rejects_reference_fields(tmp_path, extra):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{"id": "a", "database_id": "db", "question": "count?", **extra}])
    )
    source = DatasetSpec(questions=questions, gold=tmp_path / "gold.json")
    with pytest.raises(ValidationError, match="Extra inputs"):
        source.jobs()


def test_analysis_uses_observed_conditions_including_concurrency_two():
    runs = []
    for task, model in [(2, 1), (2, 2), (2, 1)]:
        runs.append(
            {"concurrency": task, "llm_concurrency": model, "sql_workers": 2, "sql_wait_seconds": 0}
        )
    groups = group_conditions(runs)
    assert {key: len(value) for key, value in groups.items()} == {
        "c2-m1-s2-w0": 2,
        "c2-m2-s2-w0": 1,
    }
    for run in runs:
        run["condition_id"] = "configured"
    assert list(group_conditions(runs)) == ["configured"]


def test_aggregation_preserves_failed_and_unscored_denominators():
    run = {
        key: 0
        for key in [
            "submitted",
            "accepted",
            "completed",
            "failed",
            "completed_correct",
            "jct_seconds_sum",
            "model_client_seconds_sum",
            "model_capacity_wait_seconds_sum",
            "application_queue_seconds_sum",
            "sql_stage_seconds_sum",
        ]
    }
    run.update(submitted=3, accepted=2, failed=2, quality={"incorrect": 1, "not_evaluated": 2})
    run.update(
        {
            key: None
            for key in [
                "mean_jct_seconds",
                "p95_jct_seconds_descriptive",
                "mean_model_capacity_wait_seconds",
            ]
        }
    )
    run.update(completed_tasks_per_second=0, correct_tasks_per_second=0)
    result = aggregate([run, run])
    assert result["submitted"] == 6 and result["accepted"] == 4 and result["failed"] == 4
    assert result["quality"] == {"incorrect": 2, "not_evaluated": 4}
    assert result["mean_jct_seconds"] is None
