import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "failed_id,rejected",
    [
        ("business.no_completed", False),
        ("business.sku_normalized", True),
    ],
)
def test_same_total_cannot_hide_a_new_single_turn_failure(
    tmp_path, monkeypatch, failed_id, rejected
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    from register_v21 import development_failures

    (tmp_path / "single-turn").mkdir()
    (tmp_path / "single-turn/evaluations.json").write_text(
        json.dumps(
            [{"id": f"known-{n}", "status": "correct"} for n in range(39)]
            + [{"id": failed_id, "status": "incorrect"}]
        )
    )
    (tmp_path / "state_on_demand").mkdir()
    (tmp_path / "state_on_demand/turns.jsonl").write_text(
        json.dumps(
            {
                "tags": [],
                "evaluation": {"status": "correct"},
                "task": {"status": "COMPLETED"},
            }
        )
        + "\n"
    )
    development = {
        "state": "completed",
        "runs": [
            {
                "policy": "state_on_demand",
                "subsets": {
                    tag: {"correct": 1, "n": 1}
                    for tag in ["inheritance", "correction", "old_version", "clarification"]
                },
            }
        ],
        "single_turn": {"quality": {"correct": 39}},
        "targeted_controls": [{"quality": {"correct": 13}, "submitted": 13} for _ in range(3)],
    }
    failures = development_failures(development, tmp_path)
    assert bool(failures) is rejected
    if rejected:
        assert failures == [{"single_turn_new_failures": [failed_id]}]


def test_paired_gate_counts_failed_workflow_even_with_correct_sql(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    from analyze_v21 import paired_cluster

    row = {
        "id": "x",
        "clarify": False,
        "tags": [],
        "conversation": "c",
        "evaluation": {"status": "correct"},
        "task": {"status": "COMPLETED"},
    }
    failed = {**row, "task": {"status": "FAILED"}}
    result = paired_cluster([row], [failed], "independent")
    assert result["cluster_equal_mean_difference"] == -1
    assert result["bootstrap95"] == [-1, -1]
