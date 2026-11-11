"""Regression checks for retrieval coverage and final reporting semantics."""

import copy
import runpy
from pathlib import Path

import pytest

from agentserve.memory.retrieval import retrieve

REPORT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/experiments/analyze_memory.py")
)


def record():
    return {
        "id": "sample-1",
        "conversation": "sample",
        "clarify": False,
        "evaluation": {"status": "correct"},
        "jct_seconds": 999,
        "task": {
            "status": "FAILED",
            "error": {"code": "MEMORY_PROTOCOL_ERROR"},
            "report": {"sql": "SELECT 1"},
            "created_at": 10,
            "ended_at": 12,
        },
    }


def test_each_retrieval_view_can_supply_unique_evidence():
    def entry(key, text, vector, keywords=(), topic=""):
        return {
            "entry_id": key,
            "turn_no": 1,
            "text": text,
            "vector": vector,
            "keywords": list(keywords),
            "topic": topic,
        }

    entries = [
        entry("dense", "unrelated words", [1.0, 0.0]),
        entry("lexical", "Shanghai revenue", [-1.0, 0.0]),
        entry("metadata", "other topic", [0.0, 1.0], ["scope"], "orders"),
    ]
    found = retrieve(entries, [1.0, 0.0], "Shanghai", ["scope"], "orders", 1)
    assert [e["entry_id"] for e in found] == ["dense", "lexical", "metadata"]


def test_correct_sql_with_failed_memory_commit_is_not_turn_success():
    assert not REPORT["turn_success"](record())


def test_expected_clarification_can_succeed_without_sql():
    row = record()
    row["clarify"] = True
    row["task"]["report"]["sql"] = None
    row["task"]["error"]["code"] = "MEMORY_CONTEXT_REQUIRED"
    assert REPORT["turn_success"](row)


def test_completed_incorrect_sql_is_not_turn_success():
    row = record()
    row["evaluation"]["status"] = "incorrect"
    row["task"]["status"] = "COMPLETED"
    assert not REPORT["turn_success"](row)


def test_latency_excludes_offline_audit_and_evaluation():
    assert REPORT["server_jct"](record()) == 2


def test_pairing_rejects_different_question_sets():
    left = record()
    right = copy.deepcopy(left)
    right["id"] = "different-1"
    with pytest.raises(ValueError, match="identical complete turn IDs"):
        REPORT["paired"]([left], [right])
