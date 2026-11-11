"""Offline V2 reporting v2. Raw records and frozen online code remain unchanged."""

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

SAMPLE_IDS = [
    "a01-03",
    "a02-10",
    "a09-10",
    "a10-03",
    "a12-12",
    "a16-12",
    "a20-12",
    "a22-10",
    "a25-42",
    "a26-48",
    "a27-46",
    "a31-03",
]


def turn_success(record):
    if record["evaluation"]["status"] != "correct":
        return False
    task = record["task"]
    if record["clarify"]:
        return (task.get("error") or {}).get("code") == "MEMORY_CONTEXT_REQUIRED" and not (
            task.get("report") or {}
        ).get("sql")
    return task["status"] == "COMPLETED"


def server_jct(record):
    task = record["task"]
    return task["ended_at"] - task["created_at"]


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "total": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "total": sum(values),
        "p50": values[(len(values) - 1) // 2],
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
    }


def paired(a, b, tag=None):
    if {r["id"] for r in a} != {r["id"] for r in b}:
        raise ValueError("Paired comparison requires identical complete turn IDs")

    def groups(records):
        result = defaultdict(list)
        for row in records:
            if not row["clarify"] and (not tag or tag in row["tags"]):
                result[row["conversation"]].append(row["evaluation"]["status"] == "correct")
        return {k: sum(v) / len(v) for k, v in result.items()}

    left, right = groups(a), groups(b)
    differences = [right[k] - left[k] for k in sorted(left)]
    if not differences:
        return None
    rng = random.Random(20261103)
    estimates = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000)
    )
    return {
        "sessions": len(differences),
        "mean_session_accuracy_difference": sum(differences) / len(differences),
        "bootstrap_95_percent": [estimates[50], estimates[1950]],
        "repetitions": 2000,
        "interpretation": "Paired session means; template correlation limits external inference",
    }


def summarize(records, database):
    lengths = Counter(r["conversation"] for r in records)
    sessions, sql_sessions = defaultdict(list), defaultdict(list)
    for row in records:
        sessions[row["conversation"]].append(turn_success(row))
        sql_sessions[row["conversation"]].append(row["evaluation"]["status"] == "correct")

    def score(rows):
        return {
            "correct": sum(r["evaluation"]["status"] == "correct" for r in rows),
            "total": len(rows),
        }

    sql = [r for r in records if not r["clarify"]]
    phases, usage, unknown = defaultdict(list), Counter(), 0
    phase_by_step = {}
    for row in records:
        mapping = {}
        for step in row["steps"]["steps"]:
            phase = step["input"].get("purpose") or (
                "sql_generation"
                if step["input"].get("query")
                else "answer"
                if step["kind"] == "model"
                else "sql_tool"
            )
            mapping[step["step_no"]] = (step["kind"], phase)
            phase_by_step[(row["task"]["task_id"], step["step_no"])] = phase
        for attempt in row["steps"]["attempts"]:
            kind, phase = mapping[attempt["step_no"]]
            if attempt["ended_at"] is not None:
                phases[phase].append(attempt["ended_at"] - attempt["started_at"])
            if kind == "model":
                if attempt["usage"]:
                    usage.update({k: v for k, v in attempt["usage"].items() if isinstance(v, int)})
                else:
                    unknown += 1
    preparation, writing = [], []
    generation, capacity = defaultdict(float), defaultdict(float)
    prepared_at, started_at, drafts_at = {}, {}, {}
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for tid, kind, offset, encoded in connection.execute(
                "SELECT task_id,type,offset_seconds,payload_json FROM events WHERE type IN "
                "('TASK_STARTED','MEMORY_PREPARED','MEMORY_DRAFT_READY','MODEL_COMPLETED','MODEL_STARTED') ORDER BY event_id"
            ):
                value = json.loads(encoded)
                if kind == "TASK_STARTED":
                    started_at[tid] = offset
                elif kind == "MEMORY_PREPARED":
                    prepared_at[tid] = offset
                elif kind == "MEMORY_DRAFT_READY":
                    drafts_at[tid] = value["client_seconds"]
                elif kind == "MODEL_COMPLETED":
                    phase = phase_by_step.get((tid, value.get("step_no")))
                    if phase in {"memory_plan", "memory_write"}:
                        generation[(tid, phase)] += value["client_seconds"]
                elif kind == "MODEL_STARTED":
                    capacity[(tid, value["phase"])] += value["capacity_wait_seconds"]
        for tid, end in prepared_at.items():
            total = end - started_at[tid]
            preparation.append(
                {
                    "total": total,
                    "non_generation": max(
                        0, total - generation[(tid, "memory_plan")] - capacity[(tid, "memory_plan")]
                    ),
                }
            )
        for tid, total in drafts_at.items():
            writing.append(
                {
                    "total": total,
                    "non_generation": max(
                        0,
                        total - generation[(tid, "memory_write")] - capacity[(tid, "memory_write")],
                    ),
                }
            )
    errors = Counter((r["task"]["error"] or {}).get("code") for r in records if r["task"]["error"])
    return {
        "sql": score(sql),
        "clarification": score([r for r in records if r["clarify"]]),
        "sql_and_workflow": {"correct": sum(turn_success(r) for r in sql), "total": len(sql)},
        "all_turn_success": {
            "correct": sum(turn_success(r) for r in records),
            "total": len(records),
        },
        "whole_conversation_success": {
            "correct": sum(all(v) for v in sessions.values()),
            "total": len(sessions),
        },
        "whole_conversation_sql_or_clarification_correct": {
            "correct": sum(all(v) for v in sql_sessions.values()),
            "total": len(sql_sessions),
        },
        "completed": sum(r["task"]["status"] == "COMPLETED" for r in records),
        "errors": dict(errors),
        "subsets": {
            tag: score([r for r in sql if tag in r["tags"]])
            for tag in ["memory", "long_reference", "correction"]
        },
        "long_session_sql": score([r for r in sql if lengths[r["conversation"]] == 48]),
        "39_turn_reference": score(
            [r for r in sql if lengths[r["conversation"]] == 48 and r["id"].endswith("-42")]
        ),
        "jct_server_seconds": distribution([server_jct(r) for r in records]),
        "jct_completed_seconds": distribution(
            [server_jct(r) for r in records if r["task"]["status"] == "COMPLETED"]
        ),
        "stage_attempt_seconds": {k: distribution(v) for k, v in phases.items()},
        "known_usage": dict(usage),
        "unknown_model_attempt_usage": unknown,
        "prepare_seconds": distribution([p["total"] for p in preparation]),
        "prepare_non_generation_seconds": distribution([p["non_generation"] for p in preparation]),
        "draft_seconds": distribution([p["total"] for p in writing]),
        "draft_non_generation_seconds": distribution([p["non_generation"] for p in writing]),
        "non_generation_definition": "Measured phase residual after HTTP generation and capacity waits; includes embeddings, tokenization, retrieval and persistence, not isolated embedding latency",
        "answer_faithfulness": {"reviewed": 0, "unreviewed": len(records)},
    }


def evidence(row, policy, gold):
    report = row["task"]["report"] or {}
    return {
        "policy": policy,
        "id": row["id"],
        "task_id": row["task"]["task_id"],
        "question": row["question"],
        "status": row["task"]["status"],
        "sql": report.get("sql"),
        "expected_sql": gold.get(row["id"], {}).get("sql"),
        "answer": report.get("answer"),
        "query_result": report.get("query_result"),
        "evaluation": row["evaluation"],
        "error": row["task"]["error"],
        "tags": row["tags"],
    }


def plots(output, metrics, comparisons, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return {"available": False, "reason": "matplotlib not installed"}
    names = list(metrics)
    aliases = {
        "window": "B0",
        "simplemem": "B1",
        "sql_episode": "F1",
        "no_binding": "-binding",
        "no_updates": "-updates",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, label in zip(
        axes.flat,
        ["sql", "memory", "long_reference", "correction"],
        [
            "All SQL turns",
            "Memory-dependent turns",
            "References outside recent window",
            "Condition corrections",
        ],
    ):
        scores = [metrics[n]["sql"] if key == "sql" else metrics[n]["subsets"][key] for n in names]
        values = [s["correct"] / max(1, s["total"]) for s in scores]
        bars = ax.bar(
            [aliases[n] for n in names], values, color=[f"C{i}" for i in range(len(names))]
        )
        ax.set(title=label, ylim=(0, 1.08))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{score['correct']}/{score['total']}",
                ha="center",
                fontsize=8,
            )
    fig.suptitle(title + ": strict SQL execution results")
    fig.tight_layout()
    fig.savefig(output / "sql-and-subsets.png", dpi=180)
    fig.savefig(output / "sql-and-subsets.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = []
    for index, (name, comparison) in enumerate(comparisons.items()):
        value = comparison["overall"]
        mean = value["mean_session_accuracy_difference"] * 100
        low, high = [v * 100 for v in value["bootstrap_95_percent"]]
        ax.errorbar(
            mean, index, xerr=[[max(0, mean - low)], [max(0, high - mean)]], fmt="o", capsize=4
        )
        right, left = name.split("_minus_")
        labels.append(aliases[right] + " - " + aliases[left])
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Mean paired session accuracy difference (percentage points)",
        title="95% paired session bootstrap; synthetic shared templates",
    )
    ax.axvline(0, color="grey", linewidth=1)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "paired-session-differences.png", dpi=180)
    plt.close(fig)
    return {
        "available": True,
        "files": ["sql-and-subsets.png", "sql-and-subsets.svg", "paired-session-differences.png"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = json.loads((args.run / "summary.json").read_text())
    if raw["state"] != "completed":
        raise ValueError(
            "Final analysis requires a completed run; partial data is not scored as acceptance"
        )
    output = args.output or args.run / (
        "derived-final-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(exist_ok=False)
    gold = json.loads((args.project_root / f"data/gold/memory_v2/{raw['split']}.json").read_text())
    all_records, metrics, samples, failures = {}, {}, [], []
    ids = None
    for policy in raw["policies"]:
        path = args.run / policy / "turns.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        keys = {r["id"] for r in records}
        if len(keys) != len(records) or ids is not None and keys != ids:
            raise ValueError("Methods must have identical unique turn IDs")
        ids = keys
        if not raw["smoke_subset"] and keys != set(gold):
            raise ValueError("Full frozen split was not evaluated")
        all_records[policy] = records
        metrics[policy] = summarize(records, args.run / policy / "tasks.sqlite")
        for row in records:
            if raw["split"] == "acceptance" and row["id"] in SAMPLE_IDS:
                samples.append(evidence(row, policy, gold))
        chosen = {}
        for row in sorted(
            records, key=lambda r: (not r["id"].endswith("-42"), "memory" not in r["tags"], r["id"])
        ):
            if not turn_success(row) and not row["clarify"]:
                chosen[row["id"]] = row
            if len(chosen) >= 4:
                break
        for row in chosen.values():
            item = evidence(row, policy, gold)
            item["context"] = {
                k: v for k, v in (row["context"] or {}).items() if k != "planning_messages"
            }
            item["sampling"] = "purposeful failure case; not an error-rate sample"
            failures.append(item)
    comparisons = {}
    for left, right in [
        ("window", "simplemem"),
        ("simplemem", "sql_episode"),
        ("no_binding", "sql_episode"),
        ("no_updates", "sql_episode"),
    ]:
        if left in all_records and right in all_records:
            comparisons[right + "_minus_" + left] = {
                "overall": paired(all_records[left], all_records[right]),
                **{
                    tag: paired(all_records[left], all_records[right], tag)
                    for tag in ["memory", "long_reference", "correction"]
                },
            }
    if raw["split"] == "acceptance" and len(samples) != 12 * len(raw["policies"]):
        raise ValueError("Preselected answer sample is incomplete")
    result = {
        "reporting_version": "memory-report-v2",
        "raw_run": str(args.run),
        "source_sha256": raw["source_sha256"],
        "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": raw["split"],
        "smoke_subset": raw["smoke_subset"],
        "state": raw["state"],
        "turns_per_method": len(ids),
        "session_count": raw["session_count"],
        "metrics": metrics,
        "paired": comparisons,
        "query_databases_unchanged": raw["query_databases_unchanged"],
        "adoption": "requires final boundary, per-case V1 regression, answer review and cost assessment",
        "corrections": [
            "Conversation success requires committed workflow as well as SQL/clarification correctness",
            "JCT uses task ended_at-created_at, excluding offline evaluation and audit calls",
        ],
        "limitations": [
            "Shared synthetic templates; no broad generalization claim",
            "Single unchanged database",
            "Answers require attributed review; unreviewed is not passing",
            "Paired intervals use session means; exploratory subset/ablation comparisons",
        ],
    }
    result["plots"] = plots(
        output, metrics, comparisons, raw["split"] + (" (probe)" if raw["smoke_subset"] else "")
    )
    for name, value in [
        ("compact.json", result),
        ("answer-sample.json", samples),
        ("selected-failures.json", failures),
    ]:
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps({"output": str(output), "policies": len(metrics), "turns_per_method": len(ids)})
    )


if __name__ == "__main__":
    main()
