"""Local-only quality aggregation; private answer reviews never need server upload."""

# ruff: noqa: UP017 -- local artifact analysis also runs on Windows Python 3.10.

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agentserve.evaluation.answers import AnswerReview, summarize_answers


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / ("derived-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    result = {"runs": {}, "answers": {}, "input_sha256": {}}
    attempts, cli_events = [], []
    total_tasks = 0
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not (run / "summary.json").exists():
            continue
        summary = read(run / "summary.json")
        result["runs"][run.name] = summary
        result["input_sha256"][str(run / "summary.json")] = hashlib.sha256(
            (run / "summary.json").read_bytes()
        ).hexdigest()
        for folder in sorted(run.iterdir()):
            if not folder.is_dir() or not (folder / "tasks").exists():
                continue
            records = [read(p) for p in sorted((folder / "tasks").glob("*.json"))]
            total_tasks += len(records)
            for record in records:
                model_steps = {
                    s["step_no"] for s in record["history"]["steps"] if s["kind"] == "model"
                }
                attempts.extend(
                    a for a in record["history"]["attempts"] if a["step_no"] in model_steps
                )
            reviews_path = folder / "answer-reviews.json"
            if reviews_path.exists():
                values = read(reviews_path)
                # This local JSON boundary is independent of the server CLI's TypeAdapter.
                for row in values:
                    if (
                        row["source"] not in {"human", "codex_assisted"}
                        or any(
                            type(row[key]) is not bool
                            for key in ["grounded", "adequate", "truncation_disclosed"]
                        )
                        or any(
                            not isinstance(row[key], str)
                            for key in ["task_id", "reviewer", "evidence"]
                        )
                    ):
                        raise ValueError("Invalid answer review JSON")
                metrics = summarize_answers(
                    records, read(folder / "evaluations.json"), [AnswerReview(**r) for r in values]
                )
                name = str(folder.relative_to(root)).replace("\\", "/")
                result["answers"][name] = metrics
                result["input_sha256"][str(reviews_path)] = hashlib.sha256(
                    reviews_path.read_bytes()
                ).hexdigest()
        if (run / "cli-report.json").exists():
            total_tasks += 1
            cli_events.extend(
                e
                for e in read(run / "cli-report.json")["events"]
                if e["type"] in {"MODEL_COMPLETED", "MODEL_FAILED"}
            )
    audit = read(root / "metrics-audit.json")
    request_ids = [a["request_id"] for a in attempts if a["request_id"]]
    request_ids.extend(
        e["payload"]["request_id"] for e in cli_events if e["payload"].get("request_id")
    )
    usages = [a["usage"] for a in attempts if a["usage"] is not None]
    usages.extend(
        e["payload"]["usage"] for e in cli_events if e["payload"].get("usage") is not None
    )
    result["evidence"] = {
        "tasks": total_tasks,
        "http_requests": sum(r["http_requests"] for r in audit["runs"]),
        "persisted_model_attempts": len(attempts),
        "direct_cli_model_calls": len(cli_events),
        "recorded_response_ids": len(request_ids),
        "unique_response_ids": len(set(request_ids)),
        "unknown_usage_attempts": sum(a["usage"] is None for a in attempts),
        "known_prompt_tokens": sum(u["prompt_tokens"] for u in usages),
        "known_completion_tokens": sum(u["completion_tokens"] for u in usages),
        "valid_gpu_samples": sum(r["gpu_samples"] for r in audit["runs"]),
        "gpu_sample_errors": sum(r["gpu_sample_errors"] for r in audit["runs"]),
        "closed_api_processes": sum(r["api_processes"] for r in audit["runs"]),
        "all_owned_processes_closed": all(
            r["apis_closed"] and r["model_closed"] for r in audit["runs"]
        ),
        "all_query_databases_unchanged": all(r["db_unchanged"] for r in audit["runs"]),
    }
    result["analysis_source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).resolve().parents[2] / "src/agentserve/evaluation/answers.py",
        ]
    }
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    development = max(
        name
        for name, r in result["runs"].items()
        if r.get("evaluated_profiles") == ["baseline", "order_context"]
    )
    product = next(name for name in result["runs"] if name.startswith("product-"))
    baseline = next(
        r
        for r in result["runs"][development]["runs"]
        if r["profile"] == "baseline" and r["dataset"] == "business"
    )
    final = result["runs"][product]["business"]
    b_review = result["answers"][development + "/baseline-business"]
    f_review = result["answers"][product + "/business"]
    acceptance = next(r for name, r in result["runs"].items() if name.startswith("acceptance-"))[
        "runs"
    ]
    os.environ.setdefault("MPLCONFIGDIR", str(root.parents[1] / ".local/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(4)
    before = [
        baseline["completed"],
        baseline["quality"]["correct"],
        b_review["faithful_bounded_responses"],
        b_review["fully_successful"],
    ]
    after = [
        final["completed"],
        final["quality"]["correct"],
        f_review["faithful_bounded_responses"],
        f_review["fully_successful"],
    ]
    for values, offset, label, color in [
        (before, -0.18, "Baseline", "#94a3b8"),
        (after, 0.18, "V1 bounded", "#2563eb"),
    ]:
        bars = axes[0].bar(x + offset, values, 0.36, label=label, color=color)
        axes[0].bar_label(bars, fontsize=9)
    axes[0].set_xticks(
        x, ["Workflow", "SQL match", "Faithful bounded", "Full business"], rotation=12
    )
    axes[0].set_ylim(0, 44)
    axes[0].set_title("Synthetic business: n=40; answer review is Codex-assisted")
    axes[0].legend()
    x = np.arange(3)
    for run, offset, color in zip(acceptance, [-0.18, 0.18], ["#94a3b8", "#2563eb"], strict=True):
        values = [run["completed"], run["quality"]["correct"], run["completed_correct"]]
        bars = axes[1].bar(x + offset, values, 0.36, label=run["profile"], color=color)
        axes[1].bar_label(bars, fontsize=9)
    axes[1].set_xticks(x, ["Workflow", "SQL match", "Completed + correct"])
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Fresh public-dev: n=100; before bounded rendering patch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "quality-comparison.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "evidence": result["evidence"],
                "business_before": {k: v for k, v in b_review.items() if k != "details"},
                "business_after": {k: v for k, v in f_review.items() if k != "details"},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
