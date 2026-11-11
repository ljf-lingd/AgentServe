"""Continue the V3.2 formal batch interrupted before its first knowledge task."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from run_v21 import write
from run_v22 import verify_owned_services
from run_v32 import CONFIG, ROOT, final_checks, frozen_inputs, gate, start_runtime
from v32_cases import EVALUATION, read, sha

from agentserve.config import Settings

RUNNER_FILES = {
    "scripts/experiments/run_v32.py", "scripts/experiments/continue_v32_acceptance.py",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    first = run / "acceptance"
    output = run / "acceptance-continuation"
    registration = read(run / "registration.json")
    summary = read(first / "summary.json")
    assert summary["stage"] == "acceptance" and summary["state"] == "started"
    assert summary["frozen"] == registration["frozen"]
    assert not any((first / "knowledge").iterdir())
    assert not any((first / k).exists() for k in ("controls", "truncation", "recovery"))
    assert summary["metric_all_db"] == {"n": 24, "passed": 24}
    assert summary["sql_regression"]["n"] == summary["sql_regression"]["passed"] == 24
    assert not summary["owned_survivors"] and summary["databases_unchanged"]
    current = frozen_inputs()
    runners = {path: current["sources"].pop(path) for path in RUNNER_FILES}
    original = registration["frozen"]
    expected = {**original, "sources": {k: v for k, v in original["sources"].items() if k not in RUNNER_FILES}}
    assert current == expected

    hashes = {
        Path(path): value
        for value, path in (line.split("  ", 1) for line in (run / "acceptance-databases.sha256").read_text().splitlines())
    }
    hashes = {(run / path).resolve(strict=True): value for path, value in hashes.items()}
    assert len(hashes) == 5 and all(sha(path) == value for path, value in hashes.items())
    snapshot = ROOT / "cache/v32-knowledge-bm25.json"
    assert sha(snapshot) == summary["knowledge_snapshot_sha256"]
    settings = Settings.load(CONFIG)
    output.mkdir()
    summary["continuation"] = {
        "original_summary_sha256": sha(first / "summary.json"),
        "registration_sha256": sha(run / "registration.json"),
        "database_freeze_sha256": sha(run / "acceptance-databases.sha256"),
        "harness_sha256": runners,
        "retained": ["metric", "sql_regression"],
        "continued": ["knowledge_regression", "counterfactual", "truncation", "recovery"],
    }
    try:
        with ExitStack() as stack:
            cpu = start_runtime(stack, settings, output, summary, registration)
            final_checks(
                "acceptance", settings, snapshot, output, summary, cpu,
                read(EVALUATION / "acceptance.json"),
            )
        summary["state"] = "completed"
    finally:
        summary["databases_unchanged"] = all(sha(path) == value for path, value in hashes.items())
        summary["owned_survivors"] = verify_owned_services(first) + verify_owned_services(output)
        if summary["state"] == "completed":
            summary["gate"] = gate(summary, "acceptance")
            summary["failed_exits"] = [key for key, passed in summary["gate"].items() if not passed]
            summary["can_enter_v4"] = not summary["failed_exits"]
        write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("state", "failed_exits", "can_enter_v4")}))
    raise SystemExit(not summary["can_enter_v4"])


if __name__ == "__main__":
    main()
