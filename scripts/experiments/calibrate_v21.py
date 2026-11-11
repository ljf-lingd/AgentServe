"""Balanced C0/C1 development calibration, fixed before observing its results."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from analyze_memory import turn_success

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = (
        ROOT
        / "results/v2.1-memory"
        / ("calibration-design-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "session_length": 24,
        "families": [0, 1, 3, 9],
        "policies": ["state_on_demand"],
        "order": ["C0", "C1", "C1", "C0"],
        "repeats_per_profile": 2,
        "warmup": "One count query per model start, not measured; independent state DB per run",
        "quality_gate": "Reject C1 on any newly incorrect case compared with either paired C0; no offset by improvements",
        "performance_gate": "C1 retained only when mean wall time improves >=5%, quality gate passes, no new infrastructure failures",
        "stop_rule": "Run all four once; stop on infrastructure failure; no best-run selection",
        "memory_budget": 0.45,
        "context_tokens": 8192,
        "max_num_seqs": {"C0": 2, "C1": 4},
    }
    path = output / "registration.json"
    path.write_text(json.dumps(design, indent=2))
    runs = []
    for index, profile in enumerate(design["order"]):
        log = output / f"{index}-{profile}.log"
        with log.open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/experiments/run_v21.py"),
                    "--stage",
                    "calibration",
                    "--policies",
                    "state_on_demand",
                    "--serving",
                    profile,
                    "--calibration-design",
                    str(path),
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        lines = log.read_text().splitlines()
        run = Path(json.loads(lines[0])["output"])
        summary = json.loads((run / "summary.json").read_text())
        assert summary["state"] == "completed"
        rows = {r["id"]: r for r in map(json.loads, (run / "state_on_demand/turns.jsonl").open())}
        runs.append(
            {
                "profile": profile,
                "path": str(run.relative_to(ROOT)),
                "wall_seconds": summary["runs"][0]["wall_seconds"],
                "outcomes": {k: turn_success(r) for k, r in rows.items()},
                "failure_codes": {
                    k: (r["task"]["error"] or {}).get("code") for k, r in rows.items()
                },
            }
        )
    regressions = []
    infrastructure = []
    for left, right in [(runs[0], runs[1]), (runs[3], runs[2])]:
        for key in left["outcomes"]:
            if left["outcomes"][key] and not right["outcomes"][key]:
                regressions.append({"id": key, "C0": left["path"], "C1": right["path"]})
            code = right["failure_codes"][key]
            if (
                code
                in {
                    "INTERNAL_ERROR",
                    "TASK_TIMEOUT",
                    "MODEL_TIMEOUT",
                    "MEMORY_RESOURCE_ERROR",
                    "MODEL_RETRY_EXHAUSTED",
                    "MODEL_BUDGET_EXHAUSTED",
                }
                and code != left["failure_codes"][key]
            ):
                infrastructure.append(
                    {"id": key, "code": code, "C0": left["path"], "C1": right["path"]}
                )
    times = {p: sum(r["wall_seconds"] for r in runs if r["profile"] == p) / 2 for p in ["C0", "C1"]}
    selected = (
        "C1"
        if not regressions and not infrastructure and times["C1"] <= 0.95 * times["C0"]
        else "C0"
    )
    (output / "decision.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "mean_wall_seconds": times,
                "new_regressions": regressions,
                "new_infrastructure_failures": infrastructure,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected": selected,
                "mean_wall_seconds": times,
                "regressions": len(regressions),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
