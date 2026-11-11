"""Owned tmux pipeline: qualified development, balanced calibration, one frozen acceptance."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from register_v21 import development_failures
from run_v21 import ROOT


def main():
    output = (
        ROOT / "results/v2.1-memory" / ("pipeline-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "state": "running",
        "stage": "prepared",
        "pid": os.getpid(),
        "tmux": "agent-v21",
        "runs": {},
    }
    pointer = ROOT / "logs/v21-active-pipeline.json"

    def save():
        status["updated_at"] = datetime.now(UTC).isoformat()
        for path in [output / "progress.json", pointer]:
            path.write_text(json.dumps({"output": str(output), **status}, indent=2))

    def execute(stage, script, args, *, required=True):
        status["stage"] = stage
        save()
        log = output / (stage + ".log")
        with log.open("w") as handle:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/experiments" / script), *args],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            status["child_pid"] = child.pid
            save()
            code = child.wait()
        if required and code:
            raise RuntimeError(f"{stage} exited {code}; see {log}")
        values = []
        for line in log.read_text().splitlines():
            if line.startswith("{"):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "output" in value or "registration" in value:
                    values.append(value)
        return values

    save()
    try:
        values = execute(
            "development",
            "run_v21.py",
            [
                "--stage",
                "development",
                "--policies",
                "window",
                "state_on_demand",
                "--public",
                "--regression",
                "--recovery",
                "--targeted-controls",
            ],
        )
        development = Path(values[0]["output"])
        status["runs"]["development"] = str(development)
        failures = development_failures(
            json.loads((development / "summary.json").read_text()), development
        )
        if failures:
            status.update(state="development_not_qualified", failures=failures)
            save()
            return
        values = execute("calibration", "calibrate_v21.py", [])
        calibration = Path(values[-1]["output"]) / "decision.json"
        status["runs"]["calibration"] = str(calibration)
        values = execute(
            "registration",
            "register_v21.py",
            ["--development", str(development), "--calibration", str(calibration)],
        )
        registration = Path(values[-1]["registration"])
        serving = values[-1]["serving"]
        status["runs"]["registration"] = str(registration)
        status["serving"] = serving
        values = execute(
            "acceptance",
            "run_v21.py",
            [
                "--stage",
                "acceptance",
                "--registration",
                str(registration),
                "--policies",
                "window",
                "state_on_demand",
                "--serving",
                serving,
                "--public",
                "--regression",
                "--recovery",
                "--targeted-controls",
            ],
        )
        acceptance = Path(values[0]["output"])
        status["runs"]["acceptance"] = str(acceptance)
        values = execute(
            "product",
            "run_memory_product.py",
            ["--policy", "state_on_demand", "--serving", serving, "--relative-probe"],
            required=False,
        )
        product = Path(values[-1]["output"])
        status["runs"]["product"] = str(product)
        execute(
            "analysis", "analyze_v21.py", [str(acceptance), "--registration", str(registration)]
        )
        (acceptance / "derived-v21/product-index.json").write_text(
            json.dumps({"path": str(product)}, indent=2)
        )
        execute("gate", "gate_v21.py", [str(acceptance), "--registration", str(registration)])
        status.update(
            state="predictions_and_non_answer_audit_completed",
            stage="awaiting_local_attributed_answer_review",
        )
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
