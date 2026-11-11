"""Sequential V2 validation with explicit server-side checkpoints and owned services."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    receipt = (
        ROOT / "logs" / ("v2-validation-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json")
    )
    state = {"state": "running", "pid": os.getpid(), "runs": []}
    receipt.write_text(json.dumps(state, indent=2))
    print(str(receipt), flush=True)
    development = None
    try:
        for phase in ["development", "long", "acceptance"]:
            before = set((ROOT / "results/v2-memory").glob("*"))
            command = [sys.executable, str(ROOT / "scripts/experiments/run_memory.py")]
            if phase == "long":
                command += ["--long-probe", "--policies", "simplemem", "sql_episode"]
            else:
                command += [
                    "--policies",
                    "window",
                    "simplemem",
                    "sql_episode",
                    "no_binding",
                    "no_updates",
                    "--regression",
                ]
                if phase == "development":
                    command += ["--recovery"]
                else:
                    command += [
                        "--split",
                        "acceptance",
                        "--freeze",
                        str(development / "freeze.json"),
                    ]
            state["phase"] = phase
            receipt.write_text(json.dumps(state, indent=2))
            with receipt.with_suffix("." + phase + ".log").open("wb") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            created = set((ROOT / "results/v2-memory").glob("*")) - before
            if len(created) != 1:
                raise RuntimeError("Unexpected run directory count")
            run = created.pop()
            if phase == "development":
                development = run
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/experiments/analyze_memory.py"), str(run)],
                cwd=ROOT,
                check=True,
            )
            state["runs"].append({"phase": phase, "path": str(run)})
            receipt.write_text(json.dumps(state, indent=2))
        state["state"] = "completed"
    except BaseException as error:
        state.update(state="failed_or_interrupted", error=repr(error))
        raise
    finally:
        receipt.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
