"""Derive figures and metrics in a new directory; preserve historical results and labels."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from agentserve.analysis import plots, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    output = args.output or root / ("derived-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    summary = report.read(root / "summary.json")
    result = report.derive(root, summary)
    result.update(
        source_summary_sha256=hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
        analysis_source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path(report.__file__), Path(plots.__file__)]
        },
    )
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plots.workload_plots(summary, root, output)
    plots.task_timeline(root, output, summary["performance_runs"])
    print(json.dumps({"output": str(output), "evidence": result["evidence"]}), flush=True)


if __name__ == "__main__":
    main()
