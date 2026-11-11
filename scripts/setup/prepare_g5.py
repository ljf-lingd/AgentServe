"""Freeze existing Spider resources into versioned, database-balanced G5 subsets."""

import json
from pathlib import Path

from agentserve.workloads.spider import prepare

if __name__ == "__main__":
    manifest = prepare(Path(__file__).resolve().parents[2])
    print(json.dumps({"version": manifest["version"], "splits": manifest["splits"]}), flush=True)
