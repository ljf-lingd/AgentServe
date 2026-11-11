import json
from pathlib import Path

from agentserve.workloads.memory import freeze

if __name__ == "__main__":
    print(json.dumps(freeze(Path(__file__).resolve().parents[2]), indent=2))
