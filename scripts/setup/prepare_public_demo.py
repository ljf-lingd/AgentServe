"""Generate the self-contained synthetic demo and its knowledge snapshots."""

import json
from pathlib import Path

from agentserve.knowledge.snapshot import assemble, digest
from agentserve.workloads.business import prepare as prepare_business
from prepare_demo import main as prepare_demo
from prepare_retail_knowledge import main as prepare_retail_knowledge

ROOT = Path(__file__).resolve().parents[2]


def main():
    prepare_demo()
    prepare_business(ROOT)
    prepare_retail_knowledge()
    snapshot = assemble(ROOT / "data/knowledge_v31/manifest.json")
    snapshot["snapshot_sha256"] = digest(snapshot)
    (ROOT / "cache/v32-knowledge-bm25.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Synthetic demo and knowledge snapshots are ready.")


if __name__ == "__main__":
    main()
