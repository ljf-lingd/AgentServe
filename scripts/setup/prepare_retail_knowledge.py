import json
from pathlib import Path

from agentserve.knowledge.snapshot import assemble, digest

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "data/knowledge_online_retail_ii"


def main():
    base = assemble(ROOT / "data/knowledge_v31/manifest.json")
    retail = assemble(RULES / "manifest.json")
    base["chunks"].extend(retail["chunks"])
    base["properties"].update(retail["properties"])
    base["snapshot_sha256"] = digest(base)
    output = ROOT / "cache/v7-knowledge-bm25.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{output}: {len(base['chunks'])} clauses {base['snapshot_sha256']}")


if __name__ == "__main__":
    main()
