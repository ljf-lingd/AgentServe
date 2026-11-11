"""Freeze small official SParC sessions, with original utterances and DB fingerprints."""

import hashlib
import json
import re
import sqlite3
import time
import zipfile
from pathlib import Path

from agentserve.workloads.spider import write_frozen

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "data/raw/sparc/official.zip"
    manifest = {
        "official_page": "https://yale-lily.github.io/sparc",
        "license": "CC BY-SA 4.0",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "One 2-6 turn original session per database; stable SHA ranking; "
        "executable fixed references with <=100 rows, schema <=18000 chars; no model-output selection.",
        "db_split": "6 official train databases for development; up to20 official dev databases for acceptance",
        "pretraining_contamination": "unknown",
        "databases": {},
        "splits": {},
    }
    with zipfile.ZipFile(source) as archive:
        for split, official, count in [("development", "train", 6), ("acceptance", "dev", 20)]:
            rows = json.loads(archive.read("sparc/" + official + ".json"))
            ranked = sorted(
                enumerate(rows),
                key=lambda item: hashlib.sha256(f"v21:{official}:{item[0]}".encode()).hexdigest(),
            )
            conversations = []
            labels = {}
            selected = set()
            for index, row in ranked:
                db = row["database_id"]
                if db in selected or not 2 <= len(row["interaction"]) <= 6:
                    continue
                member = f"sparc/database/{db}/{db}.sqlite"
                data = archive.read(member)
                target = ROOT / f"data/databases/sparc_v21/{db}/{db}.sqlite"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(data)
                assert (
                    hashlib.sha256(target.read_bytes()).hexdigest()
                    == hashlib.sha256(data).hexdigest()
                )
                with sqlite3.connect(target.as_uri() + "?mode=ro", uri=True) as conn:
                    conn.execute("PRAGMA query_only=ON")
                    schema = (
                        ";\n".join(
                            x[0]
                            for x in conn.execute(
                                "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                            )
                        )
                        + ";\n"
                    )
                    if len(schema) > 18000:
                        continue
                    valid = True
                    for turn in row["interaction"]:
                        sql = turn["query"]
                        if re.search(
                            r"\b(random|current_date|current_time|current_timestamp)\b|[\x27\"]now[\x27\"]",
                            sql,
                            re.IGNORECASE,
                        ):
                            valid = False
                            break
                        try:
                            deadline = time.monotonic() + 3
                            conn.set_progress_handler(
                                lambda deadline=deadline: int(time.monotonic() > deadline), 10000
                            )
                            if len(conn.execute(sql).fetchmany(101)) > 100:
                                valid = False
                                break
                        except sqlite3.Error:
                            valid = False
                            break
                if not valid:
                    continue
                key = "sparc_" + db
                schema_path = ROOT / f"data/processed/sparc_v21/{db}.sql"
                schema_path.parent.mkdir(parents=True, exist_ok=True)
                schema_path.write_text(schema)
                existing = ROOT / f"data/databases/spider/{db}/{db}.sqlite"
                manifest["databases"][key] = {
                    "path": str(target.relative_to(ROOT)),
                    "schema_path": str(schema_path.relative_to(ROOT)),
                    "context": "Official SParC SQLite database. Preserve the original requested outputs. Do not inherit filters from unrelated questions.",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "same_as_existing_spider": existing.exists()
                    and hashlib.sha256(existing.read_bytes()).hexdigest()
                    == hashlib.sha256(data).hexdigest(),
                    "prior_v1_domain_exposure": "possible; Spider resources already used; not claimed unseen",
                }
                sid = f"sparc-{official}-{index:04d}"
                turns = []
                for n, turn in enumerate(row["interaction"], 1):
                    tid = f"{sid}-{n:02d}"
                    turns.append({"id": tid, "question": turn["utterance"]})
                    labels[tid] = {
                        "sql": turn["query"],
                        "ordered": bool(re.search(r"\border\s+by\b", turn["query"], re.IGNORECASE)),
                        "tags": ["public_sparc"] + (["public_followup"] if n > 1 else []),
                        "clarify": False,
                        "family": db,
                        "length": len(row["interaction"]),
                    }
                conversations.append({"id": sid, "database_id": key, "family": db, "turns": turns})
                selected.add(db)
                if len(selected) == count:
                    break
            write_frozen(ROOT / f"data/splits/sparc_v21/{split}.json", conversations)
            write_frozen(ROOT / f"data/gold/sparc_v21/{split}.json", labels)
            manifest["splits"][split] = {"sessions": len(conversations), "turns": len(labels)}
    write_frozen(ROOT / "data/splits/sparc_v21/manifest.json", manifest)
    print(json.dumps({"splits": manifest["splits"], "databases": len(manifest["databases"])}))


if __name__ == "__main__":
    main()
