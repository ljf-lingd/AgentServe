"""Deterministic database-balanced Spider subsets; runtime questions never contain gold SQL."""

import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced(rows, count, seed, exclude=()):
    groups = defaultdict(list)
    excluded = set(exclude)
    for row in rows:
        if row["id"] not in excluded:
            groups[row["db_id"]].append(row)

    def rank(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    databases = sorted(groups, key=rank)
    for group in groups.values():
        group.sort(key=lambda row: rank(row["id"]))
    chosen = []
    offset = 0
    while len(chosen) < count:
        available = [groups[db][offset] for db in databases if offset < len(groups[db])]
        if not available:
            raise ValueError("Not enough distinct questions for requested subset")
        chosen.extend(available[: count - len(chosen)])
        offset += 1
    return chosen


def write_frozen(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Frozen output differs; use a new version: {path}")
    else:
        path.write_text(text, encoding="utf-8")


def prepare(root: Path, *, seed=42, sizes=(20, 100, 100)):
    raw = root / "data/raw/spider"
    sources = {"train": raw / "train_spider.json", "dev": raw / "dev.json"}
    rows = {}
    for split, path in sources.items():
        records = json.loads(path.read_text(encoding="utf-8"))
        rows[split] = []
        for index, row in enumerate(records):
            if not all(
                isinstance(row[key], str) and row[key] for key in ("db_id", "question", "query")
            ):
                raise ValueError("Invalid Spider input record")
            rows[split].append({**row, "id": f"spider.{split}.{index:06d}"})
    if {r["db_id"] for r in rows["train"]} & {r["db_id"] for r in rows["dev"]}:
        raise ValueError("Official train/dev databases must be disjoint")
    smoke = balanced(rows["train"], sizes[0], seed)
    development = balanced(rows["train"], sizes[1], seed, exclude=[r["id"] for r in smoke])
    evaluation = balanced(rows["dev"], sizes[2], seed)
    selected = {"smoke": smoke, "development": development, "evaluation": evaluation}
    base = root / "data/splits/spider_g5_v1"
    gold_base = root / "data/gold/spider_g5_v1"
    schema_base = root / "data/processed/spider_g5_v1"
    databases = {}
    database_hashes = {}
    for db in sorted({r["db_id"] for group in selected.values() for r in group}):
        db_root = (root / "data/databases/spider").resolve()
        path = (db_root / db / (db + ".sqlite")).resolve(strict=True)
        if not path.is_relative_to(db_root):
            raise ValueError("Database path escaped approved root")
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            definitions = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        schema = ";\n".join(row[0].rstrip(";") for row in definitions if row[0]) + ";\n"
        schema_path = schema_base / (db + ".sql")
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        if schema_path.exists() and schema_path.read_text(encoding="utf-8") != schema:
            raise ValueError("Frozen schema changed")
        schema_path.write_text(schema, encoding="utf-8")
        databases["spider_" + db] = {
            "path": str(path.relative_to(root.resolve())),
            "schema_path": str(schema_path.relative_to(root)),
            "context": "Use SQLite syntax. Respect exact table and column names; do not invent values.",
        }
        database_hashes[db] = {
            "database_sha256": digest(path),
            "schema_sha256": digest(schema_path),
        }
    manifests = {}
    for split, group in selected.items():
        questions = [
            {
                "id": r["id"],
                "source_db_id": r["db_id"],
                "database_id": "spider_" + r["db_id"],
                "question": r["question"],
            }
            for r in group
        ]
        gold = [
            {"id": r["id"], "sql": r["query"], "ordered": bool(r["sql"]["orderBy"])} for r in group
        ]
        qpath = base / (split + ".json")
        gpath = gold_base / (split + ".json")
        write_frozen(qpath, questions)
        write_frozen(gpath, gold)
        manifests[split] = {
            "count": len(group),
            "databases": sorted({r["db_id"] for r in group}),
            "questions_sha256": digest(qpath),
            "gold_sha256": digest(gpath),
        }
    write_frozen(base / "databases.json", databases)
    manifest = {
        "version": "spider_g5_v1",
        "seed": seed,
        "selection": "sha256_ranked_databases_round_robin_and_ranked_question_ids",
        "source_sha256": {key: digest(path) for key, path in sources.items()},
        "splits": manifests,
        "database_hashes": database_hashes,
        "train_dev_database_overlap": 0,
        "smoke_development_task_overlap": 0,
        "evaluation_is_official_public_dev_not_hidden_test": True,
    }
    write_frozen(base / "manifest.json", manifest)
    return manifest
