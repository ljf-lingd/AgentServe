"""Build and query a pinned, database-scoped knowledge snapshot."""

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from contextlib import closing
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agentserve.knowledge.facts import fact_evidence, install_facts
from agentserve.memory.resources import MemoryResources

INSTRUCTION = "Instruct: Retrieve Chinese order business rules that answer the question.\nQuery: "
Purpose = Literal["product", "test", "benchmark"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized(value):
    return unicodedata.normalize("NFKC", value).casefold().strip()


def terms(value):
    words = []
    for part in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", normalized(value)):
        if len(part) > 1 and "\u3400" <= part[0] <= "\u9fff":
            words.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            words.append(part)
    return words


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    clause_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    path: str
    heading: str | None = Field(default=None, min_length=1)
    conflict_group: str | None = Field(default=None, pattern=r"^[a-z0-9_-]+$")
    database_id: str
    purpose: Purpose
    approval_ref: str | None = Field(default=None, min_length=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    version: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    effective_from: date
    effective_to: date | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: date
    purpose: Purpose
    documents: list[Document]
    static: list[str]
    fact_index: str | None = None


def sections(raw):
    headings = []
    start = 0
    for heading in re.finditer(r"^(#{1,6})[^\S\n]+(.+)$", raw, re.MULTILINE):
        if body := raw[start : heading.start()].strip():
            yield " / ".join(name for _, name in headings), body, start
        level = len(heading[1])
        headings = [(depth, name) for depth, name in headings if depth < level]
        headings.append((level, heading[2]))
        start = heading.end()
    if body := raw[start:].strip():
        yield " / ".join(name for _, name in headings), body, start


def assemble(manifest_path):
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.resolve().parent
    chunks = []
    versions = set()
    intervals = {}
    for document in manifest.documents:
        if document.purpose != manifest.purpose or (
            manifest.purpose != "test" and document.approval_ref is None
        ):
            raise ValueError(f"Ineligible source {document.id} for {manifest.purpose} snapshot")
        key = (document.id, document.version)
        if key in versions:
            raise ValueError(f"Duplicate source version {key}")
        versions.add(key)
        path = (root / document.path).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix != ".md":
            raise ValueError(f"Invalid Markdown source {document.path}")
        if document.effective_to and document.effective_from >= document.effective_to:
            raise ValueError(f"Invalid effective interval for {document.id}")
        if document.approval_ref:
            intervals.setdefault((document.database_id, document.clause_id), []).append(
                (document.effective_from, document.effective_to)
            )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != document.source_sha256:
            raise ValueError(f"Source digest changed for {document.path}")
        source = raw.decode("utf-8").replace("\r\n", "\n")
        blocks = [
            block
            for block in sections(source)
            if document.heading is None or block[0] == document.heading
        ]
        if not blocks or (document.heading is not None and len(blocks) != 1):
            raise ValueError(f"Missing or ambiguous Markdown anchor in {document.path}")
        for i, (title, body, offset) in enumerate(blocks, 1):
            if len(body) > 2400:
                raise ValueError(f"Evidence block too large in {document.path}")
            start = source.index(body, offset)
            line = source.count("\n", 0, start) + 1
            chunks.append(
                {
                    **document.model_dump(mode="json"),
                    "chunk_id": f"{document.id}@{document.version}#{i}",
                    "title": title,
                    "text": body,
                    "anchor": f"L{line}-L{line + body.count(chr(10))}",
                    "content_sha256": digest(body),
                    "terms": " ".join(terms(title + " " + body)),
                }
            )
    for key, ranges in intervals.items():
        ordered = sorted(ranges, key=lambda interval: interval[0])
        for (_, previous_end), (start, _) in pairwise(ordered):
            if previous_end is None or start < previous_end:
                raise ValueError(f"Overlapping approved versions for {key}")
    ids = {c["chunk_id"] for c in chunks}
    approved_ids = {c["chunk_id"] for c in chunks if c["approval_ref"]}
    if not chunks or len(ids) != len(chunks) or not set(manifest.static) <= approved_ids:
        raise ValueError("Snapshot requires unique chunks and valid static IDs")
    if len(set(manifest.static)) != len(manifest.static):
        raise ValueError("Duplicate fixed evidence ID")
    properties = (
        install_facts(root, manifest.fact_index, chunks, manifest.purpose)
        if manifest.fact_index
        else None
    )
    return {
        "format": 3,
        "purpose": manifest.purpose,
        "as_of": manifest.as_of.isoformat(),
        "static": manifest.static,
        "chunks": chunks,
        "embedding_identity": None,
        "instruction": INSTRUCTION,
        **({"properties": properties} if properties is not None else {}),
    }


async def build_snapshot(manifest_path, output, resources=None):
    value = assemble(manifest_path)
    if resources:
        value["embedding_identity"] = resources.info.identity
        for offset in range(0, len(value["chunks"]), 8):
            batch = value["chunks"][offset : offset + 8]
            vectors = await resources.embed(
                [c["title"] + "\n" + c["text"] for c in batch],
                query=False,
                deadline=asyncio.get_running_loop().time() + 60,
            )
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk["vector"] = vector
    value["snapshot_sha256"] = digest(value)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    return value["snapshot_sha256"]


def model_evidence(chunks):
    return [
        {
            "title": c["title"],
            "sentences": {
                f"s{index}_{number}": match.group().strip()
                for number, match in enumerate(re.finditer(r"[^\n。！？]+[。！？]*", c["text"]), 1)
                if match.group().strip()
            },
            **({"conflict_group": c["conflict_group"]} if c["conflict_group"] else {}),
        }
        for index, c in enumerate(chunks, 1)
    ]


def evidence_for_budget(chunks, typed):
    if typed:
        return {key: text for text, key in fact_evidence(chunks).items()}
    return model_evidence(chunks)


def lexical_rank(active, question):
    query_terms = list(dict.fromkeys(terms(question)))
    if not active or not query_terms:
        return []
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, body)")
        conn.executemany(
            "INSERT INTO chunks VALUES (?,?)", ((key, c["terms"]) for key, c in active.items())
        )
        return [
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks),id LIMIT 10",
                (" OR ".join(f'"{term}"' for term in query_terms),),
            )
        ]


class KnowledgeSnapshot:
    def __init__(self, path, strategy, resources, *, purpose="product"):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.identity = data.pop("snapshot_sha256")
        if digest(data) != self.identity or data["format"] != 3:
            raise ValueError("Invalid knowledge snapshot digest or format")
        if data["purpose"] != purpose:
            raise ValueError("Knowledge snapshot purpose does not match configuration")
        self.chunks = data["chunks"]
        self.properties = data.get("properties")
        self.purpose = purpose
        self.as_of = data["as_of"]
        self.strategy, self.resources = strategy, resources
        self.static = data["static"]
        self.instruction = data["instruction"]
        if strategy in {"dense", "rrf"} and (
            data["embedding_identity"] != resources.info.identity
            or any(len(c.get("vector", [])) != 1024 for c in self.chunks)
        ):
            raise ValueError("Knowledge embedding identity or dimensions changed")

    def active(self, database_id, as_of, clause_id=None):
        return {
            c["chunk_id"]: c
            for c in self.chunks
            if c["database_id"] == database_id
            and c["approval_ref"]
            and (clause_id is None or c["clause_id"] == clause_id)
            and c["effective_from"] <= as_of
            and (c["effective_to"] is None or as_of < c["effective_to"])
        }

    async def select(self, database_id, question, deadline, *, as_of, clause_id=None):
        active = self.active(database_id, as_of, clause_id)
        lexical = lexical_rank(active, question) if self.strategy in {"bm25", "rrf"} else []
        dense = []
        if active and self.strategy in {"dense", "rrf"}:
            (vector,) = await self.resources.embed(
                [self.instruction + question], query=False, deadline=deadline
            )
            dense = sorted(
                active,
                key=lambda key: (
                    -sum(a * b for a, b in zip(vector, active[key]["vector"], strict=True)),
                    key,
                ),
            )[:10]
        if self.strategy == "static":
            rank = [key for key in self.static if key in active]
        elif self.strategy == "rrf":
            scores = {}
            for ranking in (lexical, dense):
                for position, key in enumerate(ranking, 1):
                    scores[key] = scores.get(key, 0) + 1 / (60 + position)
            rank = sorted(scores, key=lambda key: (-scores[key], key))
        else:
            rank = lexical or dense

        conflicts = defaultdict(list)
        for key, chunk in active.items():
            if group := chunk["conflict_group"]:
                conflicts[group].append(key)
        selected, tokens = [], 0
        considered = set()
        for key in rank:
            if key in considered:
                continue
            keys = conflicts.get(active[key]["conflict_group"], [key])
            considered.update(keys)
            if len(selected) + len(keys) > 4:
                continue
            candidate = selected + [
                {k: v for k, v in active[key].items() if k not in {"vector", "terms"}}
                for key in keys
            ]
            count = await self.resources.count(
                json.dumps(
                    evidence_for_budget(candidate, self.properties is not None), ensure_ascii=False
                ),
                deadline,
            )
            if count <= 1200:
                selected, tokens = candidate, count
            if len(selected) == 4:
                break
        return {
            "snapshot_sha256": self.identity,
            "strategy": self.strategy,
            "database_id": database_id,
            "as_of": as_of,
            "clause_id": clause_id,
            "candidate_ids": list(dict.fromkeys([*rank, *(c["chunk_id"] for c in selected)])),
            "chunks": selected,
            "evidence_tokens": tokens,
        }


async def build_command(args):
    if args.resource_url is None:
        return await build_snapshot(args.manifest, args.output)
    async with httpx.AsyncClient(
        base_url=args.resource_url.rstrip("/") + "/", trust_env=False
    ) as client:
        resources = MemoryResources(client)
        await resources.connect("")
        return await build_snapshot(args.manifest, args.output, resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resource-url")
    print(asyncio.run(build_command(parser.parse_args())))


if __name__ == "__main__":
    main()
