"""Run the V3.1 knowledge comparison against the pinned GPU Qwen profile."""

import argparse
import asyncio
import hashlib
import json
import time
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path

import httpx
from gpu_monitor import monitor
from http_server import http_server
from model_server import temporary_model
from run_v21 import fingerprint, write
from run_v22 import verify_owned_services
from v31_cases import DATABASE, label, metrics, p95, request_for, run_requests

from agentserve.application.bootstrap import open_task_service
from agentserve.config import KnowledgeSettings, Settings
from agentserve.domain.models import AgentError, RunReport
from agentserve.evaluation.sql import SQLEvaluator
from agentserve.knowledge.service import KnowledgeService
from agentserve.knowledge.snapshot import (
    KnowledgeSnapshot,
    assemble,
    build_snapshot,
    digest,
    evidence_for_budget,
    lexical_rank,
)
from agentserve.memory.resources import MemoryResources

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/app.v221.C1.json"
PRODUCT = ROOT / "data/knowledge_v31/manifest.json"
FIXTURES = ROOT / "data/synthetic/knowledge_v3/scenarios.json"
EVALUATION = ROOT / "data/evaluation/knowledge_v31"
ARMS = ("static", "bm25", "dense", "rrf")
PROTOCOL = "approved-properties-contract-v4"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs():
    paths = [
        *PRODUCT.parent.glob("*"),
        FIXTURES,
        *FIXTURES.parent.glob("scenarios*.md"),
        *EVALUATION.glob("*.json"),
        ROOT / "data/processed/business_orders_v1/questions.json",
        ROOT / "data/gold/business_orders_v1/answers.json",
        ROOT / "configs/models/qwen2.5-7b-instruct.manifest.json",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths) if path.is_file()}


@contextmanager
def application(base, snapshot, purpose, strategy, output, phase="api", **kwargs):
    output.mkdir(parents=True, exist_ok=True)
    settings = base.model_copy(deep=True)
    settings.knowledge = KnowledgeSettings(
        snapshot_path=snapshot, strategy=strategy, purpose=purpose
    )
    settings.state_database = output / "tasks.sqlite"
    config = output / f"{phase}-app.json"
    config.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    env = {"AGENTSERVE_CONFIG": str(config), **kwargs.pop("env", {})}
    with http_server(
        kwargs.pop("factory", "agentserve.api.app:create_app"), output, phase, env=env, **kwargs
    ) as client:
        yield client


def subset_snapshot(full, output, ids):
    value = {k: v for k, v in full.items() if k != "snapshot_sha256"}
    value["chunks"] = [c for c in full["chunks"] if c["id"] in ids]
    keys = {c["chunk_id"] for c in value["chunks"]}
    value["static"] = [c for c in full["static"] if c in keys]
    value["snapshot_sha256"] = digest(value)
    write(output, value)


def challenge_snapshots(dataset, full_path, directory):
    full = read(full_path)
    plans = {}
    versions = dataset["versions"]
    version_ids = {c["id"] for c in versions}
    before_release = {
        "chunks": [c for c in full["chunks"] if c["version"] == "1"],
        "static": full["static"],
        **{k: v for k, v in full.items() if k not in {"chunks", "static"}},
    }
    subset_snapshot(before_release, directory / "previous.json", version_ids)
    distractors = set(
        dataset.get("version_distractors", ["formal-hold-draft", "formal-hold-other"])
    )
    subset_snapshot(full, directory / "versions.json", version_ids | distractors)
    version_cases = []
    for family in versions:
        for date_field, version, value in zip(
            ("before", "after"), ("1", "2"), family["values"], strict=True
        ):
            for explicit in (False, True):
                version_cases.append(
                    {
                        "id": f"{family['id']}-{version}-{'id' if explicit else 'natural'}",
                        "question": family["question"],
                        "as_of": family[date_field],
                        **({"clause_id": family["id"]} if explicit else {}),
                        "outcome": "answered",
                        "supports": [{"chunk_id": f"{family['id']}@{version}#1", "must": [value]}],
                    }
                )
    plans["versions"] = version_cases
    conflicts = [c for c in dataset["cases"] if c.get("scenario") == "fixture"]
    conflicts.extend(dataset.get("regression_conflicts", []))
    conflict_ids = {key.split("@")[0] for c in conflicts for key in c["conflict_ids"]}
    subset_snapshot(full, directory / "conflicts.json", conflict_ids)
    plans["conflicts"] = conflicts
    for condition, value_index in (("a", 0), ("b", 1), ("empty", None), ("topical", None)):
        for family in dataset["dependencies"]:
            ident = f"{family['id']}-{condition}"
            subset_snapshot(full, directory / f"{ident}.json", {ident})
            case = {
                "id": ident,
                "question": family["question"],
                "outcome": "insufficient_evidence" if value_index is None else "answered",
            }
            if value_index is not None:
                case["supports"] = [
                    {
                        "chunk_id": f"{family['id']}-{condition}@1#1",
                        "must": [family["values"][value_index]],
                    }
                ]
            plans[ident] = [case]
    subset_snapshot(full, directory / "injection.json", {"document-injection"})
    plans["injection"] = [
        {
            "id": "document-injection",
            "question": "注入夹具登记了售后保留期限的天数吗？",
            "outcome": "insufficient_evidence",
        }
    ]
    return plans


async def snapshot_inputs(resource_url, identity, directory):
    directory.mkdir()
    async with httpx.AsyncClient(base_url=resource_url + "/", trust_env=False) as client:
        resources = MemoryResources(client)
        await resources.connect(identity)
        await build_snapshot(PRODUCT, directory / "product.json", resources)
        await build_snapshot(FIXTURES, directory / "fixtures.json", resources)


def preflight(dataset, snapshots, plans, cpu):
    product = KnowledgeSnapshot(snapshots / "product.json", "static", None)
    versions = KnowledgeSnapshot(snapshots / "versions.json", "static", None, purpose="test")
    previous = KnowledgeSnapshot(snapshots / "previous.json", "static", None, purpose="test")
    assert product.identity != versions.identity != previous.identity
    assert all(c["purpose"] == "product" and c["approval_ref"] for c in product.chunks)
    try:
        KnowledgeSnapshot(snapshots / "versions.json", "static", None)
    except ValueError as error:
        assert "purpose" in str(error)
    else:
        raise AssertionError("Product loaded a fixture")
    for case in dataset["cases"]:
        if case["outcome"] != "answered":
            continue
        by_id = product.active(DATABASE, product.as_of)
        chunks = [by_id[s["chunk_id"]] for s in case["supports"]]
        for chunk, support in zip(chunks, case["supports"], strict=True):
            assert all(phrase in chunk["text"] for phrase in support["must"]), case["id"]
        count = (
            cpu.post(
                "/count",
                json={
                    "texts": [
                        json.dumps(
                            evidence_for_budget(chunks, product.properties is not None),
                            ensure_ascii=False,
                        )
                    ]
                },
            )
            .raise_for_status()
            .json()["counts"][0]
        )
        assert len(chunks) <= 4 and count <= 1200, case["id"]
    for case in plans["versions"]:
        if case["id"].endswith("-1-id"):
            key = case["supports"][0]["chunk_id"]
            old = previous.active(DATABASE, case["as_of"])[key]
            assert old == versions.active(DATABASE, case["as_of"])[key]
            raw = (FIXTURES.parent / old["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == old["source_sha256"]
            first, last = (int(n[1:]) for n in old["anchor"].split("-"))
            assert "\n".join(raw.decode().splitlines()[first - 1 : last]).strip() == old["text"]
    assert all("evaluation" not in c["path"] for c in assemble(PRODUCT)["chunks"])
    return {"test_snapshot_rejected": True, "gold_fits_budget": True, "old_anchors_resolve": True}


def run_group(base, output, path, strategy, purpose, cases):
    snapshot = KnowledgeSnapshot(path, "static", None, purpose=purpose)
    with application(base, path, purpose, strategy, output) as api:
        results = asyncio.run(
            run_requests(
                str(api.base_url), [(c["id"], request_for(c)) for c in cases], base.task_concurrency
            )
        )
    write(output / "tasks.json", results)
    return [label(c, results[c["id"]], snapshot) for c in cases]


def date_blind_control(snapshots, cases):
    snapshot = KnowledgeSnapshot(snapshots / "versions.json", "static", None, purpose="test")
    all_versions = {
        c["chunk_id"]: c
        for c in snapshot.chunks
        if c["database_id"] == DATABASE and c["approval_ref"]
    }
    results = []
    for case in cases:
        if not case["id"].endswith("-natural"):
            continue
        active = snapshot.active(DATABASE, case["as_of"])
        rows = {}
        for name, eligible in (("hard_filter", active), ("date_blind", all_versions)):
            rank = lexical_rank(eligible, case["question"])
            rows[name] = {
                "top": rank[:4],
                "ineligible": len(set(rank[:4]) - active.keys()),
            }
        results.append({"id": case["id"], **rows})
    return results


async def direct_diagnostics(base, snapshots, dataset, output):
    positive = next(c for c in dataset["cases"] if c["outcome"] == "answered")
    snapshot = KnowledgeSnapshot(snapshots / "product.json", "static", None)
    fixture = KnowledgeSnapshot(snapshots / "fixtures.json", "static", None, purpose="test")
    active = snapshot.active(DATABASE, snapshot.as_of)
    requests = [
        ("gold", positive["question"], [active[x["chunk_id"]] for x in positive["supports"]])
    ]
    requests.extend((c["id"] + "-model-empty", c["question"], []) for c in dataset["dependencies"])
    records = []
    async with open_task_service(base) as service:
        for case_id, question, chunks in requests:
            knowledge = KnowledgeService(
                snapshot if case_id == "gold" else fixture, service.agent.model
            )
            report = RunReport(case_id, {"question": question})

            async def emit(kind, payload, report=report):
                report.events.append({"type": kind, "payload": payload})

            error_code = None
            try:
                outcome, citations, checks = await knowledge.evaluate(
                    question,
                    chunks,
                    report=report,
                    emit=emit,
                    journal=None,
                    deadline=time.monotonic() + 25,
                )
            except AgentError as error:
                outcome, citations, checks = None, [], []
                error_code = error.code
            if case_id == "gold":
                passed = outcome == "answered" and all(
                    all(
                        phrase
                        in "\n".join(
                            citation["quote"]
                            for citation in citations
                            if citation["chunk_id"] == item["chunk_id"]
                        )
                        for phrase in item["must"]
                    )
                    for item in positive["supports"]
                )
            else:
                passed = outcome == "insufficient_evidence" and not citations
            records.append(
                {
                    "id": case_id,
                    "passed": passed,
                    "outcome": outcome,
                    "citations": citations,
                    "coverage": checks,
                    "error": error_code,
                    "requests": [
                        {k: event["payload"][k] for k in ("request_id", "usage")}
                        for event in report.events
                        if event["type"] == "MODEL_COMPLETED"
                    ],
                }
            )
    write(output / "direct-model.json", records)
    return {r["id"]: r["passed"] for r in records}


async def sql_regression(base, endpoint, n, output):
    source = read(ROOT / "data/processed/business_orders_v1/questions.json")
    questions = source[:12] if n == 12 else source[12:36]
    assert len(questions) == n
    gold = {c["id"]: c for c in read(ROOT / "data/gold/business_orders_v1/answers.json")}
    results = await run_requests(
        endpoint,
        [
            (q["id"], {"database_id": q["database_id"], "question": q["question"]})
            for q in questions
        ],
        base.task_concurrency,
    )
    evaluator = SQLEvaluator({k: db.path for k, db in base.databases.items()})
    judged = []
    try:
        for q in questions:
            row = results[q["id"]]
            report = row["report"] or {}
            reference = gold[q["id"]]
            score = await evaluator.evaluate(
                q["database_id"], report.get("sql"), reference["sql"], ordered=reference["ordered"]
            )
            isolated = "knowledge" not in report and not any(
                e["type"].startswith("KNOWLEDGE_") for e in row["trace"]
            )
            judged.append(
                {
                    "id": q["id"],
                    "score": score,
                    "isolated": isolated,
                    "passed": isolated
                    and row["status"] == "COMPLETED"
                    and score["status"] == "correct",
                }
            )
    finally:
        evaluator.close()
    write(output / "sql-regression.json", {"judged": judged, "tasks": results})
    return {
        "n": len(judged),
        "passed": sum(row["passed"] for row in judged),
        "jct_p95": p95([r["jct_seconds"] for r in results.values()]),
        "failed": [r["id"] for r in judged if not r["passed"]],
    }


def qualified(stat, threshold):
    return (
        stat["evidence_complete"] >= threshold
        and stat["grounded_by_labels"] >= threshold
        and stat["negative_correct"] == stat["negative_n"]
        and not stat["safety_errors"]
        and stat["diagnostic_passed"] == stat["diagnostic_n"]
        and stat["jct_p95"] <= 15
    )


def run(stage, output, registration, *, observed_replay=False):
    output.mkdir(parents=True)
    sources, inputs = fingerprint(), frozen_inputs()
    if registration:
        assert registration["sources"] == sources and registration["inputs"] == inputs
        assert registration["protocol"] == PROTOCOL
    dataset = read(
        EVALUATION / ("development.json" if stage == "development" else "contract-acceptance.json")
    )
    if stage == "development":
        dataset["regression_conflicts"] = [
            case for case in dataset["cases"] if case.get("scenario") == "fixture"
        ]
        development_cohort = (
            "scoped-development.json" if observed_replay else "contract-development.json"
        )
        dataset["cases"] = read(EVALUATION / development_cohort)
    n = 12 if stage == "development" else 24
    assert len(dataset["cases"]) == 2 * n
    base = Settings.load(CONFIG)
    db_before = {k: sha(db.path) for k, db in base.databases.items()}
    summary = {
        "stage": stage,
        "protocol": PROTOCOL,
        "sources": sources,
        "inputs": inputs,
        "arms": {},
        "database_sha256": db_before,
        "state": "started",
        "observed_replay_only": observed_replay,
    }
    write(output / "summary.json", summary)
    try:
        with ExitStack() as stack:
            stack.enter_context(monitor(output))
            cpu = stack.enter_context(
                http_server(
                    "memory_resources:create_app",
                    output,
                    "cpu",
                    app_dir=ROOT / "scripts/experiments",
                    env={
                        "AGENT_EMBEDDING_PATH": str(ROOT / "models/Qwen3-Embedding-0.6B"),
                        "AGENT_TOKENIZER_PATH": str(ROOT / "models/Qwen2.5-7B-Instruct"),
                        "CUDA_VISIBLE_DEVICES": "",
                    },
                    python_executable=os.environ.get("VLLM_PYTHON", str(Path.home() / "miniconda3/envs/agent-vllm/bin/python")),
                    readiness_seconds=120,
                )
            )
            summary["resources"] = cpu.get("/info").raise_for_status().json()
            if registration:
                assert registration["resources"] == summary["resources"]
                assert registration["database_sha256"] == db_before
            base.memory.resources_url = str(cpu.base_url).rstrip("/")
            base.memory.resources_identity = summary["resources"]["identity"]
            snapshots = output / "snapshots"
            asyncio.run(
                snapshot_inputs(
                    base.memory.resources_url, base.memory.resources_identity, snapshots
                )
            )
            summary["snapshots"] = {
                k: sha(snapshots / f"{k}.json") for k in ("product", "fixtures")
            }
            if registration:
                assert registration["snapshots"] == summary["snapshots"]
            plans = challenge_snapshots(dataset, snapshots / "fixtures.json", snapshots)
            summary["preflight"] = preflight(dataset, snapshots, plans, cpu)
            summary["date_blind"] = date_blind_control(snapshots, plans["versions"])
            base.base_url = stack.enter_context(
                temporary_model(output, max_model_len=8192, max_num_seqs=4, verify_gpu=True)
            )
            summary["direct"] = asyncio.run(direct_diagnostics(base, snapshots, dataset, output))
            strategies = registration["arms"] if registration else ARMS
            for strategy in strategies:
                print(json.dumps({"state": "running", "strategy": strategy}), flush=True)
                directory = output / strategy
                cases = [c for c in dataset["cases"] if "scenario" not in c]
                rows = run_group(
                    base,
                    directory / "product",
                    snapshots / "product.json",
                    strategy,
                    "product",
                    cases,
                )
                diagnostic = []
                main_ids = {c["id"] for c in dataset["cases"]}
                for name, probe in plans.items():
                    result = run_group(
                        base, directory / name, snapshots / f"{name}.json", strategy, "test", probe
                    )
                    rows.extend(item for item in result if item["id"] in main_ids)
                    diagnostic.extend(item for item in result if item["id"] not in main_ids)
                stat = metrics(rows)
                if stage == "development" and not observed_replay:
                    observed = read(EVALUATION / "scoped-development.json")
                    replay = run_group(
                        base,
                        directory / "observed-replay",
                        snapshots / "product.json",
                        strategy,
                        "product",
                        observed,
                    )
                    stat["observed_replay"] = metrics(replay)
                stat.update(
                    diagnostic_n=len(diagnostic),
                    diagnostic_passed=sum(r["passed"] for r in diagnostic),
                )
                stat["eligible_by_labels"] = qualified(stat, 11 if stage == "development" else 22)
                write(directory / "labels.json", {"main": rows, "diagnostic": diagnostic})
                summary["arms"][strategy] = stat
                write(output / "summary.json", summary)
            with application(
                base, snapshots / "product.json", "product", "static", output / "sql-regression"
            ) as api:
                summary["sql_regression"] = asyncio.run(
                    sql_regression(base, str(api.base_url), n, output)
                )
            if registration:
                from v31_recovery import probe_recovery

                summary["recovery"] = probe_recovery(
                    base, snapshots, registration["candidate"] or "static", output / "recovery"
                )
            else:
                candidates = [arm for arm in ARMS[1:] if summary["arms"][arm]["eligible_by_labels"]]
                stats = summary["arms"]
                winner = min(
                    candidates,
                    key=lambda a: (
                        -stats[a]["grounded_by_labels"],
                        -stats[a]["evidence_complete"],
                        stats[a]["evidence_tokens_median"],
                        stats[a]["retrieval_p95"],
                        ARMS.index(a),
                    ),
                    default=None,
                )
                summary["candidate"] = winner
            summary["cpu_metrics"] = cpu.get("/metrics").raise_for_status().json()
        summary["state"] = "completed"
    finally:
        summary["databases_unchanged"] = db_before == {
            k: sha(db.path) for k, db in base.databases.items()
        }
        summary["owned_survivors"] = verify_owned_services(output)
        write(output / "summary.json", summary)


def register(development, output):
    summary = read(development / "summary.json")
    assert summary["state"] == "completed" and not summary["owned_survivors"]
    assert not summary["observed_replay_only"], "Observed cases cannot qualify a new candidate"
    assert summary["protocol"] == PROTOCOL
    assert summary["databases_unchanged"] and all(summary["preflight"].values())
    assert summary["sources"] == fingerprint() and summary["inputs"] == frozen_inputs()
    assert summary["sql_regression"]["n"] == summary["sql_regression"]["passed"]
    assert all(summary["direct"].values())
    candidate = summary["candidate"]
    assert candidate is not None or summary["arms"]["static"]["eligible_by_labels"]
    write(
        output,
        {
            **{
                k: summary[k]
                for k in ("sources", "inputs", "resources", "database_sha256", "snapshots")
            },
            "development_summary_sha256": sha(development / "summary.json"),
            "protocol": PROTOCOL,
            "candidate": candidate,
            "arms": ["static", candidate] if candidate else ["static"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "register", "acceptance"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--observed-replay", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if args.observed_replay and args.stage != "development":
        parser.error("Observed replay is only available during development")
    if args.stage == "register":
        if args.development is None:
            parser.error("Registration requires a completed development run")
        register(args.development, args.output)
    else:
        if (args.stage == "acceptance") != (args.registration is not None):
            parser.error("Acceptance requires a frozen registration")
        run(
            args.stage,
            args.output,
            read(args.registration) if args.registration else None,
            observed_replay=args.observed_replay,
        )


if __name__ == "__main__":
    main()
