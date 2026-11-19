"""Compare one failed knowledge question with the workbench's existing clause scope."""

import argparse
import asyncio
import json
import signal
from pathlib import Path

import v5_engine
from agentserve.knowledge.snapshot import KnowledgeSnapshot
from run_v5_phase import interrupt
from v5_expansion_eval import grade
from v5_expansion_live import run
from v6_phase_benchmark import MODEL, ROOT, Settings, dump, file_sha, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("clause_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    matrix = json.loads((args.matrix / "completed.json").read_bytes())["identity"]
    manifest = json.loads(args.manifest.read_bytes())
    config = ROOT / "configs/app.v5.json"
    settings = Settings.load(config)
    if (file_sha(args.manifest) != matrix["manifest_sha256"] or args.gpu_uuid != matrix["gpu_uuid"]
            or file_sha(args.parent) != matrix["parent_sha256"] or plan(config, settings) != json.loads(args.parent.read_bytes())
            or v5_engine.environment(args.vllm_python) != matrix["environment"]
            or v5_engine.verify_model(args.model_dir) != matrix["model_snapshot"]):
        raise ValueError("Replay requires the same application data, serving environment and GPU as the matrix")
    case = next(row for row in manifest["workload"]["knowledge"] if row["id"] == args.case_id)
    scoped = {**case, "id": case["id"] + "-scoped", "request": {**case["request"], "clause_id": args.clause_id},
              "label": {**case["label"], "clause_id": args.clause_id}}
    snapshot = KnowledgeSnapshot(settings.knowledge.snapshot_path, "static", None)
    output = args.output.resolve()
    output.mkdir()
    settings.model = MODEL
    settings.session_sam = settings.session_sam_all_phases = False
    settings.state_database = output / "tasks.sqlite"
    settings.task_concurrency = settings.llm_concurrency = 1
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)
    with (
        v5_engine.resources(output, args.vllm_python, args.model_dir, args.embedding_dir) as resources,
        v5_engine.engine(output, "ar", args.model_dir, args.vllm_python, args.gpu_uuid) as url,
    ):
        settings.base_url, settings.memory.resources_url = url, resources
        rows, _, calls = asyncio.run(run(settings, [
            ("knowledge", "original", [case]), ("knowledge", "scoped", [scoped]),
        ], manifest["business_time"], output))
    effects = {}
    for variant, item in (("original", case), ("scoped", scoped)):
        row = rows[f"knowledge:{item['id']}"]
        knowledge = row["task"]["report"]["knowledge"]
        effects[variant] = {"task_id": row["task"]["task_id"], "issue": grade(item, row, snapshot),
                            "outcome": knowledge["outcome"], "candidate_ids": knowledge["candidate_ids"],
                            "retrieved_ids": knowledge["retrieved_ids"],
                            "citation_ids": [citation["chunk_id"] for citation in knowledge["citations"]],
                            "request_ids": [attempt["output"]["reply"]["request_id"]
                                            for attempt in row["journal"]["steps"] if attempt["kind"] == "model"]}
    result = {"matrix_manifest_sha256": matrix["manifest_sha256"], "case_id": case["id"],
              "clause_id": args.clause_id, "expected": case["label"]["supports"],
              "effects": effects, "actual_model_attempts": len(calls),
              "source_sha256": file_sha(Path(__file__))}
    dump(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
