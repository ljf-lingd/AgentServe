"""Real CPU-resource equivalence checks and a compact offline source-chain audit."""

import json
from pathlib import Path


def verify_resources(client):
    cases = [
        {"texts": ["2026年2月 Shanghai completed 营收分", "同一个文本"]},
        {
            "messages": [
                {"role": "system", "content": "Return one read-only SQL."},
                {"role": "user", "content": "统计客户数量"},
            ]
        },
        {
            "messages": [{"role": "user", "content": "查询营收"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
                    },
                }
            ],
        },
    ]
    checks = []
    for route, requests in [
        ("count", cases),
        (
            "embed",
            [{"texts": ["营收统计"], "query": False}, {"texts": ["营收统计"], "query": True}],
        ),
    ]:
        for body in requests:
            first = client.post("/" + route, json=body)
            first.raise_for_status()
            hit = client.post("/" + route, json=body)
            hit.raise_for_status()
            direct = client.post("/" + route, json={**body, "bypass_cache": True})
            direct.raise_for_status()
            assert first.json() == hit.json() == direct.json(), (route, body)
            checks.append({"route": route, "body": body, "cached_equal_uncached": True})
    return {"checks": checks, "metrics": client.get("/metrics").json()}


def audit_old(root):
    old = root / "results/v2-memory/acceptance-20261103T085452Z"
    dest = root / "results/v2.1-memory/diagnosis-20261104"
    methods = ["window", "simplemem", "sql_episode"]
    records = {
        p: {r["id"]: r for r in map(json.loads, (old / p / "turns.jsonl").open())} for p in methods
    }
    inheritance = []
    chains = []
    discordance = []
    for case_id, f in records["sql_episode"].items():
        if "inheritance" in f["tags"]:
            row = {"id": case_id, "question": f["question"], "methods": {}}
            for method in methods:
                r = records[method][case_id]
                ctx = r["context"]
                sql_steps = [
                    s
                    for s in r["steps"]["steps"]
                    if s["kind"] == "model" and s["input"].get("query")
                ]
                original = records[method][case_id[:-2] + "01"]["question"]
                row["methods"][method] = {
                    "task_id": r["task"]["task_id"],
                    "status": r["evaluation"]["status"],
                    "error": r["task"]["error"],
                    "plan": ctx["plan"] if ctx else None,
                    "original_in_actual_sql_input": bool(
                        sql_steps
                        and original
                        in json.dumps(sql_steps[0]["input"]["messages"], ensure_ascii=False)
                    ),
                    "frames": [e.get("frame") for e in (ctx or {}).get("selected", [])],
                    "sql": (r["task"]["report"] or {}).get("sql"),
                }
            inheritance.append(row)
        if "long_reference" in f["tags"]:
            current = f["task"]["turn_no"]
            length = max(
                r["task"]["turn_no"]
                for r in records["sql_episode"].values()
                if r["conversation"] == f["conversation"]
            )
            source = 3 if current == length - 6 else length - 5
            row = {
                "id": case_id,
                "current_turn": current,
                "source_turn": source,
                "distance": current - source,
                "methods": {},
            }
            for method in methods:
                r = records[method][case_id]
                src = next(
                    x
                    for x in records[method].values()
                    if x["conversation"] == r["conversation"] and x["task"]["turn_no"] == source
                )
                ctx = r["context"] or {}
                plan_steps = [
                    s for s in r["steps"]["steps"] if s["input"].get("purpose") == "memory_plan"
                ]
                shown = (
                    json.loads(plan_steps[0]["input"]["messages"][-1]["content"]).get(
                        "candidates", []
                    )
                    if plan_steps
                    else []
                )
                row["methods"][method] = {
                    "source_task_id": src["task"]["task_id"],
                    "source_question": src["question"],
                    "source_completed": src["task"]["status"] == "COMPLETED",
                    "source_drafts": (src["task"]["report"] or {}).get("memory_draft"),
                    "source_in_planner_candidates": any(
                        e["task_id"] == src["task"]["task_id"] for e in shown
                    ),
                    "candidate_count": len(shown),
                    "source_in_final_context": any(
                        e["task_id"] == src["task"]["task_id"] for e in ctx.get("selected", [])
                    ),
                    "final_count": len(ctx.get("selected", [])),
                    "plan": ctx.get("plan"),
                    "final_entries": ctx.get("selected", []),
                    "sql": (r["task"]["report"] or {}).get("sql"),
                    "evaluation": r["evaluation"],
                    "error": r["task"]["error"],
                    "correct_version_completeness": "requires_semantic_review_not_inferred_from_source_hit",
                }
            chains.append(row)
        if not f["tags"]:
            b = records["window"][case_id]
            if (b["evaluation"]["status"] == "correct") != (f["evaluation"]["status"] == "correct"):
                discordance.append(
                    {
                        "id": case_id,
                        "question": f["question"],
                        "window": b["evaluation"],
                        "sql_episode": f["evaluation"],
                        "sql_window": (b["task"]["report"] or {}).get("sql"),
                        "sql_episode_sql": (f["task"]["report"] or {}).get("sql"),
                    }
                )
    for name, value in [
        ("inheritance-36", inheritance),
        ("recovery-chain-72", chains),
        ("independent-discordance", discordance),
    ]:
        (dest / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    summary = {
        "inheritance": len(inheritance),
        "recovery": len(chains),
        "independent_discordance": len(discordance),
        "f1_inheritance_original_in_sql_input": sum(
            r["methods"]["sql_episode"]["original_in_actual_sql_input"] for r in inheritance
        ),
        "source_hit_is_not_condition_correctness": True,
    }
    (dest / "chain-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    audit_old(Path(__file__).resolve().parents[2])
