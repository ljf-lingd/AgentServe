"""Effective execution identity persisted with tasks; no new state or migration."""

import hashlib
import json
from pathlib import Path

from agentserve.agents import sql_prompt
from agentserve.agents.answer_policy import (
    BOUNDED_TEMPLATE,
    CONCISE_STYLES,
    DIRECT_EVIDENCE_STYLE,
    EVIDENCE_STYLES,
)
from agentserve.agents.retail_sql import RETAIL_DATABASE
from agentserve.config import Settings
from agentserve.llm.client import SQL_TOOLS
from agentserve.tools.sqlite import FUNCTION_PROFILES

# The V7.1 SQL agent changes only how the same retail date bounds are executed.
# Keep existing sessions compatible with this exact revision. Pending tasks
# still compare the actual SQL agent fingerprint before resuming execution.
_SESSION_COMPATIBLE_SQL_AGENT = {
    "68e66a0e950ac7f874948874f9158f359e7766f9f1a1f3b12d1bf523d7411330":
        "31dc7f292c934a2ebb3ceeb8f7fee8a0a4d8dd02c8c12da114c7f8d6a1d4f0af",
}


def execution_snapshot(settings: Settings) -> dict:
    prompt = json.dumps(
        [
            sql_prompt.initial_messages(
                "",
                schema="",
                context="",
                version=settings.sql_prompt_version,
                functions=sorted(FUNCTION_PROFILES[settings.sql_function_profile]),
            ),
            sql_prompt.FORMAT_REPAIR_PROMPT,
            sql_prompt.CONCISE_ANSWER_PROMPT
            if settings.answer_style in CONCISE_STYLES or settings.session_sam
            else None,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    contract = json.dumps(SQL_TOOLS, sort_keys=True, separators=(",", ":"))
    snapshot = {
        **settings.model_dump(mode="json"),
        "execution_identity": {
            "workflow_version": "sql-agent-v1",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "tool_contract_sha256": hashlib.sha256(contract.encode()).hexdigest(),
            "sql_policy_version": "sqlite-readonly-v1",
            "allowed_functions": sorted(FUNCTION_PROFILES[settings.sql_function_profile]),
            "answer_policy_sha256": hashlib.sha256(BOUNDED_TEMPLATE.encode()).hexdigest()
            if settings.answer_style == "bounded_v1"
            else None,
            "schema_sha256": {
                key: hashlib.sha256(db.schema_path.read_bytes()).hexdigest()
                for key, db in settings.databases.items()
            },
        },
    }
    coverage = {
        key: [db.workbench.first_date.isoformat(), db.workbench.last_date.isoformat()]
        for key, db in settings.databases.items() if db.workbench
    }
    if coverage:
        snapshot["execution_identity"]["dataset_coverage"] = coverage
    if RETAIL_DATABASE in settings.databases:
        retail_policy = Path(__file__).resolve().parents[1] / "agents/retail_sql.py"
        snapshot["execution_identity"]["retail_date_policy_sha256"] = hashlib.sha256(
            retail_policy.read_bytes()
        ).hexdigest()
    if settings.knowledge is None:
        snapshot.pop("knowledge")
    if settings.metric is None:
        snapshot.pop("metric")
    if not settings.analysis_enabled:
        snapshot.pop("analysis_enabled")
    if not settings.session_sam:
        snapshot.pop("session_sam")
    if settings.session_sam_all_phases:
        root = Path(__file__).resolve().parents[1]
        snapshot["execution_identity"]["phase_sam_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("llm/client.py", "storage/steps.py", "application/service.py")
        }
    else:
        snapshot.pop("session_sam_all_phases")
    if settings.memory:
        snapshot["execution_identity"]["workflow_version"] = "sql-agent-memory-v2-repair-v1"
        memory_root = Path(__file__).resolve().parents[1] / "memory"
        material = {
            "settings": settings.memory.model_dump(mode="json"),
            "source": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(memory_root.glob("*.py"))
            },
        }
        sql_agent_sha256 = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "agents/sql.py").read_bytes()
        ).hexdigest()
        material["sql_agent_sha256"] = _SESSION_COMPATIBLE_SQL_AGENT.get(
            sql_agent_sha256, sql_agent_sha256
        )
        if material["sql_agent_sha256"] != sql_agent_sha256:
            snapshot["execution_identity"]["sql_agent_sha256"] = sql_agent_sha256
        material["model_execution_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "llm/execution.py").read_bytes()
        ).hexdigest()
        snapshot["execution_identity"]["memory_policy_sha256"] = hashlib.sha256(
            json.dumps(material, sort_keys=True).encode()
        ).hexdigest()
    else:
        snapshot.pop("memory", None)  # Preserve the V1 execution snapshot contract.
    if settings.answer_style in EVIDENCE_STYLES:
        snapshot["execution_identity"]["workflow_version"] = "sql-agent-v21-state-1"
        snapshot["execution_identity"]["answer_policy_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "agents/answer_policy.py").read_bytes()
        ).hexdigest()
    if settings.sql_prompt_version in {"evidence_v21", "evidence_v22"}:
        snapshot["execution_identity"]["sql_intent_policy_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "agents/sql_policy.py").read_bytes()
        ).hexdigest()
    if settings.sql_prompt_version == "evidence_v22":
        snapshot["execution_identity"]["workflow_version"] = (
            "sql-agent-v221-order-sql-1"
            if settings.answer_style == "evidence_v221"
            else "sql-agent-v22-order-sql-1"
        )
        snapshot["execution_identity"]["order_compiler_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "agents/order_sql.py").read_bytes()
        ).hexdigest()
    if settings.answer_style == DIRECT_EVIDENCE_STYLE:
        snapshot["execution_identity"]["workflow_version"] = "sql-agent-v5-direct-evidence-1"
    return snapshot
