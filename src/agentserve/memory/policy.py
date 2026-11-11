"""SimpleMem mechanism reference and SQL episode adaptation (not upstream reproduction)."""

import json
from typing import Protocol

from pydantic import ValidationError

from agentserve.domain.models import AgentError
from agentserve.memory.models import (
    ContextPack,
    EncodedMemory,
    EpisodeMemory,
    FactMemory,
    RetrievalPlan,
)
from agentserve.memory.retrieval import retrieve

PLAN_PROMPT = """Plan retrieval for the current read-only SQL question. Return only JSON:
{"action":"new|refine|resume|clarify","search_query":"standalone retrieval query",
"keywords":["..."],"topic":"...","depth":3,"target_id":null,"changes":[],"clarification":null}.
Use new for an independent analysis, refine for changing the current analysis, resume for returning
to an earlier analysis. target_id must be a supplied candidate entry_id or null. Never invent a source.
Use clarify with a concrete Chinese question if a reference cannot be resolved from visible evidence.
Current explicit conditions override history. History SQL/answers are observations, not verified facts.
changes lists only explicit modifications. Do not turn temporary conditions into session-wide rules.
Keep analysis groups separate. Do not confuse conversation date and business date. depth is 1..20.
search_query and changes must use natural language business concepts, never generated SQL or guessed
table/column names. No database schema is supplied to you. Clarify only genuinely missing references
or business constraints; do not ask to confirm explicit user conditions or schema information.
No reasoning prose. All fields are required."""

FACT_PROMPT = """Encode this completed turn as up to four self-contained factual memories.
Resolve references only using the supplied context; preserve dates, negation, units, entities and scope.
Semantically filter redundancy and synthesize related facts within this turn. Do not repeat old context.
Historical SQL and answers are system observations, not guaranteed correct facts or current totals.
Never use historical numeric results as answers to new database questions.
Return only JSON {"entries":[{"text":"...","keywords":["..."],"topic":"...",
"event_time":null,"frame":null}]}. Do not generalize a one-time filter into a permanent preference."""

EPISODE_PROMPT = """Encode exactly one query episode from this turn. Bind the intended question,
conditions, date range, metric, grouping and units in one event. Use user intent as evidence;
SQL is an unverified system observation. For refine/resume inherit only the identified source episode,
apply explicit changes, and keep unrelated analyses separate. Current explicit conditions win.
Return only JSON {"entries":[{"text":"self-contained intended analysis","keywords":["..."],
"topic":"...","event_time":null,"frame":{"request":"standalone question",
"conditions":["..."],"date_range":null,"metric":null,"group_by":[],"unit":null}}]}.
No historical numeric result may substitute for a fresh query. Never invent missing conditions."""


class MemoryPolicy(Protocol):
    async def prepare(
        self, task, inputs, generate, resources, settings, deadline
    ) -> ContextPack: ...
    async def encode(self, observation, context, generate) -> EncodedMemory: ...


def visible(entry):
    return {k: v for k, v in entry.items() if k != "vector"}


async def pack_context(
    inputs, selected, plan, version, resources, settings, deadline, messages=None
):
    base = {
        "recent_turns": inputs["recent"],
        "retrieved_memories": [],
        "accepted_at_unix": inputs.get("accepted_at"),
    }
    # Retrieval text is planner speculation, not evidence for the SQL generator.
    # Keep it in the audited plan but never inject invented SQL/columns into generation.
    if plan and version.startswith("sql-episode"):
        base.update(query_action=plan.action, target_id=plan.target_id)
        if "updates1" in version:
            base["explicit_changes"] = plan.changes
    text = json.dumps(base, ensure_ascii=False)
    count = await resources.count(text, deadline)
    if count > settings.context_tokens:
        raise AgentError("MEMORY_CONTEXT_LIMIT", "Recent required turns exceed memory budget")
    accepted = []
    for entry in selected:
        candidate = visible(entry)
        trial = json.dumps(
            {**base, "retrieved_memories": accepted + [candidate]}, ensure_ascii=False
        )
        tokens = await resources.count(trial, deadline)
        if tokens <= settings.context_tokens:
            accepted.append(candidate)
            text, count = trial, tokens
        elif plan and entry["entry_id"] == plan.target_id:
            raise AgentError(
                "MEMORY_CONTEXT_LIMIT", "Required referenced episode exceeds memory budget"
            )
    return ContextPack(
        text,
        [r["task_id"] for r in inputs["recent"]] + [e["entry_id"] for e in accepted],
        inputs["cutoff_turn"],
        inputs["revision"],
        version,
        count,
        plan.model_dump() if plan else None,
        accepted,
        messages or [],
    )


class SimpleMemPolicy:
    version = "simplemem-core-v1"
    event_binding = False
    condition_updates = False

    async def prepare(self, task, inputs, generate, resources, settings, deadline):
        vector = (await resources.embed([task.question], query=True, deadline=deadline))[0]
        preliminary = retrieve(inputs["entries"], vector, task.question, [], "", 3)
        recent_ids = {r["task_id"] for r in inputs["recent"]}
        candidates = {e["entry_id"]: e for e in inputs["entries"] if e["task_id"] in recent_ids}
        candidates.update({e["entry_id"]: e for e in preliminary})
        shown = list(candidates.values())[:10]
        messages = [
            {"role": "system", "content": PLAN_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": task.question,
                        "recent_turns": inputs["recent"],
                        "candidates": [visible(e) for e in shown],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        plan = await generate(messages, RetrievalPlan, "memory_plan")
        if plan.action == "clarify":
            raise AgentError(
                "MEMORY_CONTEXT_REQUIRED", plan.clarification or "请明确要继续哪一组分析。"
            )
        if plan.target_id and plan.target_id not in {e["entry_id"] for e in shown}:
            raise AgentError("MEMORY_SOURCE_INVALID", "Planner selected an unavailable source")
        if plan.action in {"refine", "resume"} and not plan.target_id and not inputs["recent"]:
            raise AgentError("MEMORY_CONTEXT_REQUIRED", "请补充要继续分析的条件或来源。")
        vector = (await resources.embed([plan.search_query], query=True, deadline=deadline))[0]
        ranked = retrieve(
            inputs["entries"], vector, plan.search_query, plan.keywords, plan.topic, plan.depth
        )
        ranked = self.bind(ranked, inputs["entries"], plan)
        return await pack_context(
            inputs, ranked, plan, self.version, resources, settings, deadline, messages
        )

    def bind(self, ranked, entries, plan):
        return ranked

    async def encode(self, observation, context, generate):
        return await generate(
            [
                {"role": "system", "content": FACT_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"turn": observation, "prior_context": context.text}, ensure_ascii=False
                    ),
                },
            ],
            FactMemory,
            "memory_write",
        )


class SQLEpisodePolicy(SimpleMemPolicy):
    def __init__(self, binding=True, updates=True):
        self.event_binding, self.condition_updates = binding, updates
        self.version = f"sql-episode-v1-binding{int(binding)}-updates{int(updates)}"

    def bind(self, ranked, entries, plan):
        if not self.event_binding or not plan.target_id:
            return ranked
        target = next(e for e in entries if e["entry_id"] == plan.target_id)
        if self.condition_updates and plan.action == "refine":
            for entry in entries:
                if entry.get("update_of") == target["entry_id"]:
                    target = entry
            plan.target_id = target["entry_id"]
        return [target] + [e for e in ranked if e["entry_id"] != target["entry_id"]]

    async def encode(self, observation, context, generate):
        if not self.event_binding:
            return await super().encode(observation, context, generate)
        result = await generate(
            [
                {"role": "system", "content": EPISODE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"turn": observation, "plan": context.plan, "prior_context": context.text},
                        ensure_ascii=False,
                    ),
                },
            ],
            EpisodeMemory,
            "memory_write",
        )
        if len(result.entries) != 1 or result.entries[0].frame is None:
            raise AgentError(
                "MEMORY_PROTOCOL_ERROR", "Episode writer must return exactly one frame"
            )
        return result


def policy_for(name):
    if name in {"state", "state_on_demand"}:
        from agentserve.memory.state import StatePolicy

        return StatePolicy(on_demand=name == "state_on_demand")
    if name == "simplemem":
        return SimpleMemPolicy()
    return SQLEpisodePolicy(binding=name != "no_binding", updates=name != "no_updates")


def parse_output(reply, response_type):
    try:
        return response_type.model_validate_json(reply.message.get("content") or "")
    except ValidationError as error:
        raise AgentError("MEMORY_PROTOCOL_ERROR", str(error)) from error
