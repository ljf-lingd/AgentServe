"""Application-owned context freezing and draft production; storage owns activation."""

import hashlib
import time
from dataclasses import asdict

from agentserve.domain.models import AgentError
from agentserve.memory.models import ContextPack, RetrievalPlan
from agentserve.memory.policy import pack_context, parse_output, policy_for
from agentserve.storage.memory import MemoryRepository


class MemoryRuntime:
    def __init__(self, model, resources, settings):
        self.model, self.resources, self.settings = model, resources, settings
        self.policy = None if settings.policy == "window" else policy_for(settings.policy)

    def generator(self, report, emit, journal, deadline):
        async def generate(messages, response_type, purpose):
            reply = await self.model.generate(
                messages,
                query=False,
                max_tokens=self.settings.plan_max_tokens
                if purpose == "memory_plan"
                else self.settings.write_max_tokens,
                deadline=deadline,
                report=report,
                emit=emit,
                journal=journal,
                purpose=purpose,
                json_schema=response_type.model_json_schema(),
            )
            return parse_output(reply, response_type)

        return generate

    async def prepare(self, task, report, emit, journal, deadline):
        if journal is None:
            raise AgentError("SESSION_REQUIRES_API", "Persistent sessions must use the Task API")
        store = MemoryRepository(journal.store.tasks)
        generate = self.generator(report, emit, journal, deadline)
        saved = await store.context(report.task_id)
        if saved:
            pack = ContextPack(**saved)
            # Reopen the committed planning step to keep the durable journal cursor aligned.
            # Reuse does not perform another model request or read newer session history.
            if pack.planning_messages:
                await generate(
                    pack.planning_messages,
                    getattr(self.policy, "plan_type", RetrievalPlan),
                    "memory_plan",
                )
            await emit("MEMORY_CONTEXT_REUSED", {"cutoff_turn": pack.cutoff_turn})
            return pack
        inputs = await store.inputs(report.task_id, self.settings.recent_turns)
        started = time.monotonic()
        if self.policy:
            pack = await self.policy.prepare(
                task, inputs, generate, self.resources, self.settings, deadline
            )
        else:
            pack = await pack_context(
                inputs, [], None, "window-v1", self.resources, self.settings, deadline
            )
            if self.model.settings.sql_prompt_version == "evidence_v22" and task.database_id in {
                "business_orders_v1",
                "demo_orders",
            }:
                from agentserve.memory.state import window_order_plan

                pack.plan = window_order_plan(task.question, inputs)
        await store.save_context(report.task_id, journal.version, asdict(pack))
        await emit(
            "MEMORY_PREPARED",
            {"client_seconds": time.monotonic() - started, "tokens": pack.token_count},
        )
        return pack

    async def encode(self, task, report, pack, emit, journal, deadline):
        if not self.policy:
            return []
        if pack.plan and pack.plan.get("route") == "recent_window":
            await emit(
                "MEMORY_WRITE_SKIPPED",
                {"reason": "unsupported_state_schema_recent_window", "raw_task_persisted": True},
            )
            return []
        started = time.monotonic()
        # No result rows: old revenue must never become an authoritative current answer.
        observation = {
            "question": task.question,
            "observed_sql": report.sql,
            "answer_is_unverified": True,
            "turn_no": pack.cutoff_turn + 1,
        }
        encoded = await self.policy.encode(
            observation, pack, self.generator(report, emit, journal, deadline)
        )
        if not encoded.entries:
            raise AgentError(
                "MEMORY_PROTOCOL_ERROR", "Writer returned no memory for a completed query"
            )
        if self.settings.policy in {"state", "state_on_demand"}:
            vectors = [[] for e in encoded.entries]
            for entry in encoded.entries:
                for field in entry.state["fields"].values():
                    if field["source_task_id"] is None:
                        field["source_task_id"] = report.task_id
        else:
            vectors = await self.resources.embed(
                [e.text for e in encoded.entries], query=False, deadline=deadline
            )
        drafts = []
        for index, (entry, vector) in enumerate(zip(encoded.entries, vectors, strict=True)):
            drafts.append(
                {
                    **entry.model_dump(),
                    "entry_id": hashlib.sha256(
                        f"{report.task_id}:{index}:{pack.policy_version}".encode()
                    ).hexdigest()[:24],
                    "task_id": report.task_id,
                    "turn_no": pack.cutoff_turn + 1,
                    "observed_sql": report.sql,
                    "source_ids": pack.source_ids,
                    "update_of": pack.plan["target_id"]
                    if self.policy.condition_updates
                    and pack.plan
                    and pack.plan["action"] == "refine"
                    else None,
                    "reference_to": pack.plan["target_id"] if pack.plan else None,
                    "vector": vector,
                }
            )
        await emit(
            "MEMORY_DRAFT_READY",
            {"count": len(drafts), "client_seconds": time.monotonic() - started},
        )
        return drafts
