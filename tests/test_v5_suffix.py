import asyncio
import json
from pathlib import Path

import httpx
from conftest import completion
from test_memory import resources

from agentserve.config import MemorySettings


async def test_suffix_counter_uses_server_root(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    from vllm_suffix_probe import counters

    def respond(request):
        if request.url.path != "/metrics":
            return httpx.Response(404)
        return httpx.Response(200, text='vllm:spec_decode_num_draft_tokens_total{model_name="agent-qwen"} 7\n')

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(respond),
    ) as client:
        assert await counters(client) == {"vllm:spec_decode_num_draft_tokens_total": 7}


def test_feedback_trims_native_prefix_using_actual_verifier_work(monkeypatch):
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "serving/vllm"))
    from suffix_feedback import attach_feedback

    batch = SimpleNamespace(req_ids=["rejected", "accepted"])
    drafter = SimpleNamespace(propose=lambda *_: [[11, 12, 13, 14], [21, 22, 23, 24]])
    runner = SimpleNamespace(input_batch=batch, drafter=drafter, requests=dict.fromkeys(batch.req_ids))
    runner.propose_draft_token_ids = lambda *args: drafter.propose(4, batch, args[1])
    attach_feedback(runner, 4)
    metadata = SimpleNamespace(num_draft_tokens=[4, 4])
    for _ in range(26):
        drafts = runner.propose_draft_token_ids(None, [[0], [1, 2, 3, 4, 5]], None, None,
                                               None, None, metadata, None, None)
    assert drafts == [[11, 12, 13], [21, 22, 23, 24]]
    runner.requests.clear()
    assert runner.propose_draft_token_ids(None, [[], []], None, None, None, None, None, None, None) == [
        [11, 12, 13, 14], [21, 22, 23, 24],
    ]


async def test_c4_retail_pairs_shuffled_engine_replies_with_their_tasks(settings, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    from retail_c4 import cases, reference_key, replay
    from retail_replay import DATABASE, workloads

    old_keys = {reference_key(turn) for cohort in ("development", "formal")
                for session in workloads(cohort) for turn in session}
    new_keys = {reference_key(turn) for session in cases() for turn in session}
    assert len(new_keys) == 72 and new_keys.isdisjoint(old_keys)
    settings.databases[DATABASE] = settings.databases.pop("demo_orders")
    settings.answer_style = "evidence_direct_v5"
    settings.memory = MemorySettings(resources_identity="test-pinned")
    returned = []
    pending = set()
    batch_ready = asyncio.Event()

    async def answer(request):
        question = next(m["content"] for m in json.loads(request.content)["messages"]
                        if m["role"] == "user" and m["content"].startswith("Count customers"))
        token = int(question.split()[-1])
        if token < 4:
            pending.add(token)
            if len(pending) == 4:
                batch_ready.set()
            async with asyncio.timeout(2):
                await batch_ready.wait()
            await asyncio.sleep((4 - token) * .015)
        result = completion(sql="SELECT COUNT(*) FROM customers")
        result["id"], result["prompt_token_ids"] = f"mock-{token}", [token]
        result["choices"][0]["token_ids"] = [token + 100]
        returned.append(token)
        return httpx.Response(200, json=result)

    turns = [{"question": f"Count customers {token}", "expected": [[2]], "ordered": False}
             for token in range(5)]
    sessions, elapsed, attempts = await replay(
        settings, "ar", [[turn] for turn in turns[:4]], turns[-1], tmp_path,
        model_transport=httpx.MockTransport(answer), resource_transport=httpx.MockTransport(resources),
    )
    assert attempts == 4 and elapsed > 0
    assert returned[0] == 4 and returned[1:] != list(range(4))
    for token, session in enumerate(sessions):
        (turn,) = session["turns"]
        assert turn["correct"] and turn["trace"][0]["prompt"] == [token]
