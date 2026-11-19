from pathlib import Path

import httpx


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
