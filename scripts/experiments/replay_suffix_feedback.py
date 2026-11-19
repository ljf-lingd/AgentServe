"""Verify native Suffix and feedback against archived Qwen token traces on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from suffix_feedback import DraftFeedback
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

CAP, MODEL_LEN = 4, 8192


def replay(requests, adaptive):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=CAP, suffix_decoding_max_tree_depth=24,
            suffix_decoding_max_spec_factor=1.0, suffix_decoding_min_token_prob=0.1,
            suffix_decoding_max_cached_requests=256,
        ),
        model_config=SimpleNamespace(max_model_len=MODEL_LEN),
    )
    native = SuffixDecodingProposer(config)
    total = Counter()
    for trace in requests:
        request_id = trace["id"]
        prompt, target = trace["prompt"], trace["tokens"]
        if not prompt or not target or len(prompt) + len(target) > MODEL_LEN:
            raise ValueError(f"Invalid archival tokens for {request_id}")
        feedback = DraftFeedback(CAP)
        batch = SimpleNamespace(
            req_ids=[request_id], req_id_to_index={request_id: 0}, num_prompt_tokens=[len(prompt)],
            token_ids_cpu=np.asarray([prompt + target], dtype=np.int32), num_tokens_no_spec=[0],
        )
        emitted = 1
        sampled = target[:1]
        total["target_steps"] += 1
        total["requests"] += 1
        total["output_tokens"] += len(target)
        while emitted < len(target):
            batch.num_tokens_no_spec[0] = len(prompt) + emitted
            (draft,) = native.propose(CAP, batch, [sampled])
            draft = draft[:max(0, trace["max_tokens"] - emitted)]
            if adaptive:
                trimmed = feedback.prefix(request_id, draft)
                total["trimmed_steps"] += len(trimmed) != len(draft)
                draft = trimmed
            accepted = 0
            for candidate, expected in zip(draft, target[emitted:]):
                if candidate != expected:
                    break
                accepted += 1
            feedback.observe(request_id, len(draft), accepted)
            sampled = target[emitted:emitted + accepted + 1]
            emitted += len(sampled)
            total["target_steps"] += 1
            total["proposed_tokens"] += len(draft)
            total["accepted_tokens"] += accepted
        if request_id in native.suffix_cache.active_requests:
            native.suffix_cache.add_active_response(request_id, sampled)
            native.suffix_cache.stop_request(request_id)
    return dict(total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    blocks = []
    for number in (1, 2):
        path = args.archive_root / f"matrix-b{number}-ar/attempt-01/retail.json"
        sessions = json.loads(path.read_bytes())["sessions"]
        traces = [trace for session in sessions for turn in session["turns"] for trace in turn["trace"]]
        if len(traces) != 50 or len({trace["id"] for trace in traces}) != 50:
            raise ValueError("Expected all 50 archived AR requests in the block")
        blocks.append({"block": number, "native": replay(traces, False), "feedback": replay(traces, True)})
    result = {"blocks": blocks, "new_model_requests": 0, "cache_warmup": 0}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
