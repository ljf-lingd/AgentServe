"""Analyze exact UCI model traces and compare fixed-prefix SAM draft opportunities."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from retail_history import exact_trace, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "serving/vllm"))
from session_sam import SamTape


def load(path, label):
    return json.loads((path / (label + ".json")).read_text(encoding="utf-8"))


def history_text(row, turn):
    if row["phase"][0]:
        if row["sql"] is not None and row["sql"] == turn["sql"]:
            return json.dumps(row["sql"], ensure_ascii=False)[1:-1]
    return row["content"] if not row["phase"][0] else None


def eligible(corpus, row):
    return [item for item in reversed(corpus)
            if item["available"] < row["start"] and item["phase"] == row["phase"]]


def longest(tape, histories, cap):
    suffix = tape.tokens[-32:]
    length, draft = tape.continuation(suffix, cap)
    for history in histories:
        matched, candidate = history.continuation(suffix, cap)
        if matched > length:
            length, draft = matched, candidate
    return draft[:max(1, length)]


def accepted(draft, target):
    length = 0
    for predicted, actual in zip(draft, target):
        if predicted != actual:
            break
        length += 1
    return length


def count_opportunity(row, session, global_entries):
    current = SamTape(row["prompt"])
    histories = [SamTape(tokens) for tokens in session]
    global_tapes = [SamTape(tokens) for tokens in global_entries]
    counts = dict(positions=0, current=0, session=0, global_=0, distinct_better=0, overlapping=0)
    example = None
    for index, token in enumerate(row["tokens"][:-1], start=1):
        current.extend([token])
        room = min(4, row["max_tokens"] - index, 8192 - len(current.tokens))
        if room <= 0:
            break
        target = row["tokens"][index:index + room]
        drafts = [longest(current, tapes, room) for tapes in ([], histories, global_tapes)]
        c, s, g = (accepted(draft, target) for draft in drafts)
        counts["positions"] += 1
        counts["current"] += c
        counts["session"] += s
        counts["global_"] += g
        counts["overlapping"] += bool(s and drafts[0] == drafts[1])
        if s > c:
            counts["distinct_better"] += 1
            if example is None or s - c > example["session_match"] - example["current_match"]:
                example = {"prefix_generated_tokens": index, "current_draft": drafts[0],
                           "session_draft": drafts[1], "target_next": target,
                           "current_match": c, "session_match": s}
    return counts, example


def offline(sessions, tokenizer, *, verify_history=False):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    corpus = []
    results = []
    for index, session in enumerate(sessions, start=1):
        detail = []
        for turn_no, turn in enumerate(session["turns"], start=1):
            for row in filter(exact_trace, turn["trace"]):
                available = eligible(corpus, row)
                same = [item for item in available if item["session"] == index][:4]
                if verify_history and row["history"] != [item["text"] for item in same]:
                    raise RuntimeError(f"Session history mismatch at session {index}, turn {turn_no}")
                if verify_history:
                    continue
                session_tokens = [item["tokens"] for item in same]
                budget = sum(map(len, session_tokens))
                remaining = budget
                chosen = []
                for item in available[:len(same)]:
                    if remaining <= 0:
                        break
                    tokens = item["tokens"][:remaining]
                    chosen.append(tokens)
                    remaining -= len(tokens)
                counts, example = count_opportunity(row, session_tokens, chosen)
                prompt = row["prompt"]
                present = sum(
                    any(prompt[i:i + len(tokens)] == tokens for i in range(len(prompt) - len(tokens) + 1))
                    for tokens in session_tokens if tokens
                )
                detail.append({
                    "turn": turn_no, "phase": row["phase"], "question": turn["question"],
                    "historical_tokens": budget, "global_tokens": budget - remaining,
                    "history_entries": len(session_tokens), "entries_already_in_prompt": present,
                    "same_history": session_tokens == chosen, **counts, "example": example,
                })
            if turn["status"] != "COMPLETED":
                continue
            for row in turn["trace"]:
                if "step_no" not in row or (text := history_text(row, turn)) is None:
                    continue
                corpus.append({"session": index, "available": turn["ended_at"],
                               "phase": row["phase"], "text": text, "tokens": encode(text)})
        results.append(detail)
    return results


def pair(current, session):
    pairs = []
    for index, (a, b) in enumerate(zip(current, session, strict=True), start=1):
        turns = []
        for c, s in zip(a["turns"], b["turns"], strict=True):
            def trace_key(trace, fields):
                return [[row.get(field) for field in fields] for row in trace]

            turns.append({
                "question": c["question"], "current_status": c["status"], "session_status": s["status"],
                "current_correct": c["correct"], "session_correct": s["correct"],
                "current_jct": c["jct_seconds"], "session_jct": s["jct_seconds"],
                "session_requests_with_history": sum(bool(row["history"]) for row in s["trace"]),
                "current_requests": len(c["trace"]), "session_requests": len(s["trace"]),
                "current_exact_traces": sum(map(exact_trace, c["trace"])),
                "session_exact_traces": sum(map(exact_trace, s["trace"])),
                "current_sql_time": c["sql_seconds"], "session_sql_time": s["sql_seconds"],
                "current_model_http_seconds": sum(row["end"] - row["start"] for row in c["trace"]),
                "session_model_http_seconds": sum(row["end"] - row["start"] for row in s["trace"]),
                "current_drafts": sum((row.get("spec") or {}).get("num_draft_tokens", 0) for row in c["trace"]),
                "session_drafts": sum((row.get("spec") or {}).get("num_draft_tokens", 0) for row in s["trace"]),
                "current_accepted": sum((row.get("spec") or {}).get("num_accepted_draft_tokens", 0) for row in c["trace"]),
                "session_accepted": sum((row.get("spec") or {}).get("num_accepted_draft_tokens", 0) for row in s["trace"]),
                "input_equal": trace_key(c["trace"], ("request_hash", "prompt")) == trace_key(s["trace"], ("request_hash", "prompt")),
                "generation_equal": trace_key(c["trace"], ("tokens", "finish")) == trace_key(s["trace"], ("tokens", "finish")),
                "draft_metrics_equal": trace_key(c["trace"], ("spec",)) == trace_key(s["trace"], ("spec",)),
                "sql_equal": c["sql"] == s["sql"], "result_equal": c["result"] == s["result"],
                "answer_equal": c["answer"] == s["answer"],
                "current_error": (c["error"] or {}).get("code"), "session_error": (s["error"] or {}).get("code"),
            })
        current_time, session_time = a["active_seconds"], b["active_seconds"]
        pairs.append({"case": index, "current_seconds": current_time, "session_seconds": session_time,
                      "relative_improvement": (current_time - session_time) / current_time,
                      "turns": turns})
    return pairs


def analyze(path, tokenizer_path, output):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    results = {}
    offline_results = []
    for block in (1, 2):
        a = load(path, f"b{block}-c")["sessions"]
        b = load(path, f"b{block}-s")["sessions"]
        offline(b, tokenizer, verify_history=True)
        offline_results.append(offline(a, tokenizer))
        results[f"block{block}"] = pair(a, b)
    manifest = load(path, "manifest")
    threshold = load(path, "threshold")
    write_json(output, {
        "manifest": manifest, "threshold": threshold, "blocks": results, "offline": offline_results,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "median_reductions": [statistics.median(row["relative_improvement"] for row in block)
                              for block in results.values()],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.run, args.tokenizer, args.output)
