"""Measure typed SQL follow-up drafts on recorded UCI tool calls."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

from retail_history import exact_trace, write_json
from retail_history_analysis import count_opportunity, eligible, history_text, load
from retail_replay import COUNTRIES


COLUMN = re.compile(r"(?P<column>invoice_date) >= '(?P<start>\d{4}-\d{2}-01)' AND invoice_date < '(?P<end>\d{4}-\d{2}-01)'")
COUNTRY = re.compile(r"country = '(?P<country>[A-Za-z ]+)'")
QUERY = re.compile(
    r"SELECT (?:ROUND\(SUM\((?:sales_gbp|cancellation_gbp)\), 6\)"
    r" AS (?:sales_gbp|cancellation_gbp)|COUNT\(DISTINCT (?:invoice|customer_id)\))"
    r" FROM retail_transactions WHERE " + COUNTRY.pattern + r" AND \(" + COLUMN.pattern + r"\)"
    r"(?: AND (?:is_cancellation = [01]|customer_id IS NOT NULL))?"
)
COUNTRY_QUESTION = re.compile(r"那(?P<country>[\u4e00-\u9fff]+)呢[？?]")
MONTH_QUESTION = re.compile(r"再看(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月呢[？?]")


def end_of_month(start):
    return date(start.year + (start.month == 12), start.month % 12 + 1, 1)


def slot_draft(question, prior_sql):
    if not prior_sql or not (shape := QUERY.fullmatch(prior_sql)):
        return None
    if shape["country"] not in COUNTRIES.values():
        return None
    try:
        start = date.fromisoformat(shape["start"])
        if end_of_month(start).isoformat() != shape["end"]:
            return None
        if country := COUNTRY_QUESTION.fullmatch(question):
            country_value = COUNTRIES.get(country["country"])
            if country_value is None:
                return None
            begin, end = shape.span("country")
            sql = prior_sql[:begin] + country_value + prior_sql[end:]
        elif month := MONTH_QUESTION.fullmatch(question):
            new_start = date(int(month["year"]), int(month["month"]), 1)
            new_end = end_of_month(new_start)
            begin, end = shape.start("start"), shape.end("end")
            sql = prior_sql[:begin] + f"{new_start}' AND invoice_date < '{new_end}" + prior_sql[end:]
        else:
            return None
    except (ValueError, OverflowError):
        return None
    call = json.dumps({"name": "execute_sql", "arguments": {"sql": sql}}, ensure_ascii=False)
    return f"<tool_call>\n{call}\n</tool_call><|im_end|>"


def analyze_block(sessions, tokenizer):
    corpus = []
    results = []
    for case, session in enumerate(sessions, start=1):
        prior = None
        for turn_no, turn in enumerate(session["turns"], start=1):
            draft = slot_draft(turn["question"], prior)
            for row in filter(exact_trace, turn["trace"]):
                pool = eligible(corpus, row)
                same = [x["tokens"] for x in pool if x["session"] == case][:4]
                budget = sum(map(len, same))
                other = [x for x in pool if x["session"] != case][:len(same)]
                remaining = budget
                cross_session = []
                for entry in other:
                    taken = entry["tokens"][:remaining]
                    cross_session.append(taken)
                    remaining -= len(taken)
                synthetic = tokenizer.encode(draft, add_special_tokens=False).ids if draft and row["phase"][0] else []
                if same or synthetic:
                    counts, example = count_opportunity(row, [synthetic] if synthetic else [], cross_session)
                    results.append({
                        "case": case, "turn": turn_no, "question": turn["question"],
                        "session_budget": budget, "other_sessions": [x["session"] for x in other],
                        "other_tokens": budget - remaining, "positions": counts["positions"],
                        "current_opportunities": counts["current"], "other_opportunities": counts["global_"],
                        "draft_opportunities": counts["session"], "draft_better_positions": counts["distinct_better"],
                        "draft_token_count": len(synthetic), "exact_tool_call": bool(synthetic and synthetic == row["tokens"]),
                        "example": example,
                    })
            if turn["status"] == "COMPLETED":
                prior = turn["sql"]
                for row in turn["trace"]:
                    if "step_no" in row and (text := history_text(row, turn)) is not None:
                        corpus.append({"session": case, "available": turn["ended_at"], "phase": row["phase"],
                                       "tokens": tokenizer.encode(text, add_special_tokens=False).ids})
    return results


if __name__ == "__main__":
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    write_json(args.output, {
        label: analyze_block(load(args.run, label)["sessions"], tokenizer)
        for label in ("dev-c1", "b1-c", "b2-c")
    })
