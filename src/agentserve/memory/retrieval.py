"""Deterministic dense, BM25 and metadata views; SQLite remains authoritative."""

import math
import re
from collections import Counter


def terms(text):
    result = []
    for word in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        result.extend(
            [word]
            if word.isascii() or len(word) < 2
            else [word[i : i + 2] for i in range(len(word) - 1)]
        )
    return result


def retrieve(entries, vector, query, keywords, topic, depth):
    if not entries:
        return []
    documents = [Counter(terms(e["text"])) for e in entries]
    lengths = [sum(d.values()) for d in documents]
    average = sum(lengths) / len(entries) or 1
    query_terms = set(terms(query))
    df = {t: sum(t in d for d in documents) for t in query_terms}
    scores = [[], [], []]
    for entry, document, length in zip(entries, documents, lengths, strict=True):
        stored = entry["vector"]
        cosine = sum(x * y for x, y in zip(vector, stored, strict=True)) / max(
            math.sqrt(sum(x * x for x in vector) * sum(x * x for x in stored)), 1e-12
        )
        lexical = sum(
            math.log(1 + (len(entries) - df[t] + 0.5) / (df[t] + 0.5))
            * document[t]
            * 2.2
            / (document[t] + 1.2 * (0.25 + 0.75 * length / average))
            for t in query_terms
            if document[t]
        )
        metadata = len(set(keywords) & set(entry["keywords"])) + bool(
            topic and topic == entry["topic"]
        )
        for view, score in zip(scores, (cosine, lexical, metadata), strict=True):
            if score > 0:
                view.append((score, entry))
    ranked = [
        sorted(view, key=lambda s: (-s[0], -s[1]["turn_no"], s[1]["entry_id"]))[:depth]
        for view in scores
    ]
    union = {}
    for index in range(depth):
        for view in ranked:
            if index < len(view):
                entry = view[index][1]
                union.setdefault(entry["entry_id"], entry)
    return list(union.values())
