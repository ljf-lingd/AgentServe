"""Use the existing date indexes for common filters on the UCI retail view."""

import re
from datetime import date

RETAIL_DATABASE = "online_retail_ii"
_TOKEN = re.compile(
    r"--[^\r\n]*|/\*.*?(?:\*/|\Z)|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|"
    r"`(?:``|[^`])*`|\[[^\]]*\]|[\w$]+|\S",
    re.DOTALL,
)
_TABLE = re.compile(
    r"retail_transactions(?:\s+(?:as\s+)?([a-z_][a-z_0-9]*))?", re.IGNORECASE
)
_DATE_FILTER = re.compile(
    r"(?i:strftime)\s*\(\s*'(?P<format>%Y-%m(?:-%d)?)'\s*,\s*"
    r"(?P<column>(?:[a-zA-Z_][a-zA-Z_0-9]*\.)?(?i:invoice_date))\s*\)\s*"
    r"(?P<operator>>=|<|=)\s*'(?P<value>[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)'"
)


def use_retail_date_index(sql: str) -> str:
    tokens = list(_TOKEN.finditer(sql))
    words = [token[0].lower() for token in tokens]
    if not words or words[0] != "select" or words.count("select") != 1:
        return sql
    if ";" in words[:-1] or any(
        word.startswith(("--", "/*"))
        or word in {"union", "intersect", "except", "'", '"', "`", "["}
        for word in words
    ):
        return sql
    if words.count("from") != 1 or words.count("where") != 1:
        return sql
    source, where = words.index("from"), words.index("where")
    table = _TABLE.fullmatch(sql[tokens[source].end():tokens[where].start()].strip())
    if not table:
        return sql

    qualifiers = {"", (table[1] or "retail_transactions").lower()}
    clauses = {"group", "having", "window", "order", "limit", ";"}
    boundaries = [tokens[where].end()]
    depth = 0
    for token in tokens[where + 1:]:
        word = token[0].lower()
        if word == "(":
            depth += 1
        elif word == ")":
            depth -= 1
        elif depth == 0 and word in clauses:
            boundaries.append(token.start())
            break
        elif depth == 0 and word in {"or", "between", "case"}:
            return sql
        elif depth == 0 and word == "and":
            boundaries.extend((token.start(), token.end()))
    else:
        boundaries.append(len(sql))

    edits = []
    day_edits = []
    day_operators = set()
    for start, end in zip(boundaries[::2], boundaries[1::2], strict=True):
        predicate = sql[start:end].strip()
        match = _DATE_FILTER.fullmatch(predicate)
        if match is None:
            continue
        format_string, column, operator, value = match.groups()
        qualifier = column.lower().rpartition(".")[0]
        if qualifier not in qualifiers:
            continue
        start += len(sql[start:end]) - len(sql[start:end].lstrip())
        end = start + len(predicate)
        if format_string == "%Y-%m" and operator == "=" and len(value) == 7:
            try:
                first = date.fromisoformat(value + "-01")
                following = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
            except ValueError:
                continue
            edits.append((start, end, f"({column} >= '{first}' AND {column} < '{following}')"))
        elif format_string == "%Y-%m-%d" and operator in {">=", "<"} and len(value) == 10:
            try:
                date.fromisoformat(value)
            except ValueError:
                continue
            day_operators.add(operator)
            day_edits.append((start, end, f"{column} {operator} '{value}'"))

    if day_operators == {">=", "<"}:
        edits.extend(day_edits)
    for start, end, replacement in sorted(edits, reverse=True):
        sql = sql[:start] + replacement + sql[end:]
    return sql
