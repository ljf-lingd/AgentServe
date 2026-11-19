"""Rewrite narrowly scoped SQL on the UCI retail view."""

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
_CUSTOMER_COUNT_QUESTION = re.compile(
    r"\s*[\u3400-\u9fff]+\s*[0-9]{4}\s*年\s*(?:0?[1-9]|1[0-2])\s*月\s*的?\s*"
    r"(?:有\s*ID\s*的\s*(?:去重|不同)客户数|去重后的客户数)"
    r"\s*(?:是多少|有几位)\s*[？?]\s*(?:只返回(?:这个值|数字)[。.]?)?\s*",
    re.IGNORECASE,
)
_INVOICE_COUNT = re.compile(
    r"\s*SELECT\s+COUNT\s*\(\s*DISTINCT\s+(?P<invoice>invoice)\s*\)"
    r"\s+FROM\s+retail_transactions\s+WHERE\b",
    re.IGNORECASE,
)


def bind_retail_customer_count(question: str, sql: str) -> str:
    if not _CUSTOMER_COUNT_QUESTION.fullmatch(question):
        return sql
    projection = _INVOICE_COUNT.match(sql)
    if projection is None:
        return sql
    words = [token[0].lower() for token in _TOKEN.finditer(sql, projection.end())]
    if not words or ";" in words[:-1] or any(
        word.startswith(("--", "/*"))
        or word in {"select", "union", "intersect", "except", "group", "having", "window", "order", "limit"}
        for word in words
    ):
        return sql
    start, end = projection.span("invoice")
    return sql[:start] + "customer_id" + sql[end:]


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
