"""Deterministic evidence presentation; original model replies stay in the journal."""

import re

from agentserve.domain.models import QueryResult

BOUNDED_TEMPLATE = (
    "查询结果已截断：当前仅返回{rows}行，完整总行数未知。"
    "请缩小查询范围或增加过滤条件。已返回的数据见结构化结果。"
)
DIRECT_EVIDENCE_STYLE = "evidence_direct_v5"
EVIDENCE_STYLES = {"evidence_v21", "evidence_v221", DIRECT_EVIDENCE_STYLE}
CONCISE_STYLES = {"concise_v1", "bounded_v1", "evidence_v21", "evidence_v221"}

SQL_PROJECTION_TOKEN = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|"
    r"--[^\n]*|/\*[\s\S]*?\*/|[(),]|\bFROM\b",
    re.IGNORECASE,
)
IDENTIFIER = r'(?:[A-Za-z_]\w*|"[^"]+"|`[^`]+`|\[[^\]]+\])'
CENTS_COLUMN = rf'(?:{IDENTIFIER}\s*\.\s*)?["`\[]?(?:unit_price_cents|price_cents)["`\]]?'
QUANTITY = rf'(?:{IDENTIFIER}\s*\.\s*)?["`\[]?quantity["`\]]?'
CENTS_TERM = (
    rf"(?:{CENTS_COLUMN}\s*\*\s*{QUANTITY}|{QUANTITY}\s*\*\s*{CENTS_COLUMN}|{CENTS_COLUMN})"
)
CENTS_PROJECTION = re.compile(
    rf"(?:{CENTS_COLUMN}|(?:SUM|AVG|MIN|MAX)\s*\(\s*{CENTS_TERM}\s*\))"
    rf"(?:\s+(?:AS\s+)?{IDENTIFIER})?",
    re.IGNORECASE,
)


def order_amount_columns(sql: str, result: QueryResult) -> list[int]:
    """Return zero-based positions of recognized cents expressions in order-domain SELECTs."""
    select = re.match(r"\s*SELECT\s+(?:(?:DISTINCT|ALL)\s+)?", sql, re.IGNORECASE)
    if select is None or re.search(
        r"\b(?:FROM|JOIN)\s*\(|\b(?:UNION|INTERSECT|EXCEPT)\b", sql, re.IGNORECASE
    ):
        return []
    depth = 0
    start = select.end()
    expressions = []
    for token in SQL_PROJECTION_TOKEN.finditer(sql, start):
        word = token[0].upper()
        if word == "(":
            depth += 1
        elif word == ")":
            depth -= 1
        elif depth == 0 and word in {",", "FROM"}:
            expressions.append(sql[start : token.start()].strip())
            if word == "FROM":
                break
            start = token.end()
    if len(expressions) != len(result.columns):
        return []
    return [i for i, expr in enumerate(expressions) if CENTS_PROJECTION.fullmatch(expr)]


def order_amount_note(sql: str, result: QueryResult) -> str:
    positions = [f"第{i + 1}列" for i in order_amount_columns(sql, result)]
    return (
        f"金额列（{'、'.join(positions)}）的单位为分；本回答原样展示查询返回值。"
        if positions
        else ""
    )


def render_answer(
    model_answer: str, result: QueryResult, style: str, unit_note: str = ""
) -> tuple[str, str]:
    if style in EVIDENCE_STYLES | {"bounded_v1"} and result.truncated:
        return BOUNDED_TEMPLATE.format(rows=len(result.rows)), "bounded_summary"
    if style in EVIDENCE_STYLES:
        # Display exact tool values without inferred arithmetic, units or business scope.
        # When requested, the model answer is stored separately in the step journal.
        import json

        if not result.rows:
            return "本次查询无匹配记录。", "tool_values"
        return (
            "本次SQL返回的原始结果（字段与数值按数据库原样保留）："
            + json.dumps({"columns": result.columns, "rows": result.rows}, ensure_ascii=False)
            + "。"
            + unit_note
            + "统计范围以本次SQL为准。"
        ), "tool_values"
    return model_answer, "model"
