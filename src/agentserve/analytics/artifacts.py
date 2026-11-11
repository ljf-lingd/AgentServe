"""Versioned, read-only rendering of completed business briefs."""

import csv
import io
import re
from html import escape

from agentserve.domain.models import AgentError

ARTIFACT_TYPES = {
    "chart.svg": "image/svg+xml",
    "report.md": "text/markdown",
    "monthly.csv": "text/csv",
}
FIELDS = ("month", "revenue_cents", "order_count")
SERIES = (
    ("revenue_cents", "成交营收（分）", "#176b64"),
    ("order_count", "完成订单量（笔）", "#305d9a"),
)


def render_artifact(analysis, name):
    if name not in analysis["artifacts"] or name not in ARTIFACT_TYPES:
        raise AgentError("ARTIFACT_NOT_FOUND", "No artifact with this name")
    if analysis["artifact_version"] != 1:
        raise AgentError("ARTIFACT_VERSION_UNSUPPORTED", "Unknown report artifact version")
    if name == "chart.svg":
        content = svg_v1(analysis["months"])
    elif name == "report.md":
        content = markdown_v1(analysis)
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(analysis["months"])
        content = output.getvalue()
    return content, ARTIFACT_TYPES[name]


def markdown_v1(analysis):
    window, totals = analysis["window"], analysis["totals"]
    city = escape(window.get("city", "全部城市"))
    city = re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", city)
    revenue = totals["revenue_cents"]
    lines = [
        "# 月度经营简报", "",
        f"统计窗口 [{window['start']}, {window['end']})，城市 {city}。", "",
        f"- 成交营收 {'NULL' if revenue is None else f'{revenue} 分'}",
        f"- 完成订单量 {totals['order_count']} 笔", "",
        "| 月份 | 成交营收（分） | 完成订单量（笔） |",
        "| --- | ---: | ---: |",
    ]
    for row in analysis["months"]:
        amount = row["revenue_cents"]
        lines.append(f"| {row['month']} | {'NULL' if amount is None else amount} | {row['order_count']} |")
    lines.extend(["", "## 批准来源", ""])
    for metric in analysis["metrics"]:
        source = metric["citation"]
        lines.append(
            f"- {source['title']}，版本 {source['version']}，"
            f"{source['path']}#{source['anchor']}，指纹 `{metric['rule_sha256']}`"
        )
    return "\n".join(lines) + "\n"


def svg_v1(rows):
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450"'
        ' role="img" aria-labelledby="title">',
        '<title id="title">月度成交营收和完成订单量</title>',
        '<rect width="800" height="450" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#233242">',
    ]
    width = 660 / len(rows)
    for panel, (field, label, color) in enumerate(SERIES):
        top = 20 + panel * 220
        values = [row[field] for row in rows if row[field] is not None]
        low, high = min([0, *values]), max([0, *values])
        span = high - low or 1
        baseline = top + 154 - 112 * (-low / span)
        parts.extend([
            f'<text x="20" y="{top + 16}" font-size="16">{label}</text>',
            f'<text x="20" y="{top + 44}" font-size="11">{high}</text>',
            f'<text x="20" y="{top + 154}" font-size="11">{low}</text>',
            f'<path d="M 112 {baseline:.2f} H 772" stroke="#9aa8b5"/>',
        ])
        for index, row in enumerate(rows):
            center = 112 + width * (index + 0.5)
            month, value = row["month"], row[field]
            if value is not None:
                end = baseline - 112 * (value / span)
                height = max(abs(end - baseline), 1)
                bar = min(width * 0.62, 48)
                parts.append(
                    f'<rect x="{center - bar / 2:.2f}" y="{min(end, baseline):.2f}" '
                    f'width="{bar:.2f}" height="{height:.2f}" fill="{color}">'
                    f'<title>{month} · {value}</title></rect>'
                )
            parts.append(
                f'<text x="{center:.2f}" y="{top + 178}" font-size="10" text-anchor="middle">{month}</text>'
            )
        if not values:
            parts.append(f'<text x="442" y="{top + 100}" text-anchor="middle">暂无金额数据</text>')
    parts.append("</g></svg>")
    return "\n".join(parts) + "\n"
