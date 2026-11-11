"""Necessary intent coverage for observed order-domain failures; not SQL equivalence proof."""

import re
from datetime import date, timedelta


def covers_order_period(sql, period):
    start, end = period
    last_day = (date.fromisoformat(end) - timedelta(days=1)).isoformat()
    literals = set(re.findall(r"['\"]([0-9]{4}(?:-[0-9]{2}){0,2})(?:-?%)?['\"]", sql))
    year, first_month, last_month = start[:4], start[:7], last_day[:7]
    full_months = start.endswith("-01") and end.endswith("-01")
    allowed = {start, end, last_day, year}
    if full_months:
        allowed.update((first_month, last_month, end[:7]))
    if literals - allowed:
        return False

    order_date = r"\b(?:\w+\.)?order_date\b"
    day_field = rf"(?:date\(\s*{order_date}\s*\)|{order_date})"
    month_field = rf"strftime\(\s*['\"]%Y-%m['\"]\s*,\s*{order_date}\s*\)"

    def match(field, operator, value):
        return re.search(rf"{field}\s*{operator}\s*['\"]{value}['\"]", sql, re.IGNORECASE)

    def has_range(field, lower, upper, inclusive_upper):
        return bool((match(field, ">=", lower)
                     and (match(field, "<", upper) or match(field, "<=", inclusive_upper)))
                    or match(field, "BETWEEN", rf"{lower}['\"]\s+AND\s+['\"]{inclusive_upper}"))

    if has_range(day_field, start, end, last_day):
        return True
    if full_months and has_range(month_field, first_month, end[:7], last_month):
        return True
    if start == last_day:
        return bool(match(day_field, "=", start))
    if full_months and first_month == last_month:
        return bool(match(month_field, "=", first_month) or match(order_date, "LIKE", first_month + "-%"))
    if start[5:] == "01-01" and last_day == year + "-12-31":
        year_field = rf"strftime\(\s*['\"]%Y['\"]\s*,\s*{order_date}\s*\)"
        return bool(match(year_field, "=", year) or match(order_date, "LIKE", year + "-%"))
    return False


def intent_violations(sql, question, state=None, *, city=None, settled=False):
    failures = []
    fields = state["fields"] if state else {}
    period = state.get("workbench_period") if state else None
    if period and not covers_order_period(sql, period):
        start, end = period
        failures.append(f"Filter order_date using the verified period: >= '{start}' and < '{end}'.")
    required = {}
    if not state and not re.search(r"取消[^，。]{0,10}城市|所有城市|全部城市|不限制城市", question):
        city_matches = re.findall(
            r"城市[^，。；]{0,8}?(Beijing|Shanghai|Shenzhen)|(Beijing|Shanghai|Shenzhen)(?:的)?客户",
            question,
        )
        cities = {city for pair in city_matches for city in pair if city}
        if len(cities) == 1:
            required["city"] = next(iter(cities))
    for field in ["city", "status"]:
        item = fields.get(field)
        if item and isinstance(item["value"], str):
            required[field] = item["value"]
    explicit_status = []
    if not state and not re.search(
        r"(?:所有|全部)(?:订单)?状态|取消[^，。]{0,10}状态|不限制状态", question
    ):
        for words, value in [
            (r"已完成|\bcompleted\b", "completed"),
            (r"\bpending\b", "pending"),
            (r"已取消|\bcancelled\b", "cancelled"),
        ]:
            if re.search(words, question, re.IGNORECASE):
                explicit_status.append(value)
        if len(explicit_status) == 1:
            required["status"] = explicit_status[0]
    for field, value in required.items():
        if not re.search(r"(['\"])" + re.escape(value) + r"\1", sql, re.IGNORECASE):
            owner = (
                "orders.status (join orders on order_id for line items)"
                if field == "status"
                else "customers.city"
            )
            failures.append(
                f"The explicit user condition {field}={value} is absent. Include it using {owner}."
            )
    if city and "city" not in required and not re.search(
        r"(['\"])" + re.escape(city) + r"\1", sql,
    ):
        failures.append(f"Use the exact stored value '{city}' for customers.city, without its display label.")
    if settled and not re.search(
        r"\bstatus\s*(?:=\s*'completed'|IN\s*\(\s*'completed'\s*\))", sql, re.IGNORECASE,
    ):
        failures.append("Settled transactions require orders.status = 'completed', even for a count.")
    if settled:
        existence = re.search(r"有(?:没有)?(?:成交|交易)(?:吗|么)?$", question.rstrip("？?。！! "))
        if existence and not re.search(r"\b(?:COUNT|EXISTS)\s*\(", sql, re.IGNORECASE):
            failures.append("To check if completed orders exist, use COUNT(*) or EXISTS on orders without joining order_items.")
        if re.search(r"\bJOIN\s+order_items\b", sql, re.IGNORECASE) and (
            re.search(r"\bCOUNT\s*\(\s*(?:\w+\.)?order_id\s*\)", sql, re.IGNORECASE)
            or (existence and re.search(r"\bCOUNT\s*\(\s*[1*]\s*\)", sql, re.IGNORECASE))
        ):
            failures.append("Count orders without joining order_items, or use COUNT(DISTINCT orders.order_id).")
    if (
        re.search(r"\bcategory\b", sql, re.IGNORECASE)
        and re.search(r"\bSUM\s*\(", sql, re.IGNORECASE)
        and re.search(
            r"\(\s*SELECT\s+(?:\w+\.)?product_id\s+FROM\s+order_items\b[^)]*\bLIMIT\s+1\s*\)",
            sql,
            re.IGNORECASE,
        )
    ):
        failures.append(
            "Category revenue must join each line item to its own product; selecting an order first product changes the aggregation grain."
        )
    metric = str((fields.get("metric") or {}).get("value", ""))
    if re.search(r"销量|总件数|售出件数", metric) and not (
        re.search(r"\bSUM\s*\(", sql, re.IGNORECASE)
        and re.search(r"\bquantity\b", sql, re.IGNORECASE)
    ):
        failures.append(
            "Total units sold requires summing line-item quantity, not counting rows or orders."
        )
    return failures
