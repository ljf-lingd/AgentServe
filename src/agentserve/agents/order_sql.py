"""Compile complete order-domain state to a schema-bound read-only SELECT."""

TABLES = {
    "catalogue": "products p",
    "customers": "customers c",
    "orders": "orders o JOIN customers c ON c.customer_id = o.customer_id",
    "items": (
        "order_items i JOIN orders o ON o.order_id = i.order_id "
        "JOIN customers c ON c.customer_id = o.customer_id "
        "JOIN products p ON p.product_id = i.product_id"
    ),
}
COLUMNS = {
    "city": "c.city",
    "status": "o.status",
    "date": "o.order_date",
    "month": "strftime('%Y-%m', o.order_date)",
    "category": "p.category",
    "price": "p.price_cents",
    "price_cents": "p.price_cents",
    "sku": "p.sku",
}
FILTERS = {
    "catalogue": {"category", "price"},
    "customers": {"city"},
    "orders": {"city", "status", "date"},
    "items": {"city", "status", "date", "category", "price"},
}
GROUPS = {
    "catalogue": {"category", "price_cents"},
    "customers": {"city"},
    "orders": {"city", "status", "month"},
    "items": {"city", "status", "month", "category", "price_cents"},
}
METRICS = {
    "商品种数": ("catalogue", "COUNT(*)", "product_count"),
    "商品当前平均标价": ("catalogue", "AVG(p.price_cents)", "average_price_cents"),
    "曾售商品种数": ("items", "COUNT(DISTINCT i.product_id)", "product_count"),
    "订单笔数": ("orders", "COUNT(*)", "order_count"),
    "订单项营收合计": ("items", "SUM(i.quantity * i.unit_price_cents)", "revenue_cents"),
    "商品销量总件数": ("items", "SUM(i.quantity)", "units"),
    "去重客户人数": ("orders", "COUNT(DISTINCT o.customer_id)", "customer_count"),
    "客户人数": ("customers", "COUNT(*)", "customer_count"),
    "从未下过任何订单的客户人数": ("customers", "COUNT(*)", "customer_count"),
    "商品目录列表": ("catalogue", None, None),
    "客户城市列表": ("customers", None, None),
}
COMPARISONS = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def sql_literal(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def compile_order_sql(state):
    fields = state["fields"]
    metric = fields.get("metric", {}).get("value")
    if not isinstance(metric, str) or metric not in METRICS:
        return None
    entity, aggregate, alias = METRICS[metric]
    groups = fields.get("group_by", {}).get("value", [])
    if not isinstance(groups, list) or set(groups) - GROUPS[entity]:
        return None
    filters = fields.keys() - {"metric", "group_by", "outputs", "sort", "limit", "unit"}
    if filters - FILTERS[entity]:
        return None
    if "unit" in fields and fields["unit"]["value"] != "分":
        return None
    predicate = []
    for key in sorted(filters):
        op, value = fields[key]["operator"], fields[key]["value"]
        if key == "date" and op == "range" and isinstance(value, list) and len(value) == 2:
            predicate.extend(
                f"{COLUMNS[key]} {cmp} {sql_literal(v)}" for cmp, v in zip((">=", "<"), value)
            )
        elif key == "city" and op == "in" and isinstance(value, list) and value and all(
            isinstance(city, str) for city in value
        ):
            predicate.append(f"c.city IN ({', '.join(sql_literal(city) for city in value)})")
        elif op in COMPARISONS and type(value) in {str, int, float}:
            predicate.append(f"{COLUMNS[key]} {COMPARISONS[op]} {sql_literal(value)}")
        else:
            return None
    if (
        metric == "订单项营收合计"
        and "status" not in fields
        and "status" not in state.get("removed_fields", [])
    ):
        predicate.append("o.status = 'completed'")
    if metric == "从未下过任何订单的客户人数":
        predicate.append(
            "NOT EXISTS (SELECT 1 FROM orders prior WHERE prior.customer_id = c.customer_id)"
        )

    available = {key: COLUMNS[key] for key in groups}
    if aggregate:
        available["metric"] = aggregate
    elif groups:
        return None
    elif metric == "客户城市列表":
        available = {"city": COLUMNS["city"]}
    else:
        available = {key: COLUMNS[key] for key in ("sku", "price_cents", "category")}
    outputs = fields.get("outputs", {}).get("value", list(available))
    if not isinstance(outputs, list) or not outputs or set(outputs) - available.keys():
        return None
    order = fields.get("sort", {}).get("value", [])
    if not isinstance(order, list) or any(
        key[1:] not in available or key[:1] not in {"+", "-"} for key in order
    ):
        return None
    limit = fields.get("limit", {}).get("value")
    if limit is not None and (type(limit) is not int or limit < 1):
        return None

    projection = [f"{available[key]} AS {alias if key == 'metric' else key}" for key in outputs]
    distinct = "DISTINCT " if metric == "客户城市列表" else ""
    sql = f"SELECT {distinct}{', '.join(projection)} FROM {TABLES[entity]}"
    if predicate:
        sql += " WHERE " + " AND ".join(predicate)
    if groups:
        sql += " GROUP BY " + ", ".join(COLUMNS[key] for key in groups)
    if order:
        sql += " ORDER BY " + ", ".join(
            available[key[1:]] + (" DESC" if key[0] == "-" else " ASC") for key in order
        )
    if limit is not None:
        sql += f" LIMIT {limit}"
    return sql
