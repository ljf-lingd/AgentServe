"""Predeclared development controls for observed category-grain and date-grain failures."""


def controls():
    join = " FROM order_items i JOIN orders o ON o.order_id=i.order_id JOIN products p ON p.product_id=i.product_id"
    category = (
        "SELECT p.category,SUM(i.quantity*i.unit_price_cents)"
        + join
        + " WHERE o.status='completed' GROUP BY p.category ORDER BY p.category"
    )
    cases = [
        (
            "category.basic",
            "各商品category的completed订单营收是多少分？返回category、营收，按category升序。",
            category,
        ),
        (
            "category.items",
            "对completed订单按商品category分别汇总历史成交金额（件数乘订单项成交单价），返回category、金额分，按category升序。",
            category,
        ),
        (
            "category.quantity",
            "统计每个商品类别completed订单售出件数和营收分，按category升序返回类别、件数、营收。",
            "SELECT p.category,SUM(i.quantity),SUM(i.quantity*i.unit_price_cents)"
            + join
            + " WHERE o.status='completed' GROUP BY p.category ORDER BY p.category",
        ),
        (
            "category.all-status",
            "包括所有订单状态，按商品category汇总成交金额，按category升序输出category、金额分。",
            "SELECT p.category,SUM(i.quantity*i.unit_price_cents)"
            + join
            + " GROUP BY p.category ORDER BY p.category",
        ),
    ]
    for month, day in [(1, 31), (2, 1), (4, 30), (5, 1)]:
        date = f"2026-{month:02d}-{day:02d}"
        cases.append(
            (
                f"day.{date}",
                f"仅统计2026年{month}月{day}日当天的订单笔数，所有状态都包括，只返回笔数。",
                f"SELECT COUNT(*) FROM orders WHERE order_date='{date}'",
            )
        )
    for month in [2, 4]:
        cases.append(
            (
                f"month.{month}",
                f"统计2026年{month}月一整月的所有状态订单笔数，只返回笔数。",
                f"SELECT COUNT(*) FROM orders WHERE order_date>='2026-{month:02d}-01' AND order_date<'2026-{month + 1:02d}-01'",
            )
        )
    jobs = [
        {"id": "v21-control." + key, "database_id": "business_orders_v1", "question": question}
        for key, question, sql in cases
    ]
    gold = {
        "v21-control." + key: {
            "id": "v21-control." + key,
            "database_id": "business_orders_v1",
            "sql": sql,
            "ordered": True,
        }
        for key, question, sql in cases
    }
    extras = [
        (
            "string.sku-first3",
            "按product_id升序列出前3个商品的sku，删除其中每一个连字符，只输出处理后的sku。",
            "SELECT REPLACE(sku,'-','') FROM products ORDER BY product_id LIMIT 3",
        ),
        (
            "string.sku-range",
            "列出product_id从5到9（含两端）的商品sku，移除所有连字符，按product_id升序，只返回新sku。",
            "SELECT REPLACE(sku,'-','') FROM products WHERE product_id>=5 AND product_id<=9 ORDER BY product_id",
        ),
        (
            "string.sku-zero",
            "按product_id升序取前4个商品，把sku里的每一个数字字符0移除，保留连字符；只输出处理后的sku。",
            "SELECT REPLACE(sku,'0','') FROM products ORDER BY product_id LIMIT 4",
        ),
    ]
    for key, question, sql in extras:
        case_id = "v21-control." + key
        jobs.append({"id": case_id, "database_id": "business_orders_v1", "question": question})
        gold[case_id] = {
            "id": case_id,
            "database_id": "business_orders_v1",
            "sql": sql,
            "ordered": True,
        }
    return jobs, gold
