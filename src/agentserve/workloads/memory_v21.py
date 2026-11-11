"""New scenario families and explicit provenance; labels never enter runtime state."""

import hashlib

from agentserve.workloads.spider import write_frozen

WORKLOAD_NAME = "memory_v21_r2"


def scenario(family, variant):
    cities = ["Beijing", "Shanghai", "Shenzhen"]
    city = cities[variant % 3]
    other = cities[(variant + 1) % 3]
    month = variant % 4 + 1
    date = f"o.order_date>='2026-{month:02d}-01' AND o.order_date<'2026-{month + 1:02d}-01'"
    join = " FROM orders o JOIN customers c ON c.customer_id=o.customer_id"
    items = join + " JOIN order_items i ON i.order_id=o.order_id"
    if family == 0:

        def sql(c):
            return (
                "SELECT SUM(i.quantity*i.unit_price_cents)"
                + items
                + f" WHERE c.city='{c}' AND o.status='completed' AND {date}"
            )

        return (
            f"2026年{month}月{city}客户completed订单营收合计，单位分，只返回营收",
            sql(city),
            f"城市改为{other}，其余不变",
            sql(other),
            "取消城市限制，其余不变",
            sql(city).replace(f"c.city='{city}' AND ", ""),
        )
    if family == 1:

        def sql(status):
            return (
                "SELECT COUNT(*)"
                + join
                + f" WHERE c.city='{city}' AND o.status='{status}' AND {date}"
            )

        return (
            f"2026年{month}月{city}客户pending订单笔数，只返回笔数",
            sql("pending"),
            "订单状态改成cancelled，其他不变",
            sql("cancelled"),
            "取消订单状态限制，其余不变",
            sql("pending").replace("o.status='pending' AND ", ""),
        )
    if family == 2:

        def sql(threshold):
            return (
                f"SELECT COUNT(*) FROM products WHERE category='Office' AND price_cents>{threshold}"
            )

        return (
            "Office类别中当前标价高于1000分的商品种数，只返回数量",
            sql(1000),
            "价格门槛改成高于2000分，其他不变",
            sql(2000),
            "取消价格门槛，其余不变",
            "SELECT COUNT(*) FROM products WHERE category='Office'",
        )
    if family == 3:

        def sql(c):
            return f"SELECT COUNT(*) FROM customers c WHERE c.city='{c}' AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id)"

        return (
            f"{city}中从未下过任何订单的客户人数，只返回人数",
            sql(city),
            f"城市改成{other}，继续只统计从未下单的人数",
            sql(other),
            "取消城市限制，仍只统计从未下单客户人数",
            sql(city).replace(f"c.city='{city}' AND ", ""),
        )
    if family == 4:

        def sql(category):
            return (
                "SELECT SUM(i.quantity)"
                + items
                + " JOIN products p ON p.product_id=i.product_id"
                + f" WHERE o.status='completed' AND p.category='{category}' AND {date}"
            )

        return (
            f"2026年{month}月completed订单中Home商品销量总件数，只返回件数",
            sql("Home"),
            "商品类别改为Electronics，其他不变",
            sql("Electronics"),
            "取消商品类别限制，其余不变",
            sql("Home").replace("p.category='Home' AND ", ""),
        )
    if family == 5:

        def sql(group):
            return f"SELECT {group},COUNT(*) FROM products GROUP BY {group} ORDER BY {group}"

        return (
            "按category统计商品种数，返回category和种数，按category升序",
            sql("category"),
            "分组维度改成当前标价price_cents，返回标价、种数，按标价升序",
            sql("price_cents"),
            "取消分组，只返回商品总种数",
            "SELECT COUNT(*) FROM products",
        )
    if family == 6:

        def sql(status):
            return (
                "SELECT COUNT(DISTINCT o.customer_id)"
                + join
                + f" WHERE c.city='{city}' AND o.status='{status}' AND {date}"
            )

        return (
            f"2026年{month}月{city}至少有一笔completed订单的去重客户人数，只返回人数",
            sql("completed"),
            "订单状态改成pending，其余不变",
            sql("pending"),
            "取消城市限制，其余不变",
            sql("pending").replace(f"c.city='{city}' AND ", ""),
        )
    if family == 7:

        def sql(limit):
            return f"SELECT sku,price_cents FROM products ORDER BY price_cents DESC,sku ASC LIMIT {limit}"

        return (
            "当前标价最高的前3个商品，返回sku、标价，价格降序并以sku升序打破并列",
            sql(3),
            "排名数量改为前5个，其余不变",
            sql(5),
            "取消排名数量限制，其余不变",
            "SELECT sku,price_cents FROM products ORDER BY price_cents DESC,sku ASC",
        )
    if family == 8:

        def sql(category):
            return f"SELECT AVG(price_cents) FROM products WHERE category='{category}'"

        return (
            "Home商品的当前平均标价，单位分，只返回平均值",
            sql("Home"),
            "类别改成Office，其余不变",
            sql("Office"),
            "取消类别限制，其余不变",
            "SELECT AVG(price_cents) FROM products",
        )
    if family == 9:

        def sql(city_value):
            return (
                "SELECT p.category,SUM(i.quantity*i.unit_price_cents)"
                + items
                + " JOIN products p ON p.product_id=i.product_id"
                + f" WHERE c.city='{city_value}' AND o.status='completed' AND {date} GROUP BY p.category ORDER BY p.category"
            )

        return (
            f"2026年{month}月{city} completed订单按商品category统计营收，返回category、营收分，按category升序",
            sql(city),
            f"城市替换成{other}，其他条件和分组不变",
            sql(other),
            "取消城市筛选，其他条件和分组不变",
            sql(city).replace(f"c.city='{city}' AND ", ""),
        )
    if family == 10:

        def sql(status):
            return (
                "SELECT c.city,COUNT(*)"
                + join
                + f" WHERE o.status='{status}' AND {date} GROUP BY c.city ORDER BY c.city"
            )

        return (
            f"2026年{month}月按客户city统计completed订单笔数，返回city、笔数，按city升序",
            sql("completed"),
            "状态改成cancelled，其他条件不变",
            sql("cancelled"),
            "取消状态筛选，保留月份与city分组",
            sql("completed").replace("o.status='completed' AND ", ""),
        )

    def sql(op):
        return f"SELECT COUNT(*) FROM products WHERE category {op} 'Electronics'"

    return (
        "不属于Electronics的商品种数，只返回数量",
        sql("!="),
        "改成只统计Electronics的商品种数",
        sql("="),
        "删除类别筛选，只返回商品总种数",
        "SELECT COUNT(*) FROM products",
    )


FILLERS = [
    (
        "各city有多少位客户？返回city、人数，按city升序。",
        "SELECT city,COUNT(*) FROM customers GROUP BY city ORDER BY city",
    ),
    (
        "全部订单中最早和最晚下单日期是多少？按最早、最晚返回。",
        "SELECT MIN(order_date),MAX(order_date) FROM orders",
    ),
    (
        "所有订单按status统计笔数，返回status、笔数，按status升序。",
        "SELECT status,COUNT(*) FROM orders GROUP BY status ORDER BY status",
    ),
    (
        "列出各category的最低和最高商品标价，返回category、最低、最高分，按category升序。",
        "SELECT category,MIN(price_cents),MAX(price_cents) FROM products GROUP BY category ORDER BY category",
    ),
    (
        "completed订单中有多少种不同商品？只返回去重商品种数。",
        "SELECT COUNT(DISTINCT i.product_id) FROM order_items i JOIN orders o ON o.order_id=i.order_id WHERE o.status='completed'",
    ),
    ("所有订单项的总件数是多少？只返回件数。", "SELECT SUM(quantity) FROM order_items"),
    (
        "各月份共有多少订单？返回YYYY-MM、笔数，按月份升序。",
        "SELECT strftime('%Y-%m',order_date),COUNT(*) FROM orders GROUP BY strftime('%Y-%m',order_date) ORDER BY 1",
    ),
    (
        "有哪些不同的客户城市？仅返回city，按city升序。",
        "SELECT DISTINCT city FROM customers ORDER BY city",
    ),
]


def build(split, development_length=12):
    conversations, labels = [], {}
    variants = 1 if split == "development" else 6
    for family in range(12):
        for v in range(variants):
            variant = v if split == "development" else v + 5
            sid = f"{split[:1]}-f{family:02d}-v{variant}"
            name = f"专题{family}-{variant}"
            length = development_length if split == "development" else [12, 24, 44, 24, 44, 44][v]
            q0, s0, q1, s1, q2, s2 = scenario(family, variant)
            turns = []

            def add(
                q,
                sql=None,
                tags=(),
                source=None,
                clarify=False,
                *,
                _sid=sid,
                _turns=turns,
                _family=family,
                _length=length,
            ):
                tid = f"{_sid}-{len(_turns) + 1:02d}"
                _turns.append({"id": tid, "question": q})
                labels[tid] = {
                    "sql": sql,
                    "ordered": True,
                    "tags": list(tags),
                    "clarify": clarify,
                    "source_turn": source,
                    "distance": len(_turns) - source if source else None,
                    "family": _family,
                    "length": _length,
                }

            add(f"建立命名分析“{name}”：{q0}。", s0)
            add(f"继续“{name}”，{q1}。", s1, ["memory", "inheritance"], 1)
            add(f"修改“{name}”：{q2}。", s2, ["memory", "correction"], 2)
            for j in range(length - 8):
                q, sql = FILLERS[(family + j) % len(FILLERS)]
                add("另做一个独立检查，不修改此前命名专题：" + q, sql)
            add(
                f"恢复“{name}”的原始第一个版本，完全沿用当时的统计要求，重新查询。",
                s0,
                ["memory", "long_reference", "old_version"],
                1,
            )
            original_turn = len(turns)
            add(
                f"回到“{name}”刚恢复的那个版本，{q1}。", s1, ["memory", "correction"], original_turn
            )
            q, sql = FILLERS[(family + 3) % len(FILLERS)]
            add("这是另一个独立问题：" + q, sql)
            add(
                f"恢复“{name}”刚才修改后的最新版本，保留所有条件、输出和排序，重新查询。",
                s1,
                ["memory", "long_reference"],
                original_turn + 1,
            )
            add(
                "恢复“本会话从未定义的海外成本专题”，完整继承其全部条件。",
                tags=["clarification"],
                clarify=True,
            )
            assert len(turns) == length
            conversations.append(
                {"id": sid, "database_id": "business_orders_v1", "family": family, "turns": turns}
            )
    return conversations, labels


def prepare(root):
    manifest = {
        "version": "memory-v21-scenarios-2",
        "source": "independently authored synthetic intent scripts",
        "families": 12,
        "limitations": "Single synthetic database; within-family variants are correlated.",
        "splits": {},
    }
    for split in ["development", "acceptance"]:
        questions, gold = build(split)
        for folder, value in [("splits", questions), ("gold", gold)]:
            path = root / f"data/{folder}/{WORKLOAD_NAME}/{split}.json"
            write_frozen(path, value)
            manifest["splits"][f"{split}-{folder}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_frozen(root / f"data/splits/{WORKLOAD_NAME}/manifest.json", manifest)
    return manifest
