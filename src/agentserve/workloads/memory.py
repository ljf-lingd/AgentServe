"""Frozen multi-turn workload. Gold labels never appear in the API payload."""

import hashlib
import json


def reference(city, month, status, grouping=False):
    start, end = f"2026-{month:02d}-01", f"2026-{month + 1:02d}-01"
    query = "SELECT " + ("p.category," if grouping else "") + "SUM(i.quantity*i.unit_price_cents)"
    query += " FROM orders o JOIN order_items i ON i.order_id=o.order_id JOIN customers c ON c.customer_id=o.customer_id"
    if grouping:
        query += " JOIN products p ON p.product_id=i.product_id"
    query += f" WHERE c.city='{city}' AND o.status='{status}' AND o.order_date>='{start}' AND o.order_date<'{end}'"
    if grouping:
        query += " GROUP BY p.category ORDER BY p.category"
    return query


def conversation(index, length, prefix):
    cities = ["Beijing", "Shanghai", "Shenzhen"]
    first, second, third = [cities[(index + i) % 3] for i in range(3)]
    month = index % 4 + 1
    name = f"{prefix}{index:02d}营收"
    runtime, gold = [], {}

    def add(question, sql=None, tags=(), clarify=False):
        turn_id = f"{prefix}{index:02d}-{len(runtime) + 1:02d}"
        runtime.append({"id": turn_id, "question": question})
        gold[turn_id] = {"sql": sql, "ordered": True, "tags": list(tags), "clarify": clarify}

    add(
        f"新建分析“{name}”：查询2026年{month}月{first}客户的completed订单营收合计，单位分，只返回营收。",
        reference(first, month, "completed"),
    )
    add(
        f"继续“{name}”，城市只改为{second}，其他条件不变。",
        reference(second, month, "completed"),
        ["memory", "inheritance"],
    )
    add(
        f"纠正“{name}”：月份应为2026年{month + 1}月，并按商品category分组，按category升序返回类别、营收，其他条件不变。",
        reference(second, month + 1, "completed", True),
        ["memory", "correction"],
    )
    add(
        f"切换到另一组临时分析：2026年{month}月{third}的pending订单有多少笔？只返回数量。",
        f"SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id=o.customer_id WHERE c.city='{third}' AND o.status='pending' AND o.order_date>='2026-{month:02d}-01' AND o.order_date<'2026-{month + 1:02d}-01'",
    )
    add(
        "换个问题：各city有多少位客户？返回city、人数，按city升序。",
        "SELECT city,COUNT(*) FROM customers GROUP BY city ORDER BY city",
    )
    # Long histories contain meaningful independent queries, not pasted irrelevant padding.
    fillers = [
        ("所有状态共有多少笔订单？仅返回总数。", "SELECT COUNT(*) FROM orders"),
        ("共有多少种商品？仅返回总数。", "SELECT COUNT(*) FROM products"),
        (
            "completed订单有多少笔？仅返回总数。",
            "SELECT COUNT(*) FROM orders WHERE status='completed'",
        ),
        (
            "各category的商品种数是多少？返回类别、种数，按category升序。",
            "SELECT category,COUNT(*) FROM products GROUP BY category ORDER BY category",
        ),
    ]
    for j in range(length - 12):
        question, sql = fillers[(index + j) % len(fillers)]
        # Each filler carries an ordinary, scoped output contract, contributing real visible history.
        add(
            f"这是独立检查{j + 1}，不修改其他分析的任何条件。{question}"
            "请只查询当前只读数据库，遵守这个问题明确写出的统计范围。不要将其他话题中的城市、"
            "月份或状态加到本次统计。结果中只列出这里请求的列，不要把排序键或内部主键当成额外输出。"
            "回答请保留数据库原始单位；本次不要求解释原因、不要求预测，也不改变此前命名分析的定义。"
            "如果查询没有匹配记录，应如实说明；聚合结果为NULL时不要自行改成零。这里所有金额默认"
            "保持整数分，不换算成元。当前只问一个独立统计值或分组表，不需要联想前面的营收结果，"
            "也不应添加未请求的排名限制。请保持查询与这个问题的明确输出要求一致。",
            sql,
        )
    add(
        f"恢复“{name}”最后一次纠正后的分析：完整沿用那组的城市、月份、状态及category分组，返回类别、营收，按类别升序。",
        reference(second, month + 1, "completed", True),
        ["memory", "long_reference", "correction"],
    )
    add(
        f"继续“{name}”，订单状态明确改为pending，其他条件都保留。",
        reference(second, month + 1, "pending", True),
        ["memory", "correction"],
    )
    add("总共有多少位客户？这个问题独立于营收分析。", "SELECT COUNT(*) FROM customers")
    add(
        f"另开相似分析：2026年{month}月{first}的completed订单营收总计多少分？只返回合计。",
        reference(first, month, "completed"),
    )
    add(
        f"回到“{name}”刚才改为pending的版本，城市改回{first}，保持它的月份和category分组。",
        reference(first, month + 1, "pending", True),
        ["memory", "long_reference", "correction"],
    )
    add(
        "恢复我尚未在本会话描述过的“从未定义的海外利润分析”，保持那组原来的全部筛选条件。",
        tags=["clarification"],
        clarify=True,
    )
    add(
        f"不再使用那组未定义分析。独立查询2026年{month}月{third}的completed订单营收总计，单位分，只返回合计。",
        reference(third, month, "completed"),
    )
    assert len(runtime) == length
    return {"id": f"{prefix}{index:02d}", "turns": runtime}, gold


def build(split):
    count = 12 if split == "development" else 36
    conversations, gold = [], {}
    for index in range(count):
        length = 48 if split == "acceptance" and index >= 24 else 12
        runtime, labels = conversation(index, length, "d" if split == "development" else "a")
        conversations.append(runtime)
        gold.update(labels)
    return conversations, gold


def freeze(root):
    from agentserve.workloads.spider import write_frozen

    manifest = {
        "version": "memory-v2-1",
        "synthetic_template_workload": True,
        "generalization_limit": "shared templates; not a public benchmark",
        "splits": {},
    }
    for split in ["development", "acceptance"]:
        inputs, gold = build(split)
        for path, value in [
            (root / "data/splits/memory_v2" / f"{split}.json", inputs),
            (root / "data/gold/memory_v2" / f"{split}.json", gold),
        ]:
            write_frozen(path, value)
        manifest["splits"][split] = {
            "sessions": len(inputs),
            "turns": len(gold),
            "inputs_sha256": hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest(),
            "gold_sha256": hashlib.sha256(json.dumps(gold, sort_keys=True).encode()).hexdigest(),
        }
    write_frozen(root / "data/splits/memory_v2/manifest.json", manifest)
    return manifest
