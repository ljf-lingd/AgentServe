"""Frozen V2.2 cases and independent SQL references for the order domain."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from contextlib import closing

WORKLOAD_NAME = "memory_v221"
CITIES = ("Beijing", "Shanghai", "Shenzhen")
CATEGORIES = ("Electronics", "Office", "Home")
STATUSES = ("completed", "pending", "cancelled")


def session_cases(family, variant):
    """Author three semantic versions of a scenario, independent of runtime state or SQL."""
    city, other = CITIES[variant % 3], CITIES[(variant + 1) % 3]
    cat, other_cat = CATEGORIES[variant % 3], CATEGORIES[(variant + 2) % 3]
    month = 5 + variant if variant < 8 else 9 + variant - 8
    period = (f"2026-{month:02d}-01", f"2026-{month + 1:02d}-01" if month < 12 else "2027-01-01")
    date_sql = f"o.order_date >= '{period[0]}' AND o.order_date < '{period[1]}'"
    status, other_status = STATUSES[variant % 3], STATUSES[(variant + 1) % 3]
    orders = "orders o JOIN customers c USING (customer_id)"
    items = orders + " JOIN order_items i USING (order_id) JOIN products p USING (product_id)"
    if family == 0:
        q = f"2026年{month}月{city}客户{status}订单营收合计，单位分，只输出营收"
        states = {
            "city": city,
            "date": list(period),
            "status": status,
            "metric": "订单项营收合计",
            "unit": "分",
            "outputs": ["metric"],
        }
        stem = f"SELECT SUM(i.quantity*i.unit_price_cents) FROM {items} WHERE {date_sql} AND o.status='{status}'"
        return (
            q,
            stem + f" AND c.city='{city}'",
            states,
            f"城市替换成{other}，其他字段不变",
            stem + f" AND c.city='{other}'",
            {"city": other},
            "移除城市筛选，其他条件不变",
            stem,
            {"city": None},
        )
    if family in {1, 6, 10}:
        metric, expression, noun = {
            1: ("订单笔数", "COUNT(*)", "订单笔数"),
            6: (
                "去重客户人数",
                "COUNT(DISTINCT o.customer_id)",
                "至少有一笔{status}订单的去重客户人数",
            ),
            10: ("订单笔数", "COUNT(*)", "订单笔数"),
        }[family]
        q = (
            f"2026年{month}月{city}客户{status}{noun}"
            if family == 1
            else (
                f"2026年{month}月{city}{noun.format(status=status)}"
                if family == 6
                else f"2026年{month}月按客户城市汇总{status}订单笔数，输出城市、笔数，按城市升序"
            )
        )
        states = {"date": list(period), "status": status, "metric": metric}
        if family == 10:
            stem = f"SELECT c.city,{expression} FROM {orders} WHERE {date_sql}"
            order = " GROUP BY c.city ORDER BY c.city"
            states.update(group_by=["city"], outputs=["city", "metric"], sort=["+city"])
        else:
            stem = f"SELECT {expression} FROM {orders} WHERE {date_sql} AND c.city='{city}'"
            order = ""
            states["city"] = city
        return (
            q,
            stem + f" AND o.status='{status}'" + order,
            states,
            f"状态替换成{other_status}，其余字段不变",
            stem + f" AND o.status='{other_status}'" + order,
            {"status": other_status},
            "移除订单状态筛选，其他条件不变",
            stem + order,
            {"status": None},
        )
    if family in {2, 8, 11}:
        aggregate = "AVG(price_cents)" if family == 8 else "COUNT(*)"
        metric = "商品当前平均标价" if family == 8 else "商品种数"
        states = {"category": cat, "metric": metric, "outputs": ["metric"]}
        stem = f"SELECT {aggregate} FROM products"
        if family == 2:
            first, second = 750 + variant * 100, 1150 + variant * 100
            scope = f" WHERE category='{cat}'"
            states["price"] = first
            return (
                f"{cat}类别中当前标价高于{first}分的目录商品种数，输出种数",
                stem + scope + f" AND price_cents>{first}",
                states,
                f"价格门槛替换成高于{second}分，其他条件不变",
                stem + scope + f" AND price_cents>{second}",
                {"price": second},
                "移除价格筛选，其他条件不变",
                stem + scope,
                {"price": None},
            )
        threshold = 1200 + variant * 175
        price_text = f"当前标价高于{threshold}分的" if variant >= 6 else ""
        base_condition = f"price_cents>{threshold}" if price_text else "1=1"
        if price_text:
            states["price"] = threshold
        stem += f" WHERE {base_condition}"
        if family == 11:
            return (
                f"不属于{cat}的{price_text}目录商品种数，只输出数量",
                stem + f" AND category<>'{cat}'",
                states,
                f"改为只统计{other_cat}目录商品种数",
                stem + f" AND category='{other_cat}'",
                {"category": other_cat},
                "移除商品类别筛选，保留输出",
                stem,
                {"category": None},
            )
        states["unit"] = "分"
        return (
            f"{price_text}{cat}目录商品的当前平均标价，单位分，输出平均值",
            stem + f" AND category='{cat}'",
            states,
            f"类别替换成{other_cat}，其他字段不变",
            stem + f" AND category='{other_cat}'",
            {"category": other_cat},
            "移除类别筛选，保留输出",
            stem,
            {"category": None},
        )
    if family == 3:
        stem = "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id)"
        return (
            f"{city}从未下过任何订单的客户人数，输出人数",
            stem + f" AND c.city='{city}'",
            {"city": city, "metric": "从未下过任何订单的客户人数", "outputs": ["metric"]},
            f"城市替换成{other}，其他字段不变",
            stem + f" AND c.city='{other}'",
            {"city": other},
            "移除城市筛选，保留输出",
            stem,
            {"city": None},
        )
    if family in {4, 9}:
        group = family == 9
        expression = "SUM(i.quantity*i.unit_price_cents)" if group else "SUM(i.quantity)"
        stem = f"SELECT {'p.category,' if group else ''}{expression} FROM {items} WHERE {date_sql} AND o.status='{status}'"
        tail = " GROUP BY p.category ORDER BY p.category" if group else ""
        states = {"date": list(period), "status": status, "category": cat}
        if group:
            states.update(
                metric="订单项营收合计",
                group_by=["category"],
                outputs=["category", "metric"],
                sort=["+category"],
                unit="分",
            )
            q = f"2026年{month}月{status}订单中{cat}商品按category汇总营收，输出category、营收分，按category升序"
        else:
            states.update(metric="商品销量总件数", outputs=["metric"])
            q = f"2026年{month}月{status}订单中{cat}商品销量总件数，输出件数"
        return (
            q,
            stem + f" AND p.category='{cat}'" + tail,
            states,
            f"商品类别替换成{other_cat}，其他条件和分组不变",
            stem + f" AND p.category='{other_cat}'" + tail,
            {"category": other_cat},
            "移除商品类别筛选，其他字段不变",
            stem + tail,
            {"category": None},
        )
    if family == 5:
        order = "降序" if variant % 2 else "升序"
        direction = "DESC" if variant % 2 else "ASC"
        sign = "-" if variant % 2 else "+"
        stem = f"FROM products WHERE category <> '{cat}'"
        return (
            f"按category统计不属于{cat}的目录商品种数，输出category、种数，按category{order}",
            f"SELECT category,COUNT(*) {stem} GROUP BY category ORDER BY category {direction}",
            {
                "category": cat,
                "group_by": ["category"],
                "metric": "商品种数",
                "outputs": ["category", "metric"],
                "sort": [sign + "category"],
            },
            f"分组维度改成当前标价price_cents，输出标价、种数，按标价{order}",
            f"SELECT price_cents,COUNT(*) {stem} GROUP BY price_cents ORDER BY price_cents {direction}",
            {
                "group_by": ["price_cents"],
                "outputs": ["price_cents", "metric"],
                "sort": [sign + "price_cents"],
            },
            "取消分组和排序，只输出商品总种数",
            f"SELECT COUNT(*) {stem}",
            {"group_by": None, "sort": None, "outputs": ["metric"]},
        )
    first, second = 2 + variant % 3, 6 + variant % 3
    stem = f"SELECT sku,price_cents FROM products WHERE category='{cat}' ORDER BY price_cents DESC,sku ASC"
    return (
        f"当前标价最高的前{first}个{cat}商品，输出sku、标价，价格降序并以sku升序打破并列",
        stem + f" LIMIT {first}",
        {
            "category": cat,
            "limit": first,
            "metric": "商品目录列表",
            "outputs": ["sku", "price_cents"],
            "sort": ["-price_cents", "+sku"],
        },
        f"排名数量替换成前{second}个，保留输出和排序",
        stem + f" LIMIT {second}",
        {"limit": second},
        "取消排名数量限制，保留输出和排序",
        stem,
        {"limit": None},
    )


def build(split):
    cases, gold = [], {}
    design = (
        [(family, v, 4) for family in (5, 10, 8, 11) for v in range(4)]
        if split == "development"
        else [(family, 8 + v, length) for family in range(12) for v, length in enumerate((12, 44))]
    )
    for family, variant, length in design:
        name = f"经营核对{family}号{variant}"
        sid = f"{split[0]}-f{family:02d}-v{variant}"
        turns = []
        q0, s0, state0, q1, s1, delta1, q2, s2, delta2 = session_cases(family, variant)
        state1 = state0 | delta1
        state2 = {k: v for k, v in (state1 | delta2).items() if v is not None}
        removed = sorted(k for k, v in delta2.items() if v is None)

        def add(
            question,
            sql,
            bucket,
            expected,
            source=None,
            deleted=(),
            operator_changes=None,
            *,
            turns=turns,
            sid=sid,
            family=family,
            length=length,
        ):
            turn_no = len(turns) + 1
            tid = f"{sid}-{turn_no:02d}"
            turns.append({"id": tid, "question": question + "。"})
            source_id = turns[source - 1]["id"] if source else None
            inherited = gold[source_id]["operators"] if source_id else {}
            authored = inherited | (operator_changes or {})
            defaults = {
                "date": "range",
                "price": "gt",
                "metric": "describe",
                "outputs": "describe",
                "sort": "describe",
            }
            operators = {
                key: authored.get(key, defaults.get(key, "eq")) for key in (expected or {})
            }
            gold[tid] = {
                "sql": sql,
                "state": expected,
                "removed": list(deleted),
                "bucket": bucket,
                "operators": operators,
                "ordered": True,
                "tags": [bucket],
                "clarify": bucket == "clarification",
                "family": family,
                "length": length,
                "source_turn": source,
                "source_id": source_id,
                "distance": turn_no - source if source else None,
            }

        def filler(index, *, family=family, variant=variant):
            city = CITIES[(family + variant + index) % 3]
            cat = CATEGORIES[(family + index) % 3]
            filler_status = STATUSES[(variant + index) % 3]
            q, sql, state = (
                (
                    f"{city}客户人数，输出人数",
                    f"SELECT COUNT(*) FROM customers WHERE city='{city}'",
                    {"city": city, "metric": "客户人数", "outputs": ["metric"]},
                ),
                (
                    f"{cat}目录商品种数，输出数量",
                    f"SELECT COUNT(*) FROM products WHERE category='{cat}'",
                    {"category": cat, "metric": "商品种数", "outputs": ["metric"]},
                ),
                (
                    f"{filler_status}订单中有多少种不同商品，输出种数",
                    f"SELECT COUNT(DISTINCT i.product_id) FROM order_items i JOIN orders o USING (order_id) WHERE o.status='{filler_status}'",
                    {"status": filler_status, "metric": "曾售商品种数", "outputs": ["metric"]},
                ),
                (
                    f"{cat}目录商品的当前平均标价，输出平均值",
                    f"SELECT AVG(price_cents) FROM products WHERE category='{cat}'",
                    {"category": cat, "metric": "商品当前平均标价", "outputs": ["metric"]},
                ),
            )[index % 4]
            add("另做一个独立检查：" + q, sql, "independent", state)

        first_ops = {"category": "ne"} if family in {5, 11} else None
        set_ops = {"category": "eq"} if family == 11 else None
        add(f"新建命名分析“{name}”：{q0}", s0, "independent", state0, operator_changes=first_ops)
        add(f"继续“{name}”，{q1}", s1, "recent", state1, 1, operator_changes=set_ops)
        add(f"修改“{name}”，{q2}", s2, "correction", state2, 2, removed)
        if length == 4:
            keep = (
                "保留输出和排序",
                "保持字段、列顺序和排序",
                "沿用所有条件及输出",
                "保留筛选、分组、输出及排序",
            )[variant]
            selector = ("修改后的最新版", "当前版本", "的最新版本", "turn:3")[variant]
            add(f"恢复“{name}”{selector}，{keep}，重新查询", s2, "latest", state2, 3, removed)
        else:
            clarification = 5 if length == 12 else 4
            for turn_no in range(4, length - 4):
                if turn_no == clarification:
                    add(
                        "恢复“本会话尚未描述的复购分析”，完整继承全部条件",
                        None,
                        "clarification",
                        None,
                    )
                else:
                    filler(turn_no)
            original = length - 4
            original_selector = "的最初版本" if length == 12 else "turn:1"
            add(
                f"恢复“{name}”{original_selector}，完全沿用当时的统计要求，重新查询",
                s0,
                "distant",
                state0,
                1,
            )
            add(
                f"回到“{name}”刚恢复的那个版本，{q1}",
                s1,
                "correction",
                state1,
                original,
                operator_changes=set_ops,
            )
            filler(length - 2)
            latest_selector = (
                "调整后的最新版",
                "更新后的最新版本",
                "修订后的最新版",
                "改动后的最新版本",
            )[(family + variant) % 4]
            add(
                f"恢复“{name}”{latest_selector}，保持原有筛选、输出及排序，重新查询",
                s1,
                "latest",
                state1,
                original + 1,
            )
            filler(length)
        assert len(turns) == length
        cases.append(
            {"id": sid, "database_id": "business_orders_v1", "family": family, "turns": turns}
        )
    return cases, gold


def create_instances(destination, schema, split):
    """One primary and three distinguishers, with separate seeded rows for each split."""
    seeds = (22111, 22112, 22113, 22114) if split == "development" else (22151, 22152, 22153, 22154)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, seed in zip(("primary", "catalogue", "grouping", "conditions"), seeds, strict=True):
        path = destination / f"{key}.sqlite"
        if path.exists():
            raise FileExistsError(path)
        rng = random.Random(seed)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(schema)
            customers = [(k, f"Customer {k % 4}", CITIES[(k - 1) % 3]) for k in range(1, 16)]
            products = [
                (
                    k,
                    f"Product {k}",
                    f"P{k:03d}",
                    CATEGORIES[(k - 1) % 3],
                    550 + 175 * k + rng.choice((0, 50, 200)),
                )
                for k in range(1, 22)
            ]
            db.executemany("INSERT INTO customers VALUES (?,?,?)", customers)
            db.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
            oid = 1
            for month in range(1, 13):
                for cid in rng.sample(range(1, 13), 10 if key == "grouping" else 8):
                    status = rng.choice(STATUSES)
                    if key == "conditions" and month in {5, 10}:
                        status = "pending"
                    db.execute(
                        "INSERT INTO orders VALUES (?,?,?,?)",
                        (oid, cid, status, f"2026-{month:02d}-{rng.randint(1, 28):02d}"),
                    )
                    for pid in rng.sample(
                        range(1, 10 if key == "catalogue" else 15), rng.randint(1, 3)
                    ):
                        db.execute(
                            "INSERT INTO order_items VALUES (?,?,?,?)",
                            (
                                oid,
                                pid,
                                rng.randint(1, 6),
                                products[pid - 1][-1] + rng.choice((-150, 125)),
                            ),
                        )
                    oid += 1
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        paths[key] = path
    return paths


COUNTEREXAMPLES = {
    "group_by_customer": (
        "SELECT city,COUNT(*) FROM orders JOIN customers USING(customer_id) GROUP BY city ORDER BY city",
        "SELECT city,COUNT(*) FROM orders JOIN customers USING(customer_id) GROUP BY customer_id ORDER BY city",
    ),
    "sold_instead_of_catalogue": (
        "SELECT COUNT(*) FROM products",
        "SELECT COUNT(DISTINCT product_id) FROM order_items",
    ),
    "extra_order_condition": (
        "SELECT AVG(price_cents) FROM products",
        "SELECT AVG(price_cents) FROM products WHERE product_id IN (SELECT product_id FROM order_items JOIN orders USING(order_id) WHERE status='completed')",
    ),
    "retained_category": (
        "SELECT COUNT(*) FROM products",
        "SELECT COUNT(*) FROM products WHERE category='Office'",
    ),
    "dropped_projection": (
        "SELECT city,COUNT(*) FROM customers GROUP BY city ORDER BY city",
        "SELECT COUNT(*) FROM customers GROUP BY city ORDER BY city",
    ),
    "count_instead_of_units": (
        "SELECT SUM(quantity) FROM order_items",
        "SELECT COUNT(*) FROM order_items",
    ),
    "zero_instead_of_null": (
        "SELECT AVG(price_cents) FROM products WHERE price_cents<0",
        "SELECT COALESCE(AVG(price_cents), 0) FROM products WHERE price_cents<0",
    ),
}


def validate_references(gold, paths):
    for case_id, label in gold.items():
        state = label["state"] or {}
        sql = label["sql"] or ""
        if "category" in state:
            predicate = re.search(r"\bcategory\s*(<>|=)\s*'([^']+)'", sql)
            expected = ("<>" if label["operators"]["category"] == "ne" else "=", state["category"])
            if predicate is None or predicate.groups() != expected:
                raise ValueError(f"Category reference disagrees with state for {case_id}")
        if label["bucket"] in {"latest", "distant"}:
            source = gold[label["source_id"]]
            if any(label[key] != source[key] for key in ("state", "operators", "removed", "sql")):
                raise ValueError(f"Restoration reference disagrees with source for {case_id}")
    references = {label["sql"] for label in gold.values() if label["sql"]}
    coverage = {key: [] for key in COUNTEREXAMPLES}
    for key, path in paths.items():
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for sql in references:
                cursor = db.execute(sql)
                rows = cursor.fetchall()
                if (
                    cursor.description is None
                    or len(rows) > 100
                    or len(json.dumps(rows).encode()) > 3000
                ):
                    raise ValueError(f"Reference exceeds product result contract in {key}: {sql}")
            for name, (correct, wrong) in COUNTEREXAMPLES.items():
                if db.execute(correct).fetchall() != db.execute(wrong).fetchall():
                    coverage[name].append(key)
    if not all(coverage.values()):
        raise ValueError(f"Undistinguished SQL variants: {coverage}")
    return {"unique_reference_queries": len(references), "coverage": coverage}


def prepare(root, destination, split):
    conversations, gold = build(split)
    schema = (root / "data/processed/business_orders_v1/schema.sql").read_text(encoding="utf-8")
    paths = create_instances(destination, schema, split)
    oracle = validate_references(gold, paths)
    for name, value in (("questions", conversations), ("gold", gold), ("oracle", oracle)):
        with (destination / f"{name}.json").open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    files = [
        *paths.values(),
        *(destination / f"{name}.json" for name in ("questions", "gold", "oracle")),
    ]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
