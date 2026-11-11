"""Frozen synthetic order-analysis acceptance data; never represents production traffic."""

import hashlib
import random
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

from agentserve.workloads.spider import write_frozen

SCHEMA = """CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL);
CREATE TABLE products(product_id INTEGER PRIMARY KEY, name TEXT NOT NULL, sku TEXT NOT NULL, category TEXT NOT NULL, price_cents INTEGER NOT NULL);
CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id), status TEXT NOT NULL CHECK(status IN ('completed','pending','cancelled')), order_date TEXT NOT NULL);
CREATE TABLE order_items(order_id INTEGER NOT NULL REFERENCES orders(order_id), product_id INTEGER NOT NULL REFERENCES products(product_id), quantity INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL, PRIMARY KEY(order_id,product_id));
"""
CONTEXT = (
    "This is synthetic acceptance data, not production orders. Dates use YYYY-MM-DD in 2026. "
    "All monetary values are integer cents; preserve cents in answers. Revenue includes only "
    "completed orders and is SUM(order_items.quantity * order_items.unit_price_cents), using "
    "historical item prices, not products.price_cents. Include pending/cancelled only when asked. "
    "City/category/name/sku values are case-sensitive text. Names may repeat; customer_id identifies people."
)
JOIN = " FROM orders o JOIN order_items i ON i.order_id=o.order_id"
CITY_JOIN = JOIN + " JOIN customers c ON c.customer_id=o.customer_id"
PRODUCT_JOIN = JOIN + " JOIN products p ON p.product_id=i.product_id"

# Questions are frozen before model runs. Gold is stored separately from runtime inputs.
CASES = [
    ("customers", "基础计数", "共有多少位客户？", "SELECT COUNT(*) FROM customers"),
    ("orders", "基础计数", "所有状态合计有多少笔订单？", "SELECT COUNT(*) FROM orders"),
    (
        "completed",
        "状态",
        "已完成（completed）的订单有多少笔？",
        "SELECT COUNT(*) FROM orders WHERE status='completed'",
    ),
    (
        "pending",
        "状态",
        "待处理（pending）订单有多少笔？",
        "SELECT COUNT(*) FROM orders WHERE status='pending'",
    ),
    (
        "cancelled",
        "状态",
        "已取消（cancelled）订单有多少笔？",
        "SELECT COUNT(*) FROM orders WHERE status='cancelled'",
    ),
    (
        "january",
        "日期",
        "2026年1月共有多少笔订单？统计所有状态。",
        "SELECT COUNT(*) FROM orders WHERE order_date>='2026-01-01' AND order_date<'2026-02-01'",
    ),
    (
        "february",
        "日期",
        "2026年2月有多少笔已完成订单？",
        "SELECT COUNT(*) FROM orders WHERE status='completed' AND order_date>='2026-02-01' AND order_date<'2026-03-01'",
    ),
    (
        "jan_last_day",
        "边界日期",
        "2026年1月31日有多少笔订单？包含所有状态。",
        "SELECT COUNT(*) FROM orders WHERE order_date='2026-01-31'",
    ),
    (
        "feb_first_day",
        "边界日期",
        "2026年2月1日有多少笔订单？包含所有状态。",
        "SELECT COUNT(*) FROM orders WHERE order_date='2026-02-01'",
    ),
    (
        "revenue",
        "金额口径",
        "所有已完成订单的营收合计多少分？",
        "SELECT SUM(i.quantity*i.unit_price_cents)" + JOIN + " WHERE o.status='completed'",
    ),
    (
        "jan_revenue",
        "金额日期",
        "2026年1月已完成订单营收是多少分？",
        "SELECT SUM(i.quantity*i.unit_price_cents)"
        + JOIN
        + " WHERE o.status='completed' AND o.order_date>='2026-01-01' AND o.order_date<'2026-02-01'",
    ),
    (
        "pending_amount",
        "状态金额",
        "只统计pending订单，订单项金额总计多少分？",
        "SELECT SUM(i.quantity*i.unit_price_cents)" + JOIN + " WHERE o.status='pending'",
    ),
    (
        "beijing_revenue",
        "城市金额",
        "Beijing客户的已完成订单营收总计多少分？",
        "SELECT SUM(i.quantity*i.unit_price_cents)"
        + CITY_JOIN
        + " WHERE o.status='completed' AND c.city='Beijing'",
    ),
    (
        "city_revenue",
        "城市聚合",
        "列出各city的已完成订单营收（分），只返回city和营收，按city升序。",
        "SELECT c.city,SUM(i.quantity*i.unit_price_cents)"
        + CITY_JOIN
        + " WHERE o.status='completed' GROUP BY c.city ORDER BY c.city",
    ),
    (
        "status_counts",
        "分组",
        "各status的订单数量分别是多少？只返回status和数量，按status升序。",
        "SELECT status,COUNT(*) FROM orders GROUP BY status ORDER BY status",
    ),
    (
        "month_counts",
        "日期分组",
        "按2026年的月份列出所有状态订单数，只返回月份和数量，按月份升序。",
        "SELECT substr(order_date,1,7),COUNT(*) FROM orders GROUP BY substr(order_date,1,7) ORDER BY substr(order_date,1,7)",
    ),
    (
        "city_customers",
        "客户聚合",
        "每个city有多少位客户？只返回city和人数，按city升序。",
        "SELECT city,COUNT(*) FROM customers GROUP BY city ORDER BY city",
    ),
    (
        "category_products",
        "商品聚合",
        "每个category有多少种商品？返回category和种数，按category升序。",
        "SELECT category,COUNT(*) FROM products GROUP BY category ORDER BY category",
    ),
    (
        "product_top3",
        "排序",
        "列出已完成订单销量最高的3种商品名称和总quantity，按销量降序、product_id升序；不要返回product_id。",
        "SELECT p.name,SUM(i.quantity) AS units"
        + PRODUCT_JOIN
        + " WHERE o.status='completed' GROUP BY p.product_id,p.name ORDER BY units DESC,p.product_id ASC LIMIT 3",
    ),
    (
        "product_price",
        "排序",
        "标价最高的商品名称和price_cents是多少？并列时取product_id最小的商品，只返回名称与标价（分）。",
        "SELECT name,price_cents FROM products ORDER BY price_cents DESC,product_id ASC LIMIT 1",
    ),
    (
        "zero_customers",
        "零订单",
        "列出没有任何订单的客户姓名，按customer_id升序，只返回姓名。",
        "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id=c.customer_id WHERE o.order_id IS NULL ORDER BY c.customer_id",
    ),
    (
        "customer_counts",
        "输出列",
        "列出每位客户的姓名与订单数，包含零订单客户，按customer_id升序，只输出姓名和订单数。",
        "SELECT c.name,COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON o.customer_id=c.customer_id GROUP BY c.customer_id,c.name ORDER BY c.customer_id",
    ),
    (
        "first10_customers",
        "输出列",
        "按customer_id升序列出前10位客户的姓名和city，只返回姓名与city。",
        "SELECT name,city FROM customers ORDER BY customer_id LIMIT 10",
    ),
    (
        "all_names",
        "截断",
        "列出全部客户姓名，按customer_id升序；只返回姓名。",
        "SELECT name FROM customers ORDER BY customer_id",
    ),
    (
        "empty_city",
        "空结果",
        "列出city为Atlantis的客户姓名，按customer_id升序。",
        "SELECT name FROM customers WHERE city='Atlantis' ORDER BY customer_id",
    ),
    (
        "empty_period",
        "空结果",
        "列出2025年12月的订单编号，按order_id升序。",
        "SELECT order_id FROM orders WHERE order_date>='2025-12-01' AND order_date<'2026-01-01' ORDER BY order_id",
    ),
    (
        "null_sum",
        "NULL",
        "2025年12月已完成订单的金额SUM是多少分？保留SQL的NULL语义，不要把NULL改成0。",
        "SELECT SUM(i.quantity*i.unit_price_cents)"
        + JOIN
        + " WHERE o.status='completed' AND o.order_date>='2025-12-01' AND o.order_date<'2026-01-01'",
    ),
    (
        "distinct_buyers",
        "去重",
        "至少有一笔completed订单的不同客户共有多少位？",
        "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE status='completed'",
    ),
    (
        "joined_order_count",
        "连接去重",
        "包含Office类别商品的已完成订单有多少笔？同一订单只计一次。",
        "SELECT COUNT(DISTINCT o.order_id)"
        + PRODUCT_JOIN
        + " WHERE o.status='completed' AND p.category='Office'",
    ),
    (
        "sold_units",
        "销量",
        "已完成订单合计售出多少件商品？按order_items.quantity求和。",
        "SELECT SUM(i.quantity)" + JOIN + " WHERE o.status='completed'",
    ),
    (
        "historical_price",
        "历史价格",
        "order_id为1的订单项金额合计多少分？使用订单项历史单价，不使用商品当前标价。",
        "SELECT SUM(quantity*unit_price_cents) FROM order_items WHERE order_id=1",
    ),
    (
        "average_order",
        "二级聚合",
        "已完成订单的平均每单金额是多少分？先对每笔订单汇总，再求平均，四舍五入到小数点后2位。",
        "SELECT ROUND(AVG(total),2) FROM (SELECT SUM(i.quantity*i.unit_price_cents) AS total"
        + JOIN
        + " WHERE o.status='completed' GROUP BY o.order_id)",
    ),
    (
        "category_revenue",
        "金额聚合",
        "列出各category已完成订单营收（分），按category升序，只返回类别和营收。",
        "SELECT p.category,SUM(i.quantity*i.unit_price_cents)"
        + PRODUCT_JOIN
        + " WHERE o.status='completed' GROUP BY p.category ORDER BY p.category",
    ),
    (
        "customer_one",
        "实体",
        "customer_id为1的客户共有多少笔订单？含所有状态。",
        "SELECT COUNT(*) FROM orders WHERE customer_id=1",
    ),
    (
        "duplicate_name",
        "重名",
        "姓名为Alex的客户有多少位？按客户记录计数。",
        "SELECT COUNT(*) FROM customers WHERE name='Alex'",
    ),
    (
        "no_completed",
        "零完成订单",
        "列出没有completed订单的客户姓名，按customer_id升序，只输出姓名。",
        "SELECT c.name FROM customers c WHERE NOT EXISTS(SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id AND o.status='completed') ORDER BY c.customer_id",
    ),
    (
        "sku_normalized",
        "字符串",
        "列出product_id前3个商品的sku，去掉sku中的所有连字符，按product_id升序，只返回处理后的sku。",
        "SELECT REPLACE(sku,'-','') FROM products ORDER BY product_id LIMIT 3",
    ),
    (
        "name_normalized",
        "字符串",
        "product_id为1的商品名称去掉连字符后是什么？只输出处理后的名称。",
        "SELECT REPLACE(name,'-','') FROM products WHERE product_id=1",
    ),
    (
        "like_customer",
        "文本过滤",
        "姓名以Customer00开头的客户有多少位？",
        "SELECT COUNT(*) FROM customers WHERE name LIKE 'Customer00%'",
    ),
    (
        "largest_order",
        "并列排序",
        "金额最高的一笔订单的order_id与金额（分）是多少？包含所有状态；并列取order_id最小。",
        "SELECT o.order_id,SUM(i.quantity*i.unit_price_cents) AS total"
        + JOIN
        + " GROUP BY o.order_id ORDER BY total DESC,o.order_id ASC LIMIT 1",
    ),
]


def prepare(root: Path):
    rng = random.Random(20261102)
    cities = ["Beijing", "Shanghai", "Shenzhen", "Chengdu", "Hangzhou", "Wuhan"]
    customers = [
        (i, "Alex" if i in (1, 2) else f"Customer{i:03d}", cities[(i - 1) % 6])
        for i in range(1, 121)
    ]
    products = [
        (
            i,
            f"Product-{i:02d}",
            f"SKU-{i:04d}",
            ["Electronics", "Office", "Home"][(i - 1) % 3],
            500 + i * 375,
        )
        for i in range(1, 25)
    ]
    orders, items = [], []
    for i in range(1, 1001):
        status = rng.choices(["completed", "pending", "cancelled"], [60, 25, 15])[0]
        day = date(2026, 1, 1) + timedelta(days=rng.randrange(181))
        if i in (999, 1000):
            day, status = date(2026, 1, 31) + timedelta(days=i - 999), "completed"
        orders.append((i, (i - 1) % 110 + 1, status, day.isoformat()))
        for product in sorted(rng.sample(range(1, 25), rng.randint(1, 4))):
            items.append((i, product, rng.randint(1, 4), products[product - 1][4] - (i % 7) * 50))
    gold = []
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES(?,?,?)", customers)
        conn.executemany("INSERT INTO products VALUES(?,?,?,?,?)", products)
        conn.executemany("INSERT INTO orders VALUES(?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_items VALUES(?,?,?,?)", items)
        conn.commit()
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        for key, category, question, sql in CASES:
            cursor = conn.execute(sql)
            gold.append(
                {
                    "id": "business." + key,
                    "category": category,
                    "sql": sql,
                    "columns": [c[0] for c in cursor.description],
                    "rows": cursor.fetchall(),
                    "ordered": True,
                }
            )
        completed_ids = {o[0] for o in orders if o[2] == "completed"}
        revenue = sum(q * p for order, _, q, p in items if order in completed_ids)
        assert next(g for g in gold if g["id"] == "business.revenue")["rows"] == [(revenue,)]
        assert next(g for g in gold if g["id"] == "business.null_sum")["rows"] == [(None,)]
        database_bytes = conn.serialize()
    path = root / "data/databases/business_orders_v1/business_orders_v1.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != database_bytes:
            raise ValueError("Frozen business database differs; use a new version")
    else:
        path.write_bytes(database_bytes)
    path.chmod(0o444)
    schema_path = root / "data/processed/business_orders_v1/schema.sql"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    if schema_path.exists() and schema_path.read_text(encoding="utf-8") != SCHEMA:
        raise ValueError("Frozen business schema differs")
    schema_path.write_text(SCHEMA, encoding="utf-8")
    questions = [
        {"id": "business." + key, "database_id": "business_orders_v1", "question": q}
        for key, _, q, _ in CASES
    ]
    write_frozen(root / "data/processed/business_orders_v1/questions.json", questions)
    write_frozen(root / "data/gold/business_orders_v1/answers.json", gold)
    manifest = {
        "version": "business_orders_v1",
        "synthetic": True,
        "seed": 20261102,
        "rows": {
            "customers": len(customers),
            "products": len(products),
            "orders": len(orders),
            "order_items": len(items),
        },
        "question_count": len(questions),
        "database_sha256": hashlib.sha256(database_bytes).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA.encode()).hexdigest(),
        "database": {
            "path": str(path.relative_to(root)),
            "schema_path": str(schema_path.relative_to(root)),
            "context": CONTEXT,
        },
    }
    write_frozen(root / "data/processed/business_orders_v1/manifest.json", manifest)
    return manifest
