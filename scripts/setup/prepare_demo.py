"""Create deterministic preflight fixtures; keep questions separate from gold SQL."""

import hashlib
import json
import random
import sqlite3
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    category TEXT NOT NULL, price_cents INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL CHECK(status IN ('completed', 'pending', 'cancelled')),
    order_date TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
"""
PROBES = [
    ("customers_count", "一共有多少位客户？", "SELECT COUNT(*) FROM customers"),
    ("orders_count", "一共有多少笔订单？", "SELECT COUNT(*) FROM orders"),
    ("completed_count", "状态为 completed 的订单有多少笔？", "SELECT COUNT(*) FROM orders WHERE status='completed'"),
    ("products_count", "一共有多少种商品？", "SELECT COUNT(*) FROM products"),
    ("shanghai_count", "city 为 Shanghai 的客户有多少位？", "SELECT COUNT(*) FROM customers WHERE city='Shanghai'"),
    ("beijing_names", "列出 city 为 Beijing 的客户姓名，按 name 升序。", "SELECT name FROM customers WHERE city='Beijing' ORDER BY name"),
    ("completed_revenue", "所有 completed 订单的总金额是多少分？金额用 quantity * unit_price_cents 计算。", "SELECT SUM(i.quantity*i.unit_price_cents) FROM order_items i JOIN orders o ON o.order_id=i.order_id WHERE o.status='completed'"),
    ("orders_per_customer", "列出每位客户的姓名和订单数，包含零订单客户，按 customer_id 升序。", "SELECT c.name,COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON o.customer_id=c.customer_id GROUP BY c.customer_id,c.name ORDER BY c.customer_id"),
    ("top_products", "只统计 completed 订单，列出销量最高的三种商品名称及总 quantity，按销量降序、product_id 升序。", "SELECT p.name,SUM(i.quantity) AS units FROM products p JOIN order_items i ON i.product_id=p.product_id JOIN orders o ON o.order_id=i.order_id WHERE o.status='completed' GROUP BY p.product_id,p.name ORDER BY units DESC,p.product_id ASC LIMIT 3"),
    ("no_orders", "列出没有任何订单的客户姓名，按 customer_id 升序。", "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id=c.customer_id WHERE o.order_id IS NULL ORDER BY c.customer_id"),
    ("revenue_per_city", "列出各 city 的 completed 订单总金额（分），按 city 升序。", "SELECT c.city,SUM(i.quantity*i.unit_price_cents) FROM customers c JOIN orders o ON o.customer_id=c.customer_id JOIN order_items i ON i.order_id=o.order_id WHERE o.status='completed' GROUP BY c.city ORDER BY c.city"),
    ("pending_orders", "列出 pending 订单的 order_id 和客户姓名，按 order_id 升序。", "SELECT o.order_id,c.name FROM orders o JOIN customers c ON c.customer_id=o.customer_id WHERE o.status='pending' ORDER BY o.order_id"),
    ("most_expensive", "列出标价 price_cents 最高的商品名称和标价（分），并列时取 product_id 最小的一种。", "SELECT name,price_cents FROM products ORDER BY price_cents DESC,product_id ASC LIMIT 1"),
    ("category_counts", "列出每个 category 及该类别的商品种数，按 category 升序。", "SELECT category,COUNT(*) FROM products GROUP BY category ORDER BY category"),
    ("january_orders", "2026年1月有多少笔订单？包括所有状态。", "SELECT COUNT(*) FROM orders WHERE order_date>='2026-01-01' AND order_date<'2026-02-01'"),
    ("empty_result", "列出 city 为 Atlantis 的客户姓名，按 customer_id 升序。", "SELECT name FROM customers WHERE city='Atlantis' ORDER BY customer_id"),
]


def main():
    customers = [(1, "Alice", "Beijing"), (2, "Bob", "Shanghai"),
                 (3, "Carol", "Shenzhen"), (4, "David", "Beijing"),
                 (5, "Eve", "Shanghai"), (6, "Frank", "Shenzhen")]
    products = [(1, "Keyboard", "electronics", 12900), (2, "Mouse", "electronics", 5900),
                (3, "Notebook", "stationery", 1500), (4, "Pen", "stationery", 500),
                (5, "Mug", "home", 3900)]
    statuses = ["completed", "completed", "pending", "cancelled", "completed", "completed",
                "pending", "completed", "completed", "completed", "cancelled", "completed"]
    orders = [(i, (i - 1) % 5 + 1, status,
               f"2026-{'01' if i <= 8 else '02'}-{i:02d}")
              for i, status in enumerate(statuses, 1)]
    rng = random.Random(SEED)
    items = [(order_id, product_id, rng.randint(1, 3), products[product_id - 1][3])
             for order_id, *_ in orders
             for product_id in sorted(rng.sample(range(1, 6), rng.randint(1, 3)))]
    questions, gold = [], []
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?,?,?)", customers)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)
        conn.commit()
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        for case_id, question, sql in PROBES:
            questions.append({"id": case_id, "database_id": "demo_orders", "question": question})
            gold.append({"id": case_id, "sql": sql, "rows": conn.execute(sql).fetchall(), "ordered": True})
        database_bytes = conn.serialize()

    database = ROOT / "data/databases/demo_orders/demo_orders.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        if database.read_bytes() != database_bytes:
            raise RuntimeError("Existing demo database differs; refusing to overwrite it")
    else:
        with database.open("xb") as stream:
            stream.write(database_bytes)
    database.chmod(0o444)
    artifacts = {
        "data/processed/demo_orders/questions.json": questions,
        "data/gold/demo_orders/answers.json": gold,
        "data/synthetic/preflight/scenarios.json": {
            "seed": SEED, "purpose": "future deterministic system tests, not a benchmark",
            "executed": False,
            "scenarios": [
                {"id": "single_round", "steps": [{"id": "llm1", "kind": "llm", "depends_on": [], "max_output_tokens": 64}, {"id": "sql1", "kind": "sql", "depends_on": ["llm1"], "delay_ms": 0}]},
                {"id": "two_rounds", "steps": [{"id": "llm1", "kind": "llm", "depends_on": [], "max_output_tokens": 64}, {"id": "sql1", "kind": "sql", "depends_on": ["llm1"], "delay_ms": 0}, {"id": "llm2", "kind": "llm", "depends_on": ["sql1"], "max_output_tokens": 64}, {"id": "sql2", "kind": "sql", "depends_on": ["llm2"], "delay_ms": 0}]},
                {"id": "tool_wait_100ms", "steps": [{"id": "llm1", "kind": "llm", "depends_on": [], "max_output_tokens": 64}, {"id": "sql1", "kind": "sql", "depends_on": ["llm1"], "delay_ms": 100}]},
                {"id": "sql_error_once", "steps": [{"id": "llm1", "kind": "llm", "depends_on": [], "max_output_tokens": 64}, {"id": "sql1", "kind": "sql", "depends_on": ["llm1"], "inject_error": "SQL_EXECUTION_ERROR"}, {"id": "llm2", "kind": "llm", "depends_on": ["sql1"], "max_output_tokens": 64}]},
            ],
        },
    }
    for relative, content in artifacts.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path = ROOT / "data/processed/demo_orders/schema.sql"
    schema_path.write_text(SCHEMA.strip() + "\n", encoding="utf-8")
    manifest = {
        "fixture_version": 1, "seed": SEED, "database_id": "demo_orders",
        "database": str(database.relative_to(ROOT)),
        "database_sha256": hashlib.sha256(database_bytes).hexdigest(),
        "rows": {"customers": len(customers), "products": len(products),
                 "orders": len(orders), "order_items": len(items)},
        "questions": len(questions), "synthetic_scenarios": 4,
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "artifact_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in artifacts},
    }
    path = ROOT / "data/processed/demo_orders/manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
