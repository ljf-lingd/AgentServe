"""Build an AgentServe read-only fixture from UCI Online Retail II (dataset 502)."""

import argparse
import hashlib
import io
import json
import sqlite3
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

SOURCE = "https://doi.org/10.24432/C5CG6D"
EXPECTED_ROWS = 1_067_371
PRICE_SCALE = 1_000_000
SCHEMA = """CREATE TABLE retail_lines (
    line_id INTEGER PRIMARY KEY,
    source_sheet TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    invoice TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL,
    invoice_date TEXT NOT NULL,
    unit_price_millionths INTEGER NOT NULL,
    customer_id TEXT,
    country TEXT,
    is_cancellation INTEGER GENERATED ALWAYS AS (UPPER(SUBSTR(invoice,1,1)) = 'C') STORED,
    sales_millionths INTEGER GENERATED ALWAYS AS (
        CASE WHEN NOT is_cancellation AND quantity > 0 AND unit_price_millionths > 0
            THEN quantity * unit_price_millionths ELSE 0 END
    ) STORED,
    cancellation_millionths INTEGER GENERATED ALWAYS AS (
        CASE WHEN is_cancellation AND quantity < 0 AND unit_price_millionths > 0
            THEN -quantity * unit_price_millionths ELSE 0 END
    ) STORED
);
"""
VIEW = """CREATE VIEW retail_transactions AS
SELECT invoice, stock_code, description, quantity, invoice_date, customer_id, country,
       is_cancellation,
       CASE WHEN sales_millionths > 0 THEN sales_millionths / 1000000.0 END AS sales_gbp,
       CASE WHEN cancellation_millionths > 0 THEN cancellation_millionths / 1000000.0 END AS cancellation_gbp
FROM retail_lines;
"""
QUERY_SCHEMA = """-- Query interface for the retail_transactions view. Each row is one source line.
retail_transactions(
  invoice TEXT, stock_code TEXT, description TEXT, quantity INTEGER,
  invoice_date TEXT, customer_id TEXT, country TEXT,
  is_cancellation INTEGER, sales_gbp REAL, cancellation_gbp REAL
)
"""
INDEXES = """CREATE INDEX retail_date ON retail_lines(invoice_date);
CREATE INDEX retail_country_date ON retail_lines(country, invoice_date);
CREATE INDEX retail_stock_date ON retail_lines(stock_code, invoice_date);
"""
COLUMNS = ("invoice", "stockcode", "description", "quantity", "invoicedate", "price",
           "customerid", "country")


def sheet_rows(sheet, audit):
    rows = sheet.values
    header = tuple(str(value).lower().replace(" ", "") for value in next(rows))
    if header != COLUMNS:
        raise ValueError(f"{sheet.title}: unexpected header {header}")
    for row_no, values in enumerate(rows, start=2):
        invoice, stock, description, quantity, date, price, customer, country = values
        if not isinstance(date, datetime) or invoice is None or stock is None:
            raise ValueError(f"{sheet.title}:{row_no}: missing invoice, product or date")
        scaled = Decimal(str(price)) * PRICE_SCALE
        if scaled != scaled.to_integral_value() or quantity != int(quantity):
            raise ValueError(f"{sheet.title}:{row_no}: non-integral quantity or scaled price")
        audit["rows"] += 1
        audit["null_customer"] += customer is None
        audit["null_description"] += description is None
        audit["null_country"] += country is None
        cancelled = str(invoice).upper().startswith("C")
        audit["cancellation_lines"] += cancelled
        audit["sales_lines"] += not cancelled and quantity > 0 and scaled > 0
        audit["valued_cancellation_lines"] += cancelled and quantity < 0 and scaled > 0
        audit["adjustment_lines"] += not (
            scaled > 0 and ((not cancelled and quantity > 0) or (cancelled and quantity < 0))
        )
        audit["absolute_amount_bound"] += abs(int(quantity) * int(scaled))
        yield (
            sheet.title, row_no, str(invoice), str(stock), description, int(quantity),
            date.isoformat(sep=" ", timespec="seconds"), int(scaled),
            None if customer is None else str(int(customer)), country,
        )


def content_hash(database):
    """Hash normalized source rows in workbook order, independent of SQLite page layout."""
    hasher = hashlib.sha256()
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for row in db.execute(
            """SELECT source_sheet,source_row,invoice,stock_code,description,quantity,
               invoice_date,unit_price_millionths,customer_id,country
               FROM retail_lines ORDER BY line_id"""
        ):
            hasher.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
            hasher.update(b"\n")
    db.close()
    return hasher.hexdigest()


def build(source, destination):
    from openpyxl import load_workbook

    destination.mkdir(parents=True, exist_ok=True)
    db_path = destination / "online_retail_ii.sqlite"
    stage = destination / "online_retail_ii.importing.sqlite"
    if db_path.exists() or stage.exists():
        raise FileExistsError(f"Use a fresh output directory: {destination}")
    payload = source.read_bytes()
    if source.suffix.lower() == ".zip":
        with ZipFile(io.BytesIO(payload)) as archive:
            (member,) = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
            workbook_bytes = archive.read(member)
    else:
        workbook_bytes = payload

    workbook = load_workbook(io.BytesIO(workbook_bytes), read_only=True, data_only=True)
    manifest = {
        "source": SOURCE, "license": "CC BY 4.0", "source_sha256": hashlib.sha256(payload).hexdigest(),
        "workbook_sha256": hashlib.sha256(workbook_bytes).hexdigest(),
        "price_scale": PRICE_SCALE, "sheets": {},
    }
    try:
        if len(workbook.worksheets) != 2:
            raise ValueError(f"Expected two sheets, got {workbook.sheetnames}")
        with sqlite3.connect(stage) as db:
            db.executescript(SCHEMA)
            for sheet in workbook:
                audit = Counter()
                db.executemany(
                    """INSERT INTO retail_lines(source_sheet,source_row,invoice,stock_code,
                    description,quantity,invoice_date,unit_price_millionths,customer_id,country)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    sheet_rows(sheet, audit),
                )
                manifest["sheets"][sheet.title] = dict(audit)
            total, first, last, countries = db.execute(
                """SELECT count(*),date(min(invoice_date)),date(max(invoice_date)),
                count(DISTINCT country) FROM retail_lines"""
            ).fetchone()
            if total != EXPECTED_ROWS:
                raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, loaded {total:,}")
            if sum(s["absolute_amount_bound"] for s in manifest["sheets"].values()) >= 2**63:
                raise ValueError("A full-table integer amount aggregate can overflow")
            db.executescript(INDEXES)
            db.execute("ANALYZE")
            db.executescript(VIEW)
            manifest.update(rows=total, first_date=first, last_date=last, countries=countries)
        db.close()
        stage.replace(db_path)
    finally:
        workbook.close()
    (destination / "schema.sql").write_text(QUERY_SCHEMA, encoding="utf-8")
    with db_path.open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["database_sha256"] = digest.hexdigest()
    manifest["content_sha256"] = content_hash(db_path)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official UCI ZIP or workbook")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination), indent=2))

