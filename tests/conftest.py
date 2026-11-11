import json
import sqlite3
from pathlib import Path

import pytest

from agentserve.config import Settings


@pytest.fixture
def settings(tmp_path):
    database = tmp_path / "orders.sqlite"
    schema = "CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT);"
    with sqlite3.connect(database) as conn:
        conn.executescript(schema)
        conn.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text(schema)
    return Settings(
        databases={"demo_orders": {"path": database, "schema_path": schema_path}},
        state_database=tmp_path / "state.sqlite",
    )


def completion(
    sql="SELECT COUNT(*) FROM customers",
    *,
    answer=None,
    calls=1,
    finish_reason=None,
    arguments=None,
):
    message = {"role": "assistant", "content": answer}
    if answer is None:
        message["tool_calls"] = [
            {
                "id": f"call-{i}",
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "arguments": arguments or json.dumps({"sql": sql}),
                },
            }
            for i in range(calls)
        ]
    return {
        "id": "request-test",
        "choices": [
            {
                "message": message,
                "finish_reason": finish_reason or ("stop" if answer else "tool_calls"),
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


ROOT = Path(__file__).resolve().parents[1]
