"""SQL prompt content and message construction; independent of execution and storage."""

SYSTEM_PROMPT = (
    "Answer the question using execute_sql on the fixed SQLite database. "
    "Use exactly one read-only query per call. Never invent database values. "
    "Select only the output columns requested by the user; a sorting key can appear "
    "in ORDER BY without being returned. Verify each column's owning table against "
    "the schema. If execution fails, correct the SQL using the tool error. "
    "After a successful tool result, answer briefly in Chinese using only that result. "
    "Report truncation explicitly; do not present partial rows as complete.\n"
)
FORMAT_REPAIR_PROMPT = (
    "Invalid response format. Return exactly one execute_sql "
    "tool call with JSON arguments containing only sql."
)
PROJECTION_INSTRUCTIONS = (
    "Treat the requested SELECT columns as an ordered output contract. Keep their requested order. "
    "Columns used only for ORDER BY or GROUP BY must not enter SELECT. "
    "If only a count is requested, return one count column. For all data from a table, "
    "return that table's original columns once, adding only explicitly requested extra columns.\n"
)
CONCISE_ANSWER_PROMPT = (
    "请在120个汉字以内简洁回答。结构化表格已单独保存，不要逐行复述；必要时最多举3行并标注示例。"
    "只引用工具实际返回的数据，保留单位；NULL不能说成0，空结果应说明无匹配。"
    "若truncated为true，必须明确说明结果已截断，不能声称是全部数据或完整数量。"
    "不要自行推算未提供的总数，不要推测业务原因。"
)

EVIDENCE_INSTRUCTIONS = (
    "Removing ALL occurrences of a substring uses SQLite replace(text, old, ''). "
    "That operation already removes every position; do not wrap it in another string function. "
    "Order counts without an explicit status include ALL statuses; never leak the completed-only "
    "revenue default into COUNT queries. A customer absence-of-orders condition does not replace "
    "an explicit customer city filter: enforce both conditions. "
    "Units sold or item quantities means SUM of item quantity, never COUNT of rows or orders. "
    "When filtering orders by customer city, join customers.customer_id to orders.customer_id; "
    "order_id is not a customer identifier. Removing a filter means omitting that predicate, "
    "not requiring the column to be NULL. "
    "For category revenue, aggregate each line item by its own product category, joining "
    "order_items.product_id to products.product_id. Never assign an entire order to its first item. "
    "In the order schema, status belongs to orders, not order_items or products. "
    "A completed-only item aggregation must also join orders on order_id and filter orders.status. "
    "Respect the requested date granularity: a question naming a specific day requires that day "
    "only, not its whole month. Only a month WITHOUT a day means the complete month, using an "
    "inclusive start and exclusive next-month end. "
    "For follow-ups, apply current explicit changes to the identified prior analysis; retain every "
    "other explicit condition. A request to continue with complete evidence is permission to execute, "
    "not a request for confirmation. Clarify only when the reference or a required condition is absent. "
    "Resolved query state is the complete intent, not an unverified past SQL result.\n"
)


def initial_messages(
    question: str,
    *,
    schema: str,
    context: str,
    version: str = "baseline",
    functions: list[str] | None = None,
) -> list[dict]:
    instructions = SYSTEM_PROMPT + (PROJECTION_INSTRUCTIONS if version == "projection_v1" else "")
    if version in {"evidence_v21", "evidence_v22"}:
        instructions += EVIDENCE_INSTRUCTIONS
    system = instructions + context + "\nSchema:\n" + schema
    if version in {"evidence_v21", "evidence_v22"} and functions is not None:
        system += (
            "\nAllowed SQLite functions: "
            + ", ".join(functions)
            + ". Use only this list; use SQLite operators for other operations.\n"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
