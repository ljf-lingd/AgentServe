"""Bounded order-domain intent rules. No SQL templates, task IDs, gold or result values."""

import re

from agentserve.domain.models import AgentError

# Labels bind to logical fields. SQL aliases and source columns are chosen during compilation.
OUTPUT_FIELDS = {
    "city": "city",
    "城市": "city",
    "category": "category",
    "类别": "category",
    "status": "status",
    "状态": "status",
    "sku": "sku",
    "标价": "price_cents",
    "price_cents": "price_cents",
    "月份": "month",
    "YYYY-MM": "month",
    **dict.fromkeys(
        (
            "营收",
            "营收分",
            "笔数",
            "数量",
            "件数",
            "人数",
            "种数",
            "平均值",
            "平均标价",
            "订单笔数",
            "商品种数",
            "商品总种数",
            "去重商品种数",
        ),
        "metric",
    ),
}

KEEP_CLAUSE = re.compile(
    r"(?:完整继承|完全沿用|保留|沿用|保持)(?:(?:此前|当时|原有|原来|原先|现有|其余|其他|所有|全部|原|的|"
    r"条件|统计要求|要求|字段|筛选|范围|输出|排序|投影|列顺序|分组|月份|city|与|和|及|、|\s)+)"
)
ORDER_HEADER = re.compile(
    r"^\s*(?P<command>新建|建立|另开|创建|继续|修改|纠正|恢复|回到|沿用)\s*"
    r'(?:(?:命名)?分析\s*)?(?:[“「"](?P<name>[^”」"]+)[”」"])?'
)
VERSION_SELECTOR = re.compile(
    r"^\s*(?:的\s*)?(?:(?P<original>原始第一个版本|原始版本|最初版本|初始版本|第一个版本|第一版|原版|原始|最初|初始)|"
    r"(?P<explicit>turn:\d+)|(?P<latest>(?:(?:刚才)?(?:修改|调整|更新|修订|改动)后(?:的)?)?"
    r"(?:最新版本|最新版)|刚恢复的那个版本|当前版本))(?=$|[\s，。；：、？！,.?:;!])"
)

# Only the bounded vocabulary below can claim complete coverage. In particular, unknown
# qualifiers such as exclusions, extrema, averages per order, and non-schema filters remain verbatim.
ORDER_TERMS = re.compile(
    r"命名为|重新查询|"
    r"(?:(?:其他|其余)(?:条件(?:和分组)?|字段)?不变)|"
    r"取消分组和排序|(?:取消|移除|删除)(?:订单|商品|客户)?(?:类别|状态|城市|价格门槛|价格|排名数量|数量|分组|排序|日期|月份)(?:筛选|限制)?|"
    r"(?:订单状态|商品类别|分组维度|价格门槛|排名数量|类别|状态|城市)?(?:替换成|改为|改成)|"
    r"(?:保留|沿用|保持)(?:统计要求|所有条件|全部条件|字段|投影|输出|排序|列顺序|分组|筛选|月份|city|、|和|及|与)+|"
    r"\d{4}年\d{1,2}月(?:\d{1,2}日)?|上个月|本月|这个月|"
    r"(?:不低于|不少于|不超过|高于|大于|超过|低于|小于|至多)\s*\d+(?:\.\d+)?\s*分|前\s*\d+\s*(?:个|名|种)|"
    r"另做一个独立检查|不修改此前命名专题|这是另一个独立问题|独立统计|独立检查|"
    r"从未下过任何订单|从未下单|至少有一笔|去重|不同|目录|曾售|售出件数|已售|"
    r"商品当前平均标价|当前平均标价|平均标价|标价均值|营收合计|订单笔数|商品总种数|商品种数|销量总件数|"
    r"价格降序|价格升序|标价降序|标价升序|降序|升序|并以|打破并列|当前标价最高的|"
    r"不属于|所有订单状态|全部订单状态|不限制状态|不再分组|"
    r"price_cents|YYYY-MM|Electronics|Beijing|Shanghai|Shenzhen|completed|cancelled|pending|Office|Home|"
    r"category|status|city|sku|"
    r"单位分|营收分|成交金额|只统计|只输出|仅返回|只返回|返回|输出|"
    r"有多少种|多少种|有多少位|多少位|多少笔|多少订单|有哪些|总件数|件数|数量|笔数|人数|种数|平均值|营收|"
    r"当前标价|标价|价格|客户|城市|类别|商品|订单项|订单|状态|统计|汇总|分组|月份|共有|"
    r"所有|全部|各|按|仍|仅|的|中|分|[\s，。；：、？！,.?:;]"
)


def order_question_parts(question):
    """Consume the command and one complete selector before inspecting business clauses."""
    header = ORDER_HEADER.match(question)
    if header is None:
        return None, None, None, question
    command, name = header.group("command", "name")
    body = question[header.end() :]
    selector = None if command in {"新建", "建立", "另开", "创建"} else VERSION_SELECTOR.match(body)
    if selector:
        body = body[selector.end() :]
    return command, name, selector, body


def complete_order_question(question):
    _, _, _, body = order_question_parts(question)
    body = KEEP_CLAUSE.sub("", body)
    outputs = re.search(r"(?:只输出|仅返回|只返回|返回|输出)([^，。；？！]+)", body)
    if outputs and all(
        label.strip() in OUTPUT_FIELDS for label in re.split(r"、|和|及|与", outputs[1])
    ):
        body = body[: outputs.start()] + body[outputs.end() :]
    return not ORDER_TERMS.sub("", body)


def order_intent(question, entries, accepted_at):
    # Imported here because the typed contract is also used by the optional model fallback.
    from agentserve.memory.state import (
        StateChange,
        StateIntent,
        independent_request,
        normalize_date,
    )

    command, named, selector, body = order_question_parts(question)
    active = entries[-1]["state"]["name"] if entries else None
    independent = independent_request(question)
    creates = command in {"新建", "建立", "另开", "创建"}
    references = (
        bool(
            re.search(
                r"继续|恢复|回到|沿用|刚才|上次|其余不变|其他不变|修改|纠正|那组|这组|那个",
                question,
            )
            or re.fullmatch(
                r"\s*(?:城市\s*)?(?:改为|改成|替换成)\s*(?:Beijing|Shanghai|Shenzhen)\s*[。！？!?]?\s*",
                question,
            )
        )
        and not independent
    )
    if references and re.search(r"不存在|未定义|从未定义|尚未描述", question):
        raise AgentError("MEMORY_CONTEXT_REQUIRED", "该分析尚未定义，请补充完整条件。")
    action = (
        "new"
        if creates or not references
        else ("resume" if command in {"恢复", "回到"} else "refine")
    )
    name = named if action == "new" else (named or active)
    mentioned_names = set(re.findall(r'[“「"]([^”」"]+)[”」"]', question))
    known_names = {e["state"]["name"] for e in entries}
    if action != "new" and len(mentioned_names & known_names) > 1:
        raise AgentError("MEMORY_CONTEXT_REQUIRED", "请明确要继续哪组分析。")
    if action != "new" and not named:
        mentioned = {e["state"]["name"] for e in entries if e["state"]["name"] in question}
        if len(mentioned) > 1:
            raise AgentError("MEMORY_CONTEXT_REQUIRED", "请明确要继续哪组分析。")
        if mentioned:
            name = mentioned.pop()
            if command and body.startswith(name):
                body = body[len(name) :]
                selector = VERSION_SELECTOR.match(body)
                if selector:
                    body = body[selector.end() :]
    matching = [e for e in entries if e["state"]["name"] == name]
    if action != "new" and not matching:
        raise AgentError("MEMORY_CONTEXT_REQUIRED", "本会话没有指定分析，请补充名称或完整条件。")
    if action != "new" and "request" in matching[-1]["state"]["fields"]:
        return None
    version = None
    if action != "new":
        if selector and selector.group("original"):
            version = matching[0]["state"]["version_id"]
        elif selector and selector.group("explicit"):
            version = selector.group("explicit")
        if (
            action == "resume"
            and selector is None
            and re.search(r"版本|原版|初版|最新版|纠正后|turn:", body)
        ):
            return None
    # KEEP clauses mention existing fields without supplying replacement values.
    body = KEEP_CLAUSE.sub("", body)
    # Unrepresented constraints retain the model+verbatim-source path instead of disappearing.
    if re.search(
        r"北京|上海|深圳|已完成|已取消|待支付|待处理|今年|去年|上周|本周|昨天|今天|季度",
        body,
    ):
        return None
    if re.search(r"利润|折扣|优惠|退货|税|汇率|成本|没有completed|不包含|除了|不在", body):
        return None
    known = {
        "Beijing",
        "Shanghai",
        "Shenzhen",
        "completed",
        "pending",
        "cancelled",
        "Electronics",
        "Office",
        "Home",
        "category",
        "city",
        "price_cents",
        "sku",
        "customer_id",
        "order_date",
        "status",
        "turn",
        "YYYY",
        "MM",
    }
    if any(word not in known for word in re.findall(r"[A-Za-z_]+", body)):
        return None
    numeric = re.sub(
        r"turn:\d+|\d{4}年\d{1,2}月(?:\d{1,2}日)?|前\s*\d+\s*(?:个|名|种)|(?:高于|大于|超过|低于|小于|不少于|不低于|至多|不超过)\s*\d+(?:\.\d+)?\s*分?",
        "",
        body,
    )
    if re.search(r"\d", numeric) or ("至少" in body and "至少有一笔" not in body):
        return None
    changes = []

    def put(field, value, operator="eq"):
        changes.append(
            StateChange(
                field=field, operation="set", operator=operator, value=value, evidence=question
            )
        )

    def delete(field):
        changes.append(StateChange(field=field, operation="delete", value=None, evidence=question))

    for field, terms in [
        ("city", "城市|city"),
        ("status", "状态|status"),
        ("category", "类别|商品类别|category"),
        ("price", "价格|标价|价格门槛"),
        ("date", "日期|月份|时间范围"),
        ("limit", "排名数量|数量限制"),
    ]:
        if re.search(r"(?:取消|移除|删除)[^，。；]{0,10}(?:" + terms + ")", body):
            delete(field)
    if re.search(r"(?:所有|全部)(?:订单)?状态|不限制状态", body):
        delete("status")
    for field, values in [
        ("city", ["Beijing", "Shanghai", "Shenzhen"]),
        ("status", ["completed", "pending", "cancelled"]),
        ("category", ["Electronics", "Office", "Home"]),
    ]:
        found = [v for v in values if re.search(r"(?<![A-Za-z])" + v + r"(?![A-Za-z])", body)]
        if found and any(c.field == field and c.operation == "delete" for c in changes):
            return None
        if len(found) > 1:
            return None
        if len(found) == 1:
            put(
                field,
                found[0],
                "ne" if field == "category" and re.search(r"不属于|排除|不包括", body) else "eq",
            )
    date = re.search(r"\d{4}年\d{1,2}月(?:\d{1,2}日)?", body)
    if date:
        change = StateChange(field="date", operation="set", value=date[0], evidence=date[0])
        normalize_date(change, accepted_at)
        changes.append(change)
    else:
        relative = re.search(r"上个月|本月|这个月", body)
        if relative:
            change = StateChange(
                field="date", operation="set", value=relative[0], evidence=relative[0]
            )
            normalize_date(change, accepted_at)
            changes.append(change)
    if re.search(r"价格|标价|门槛", body):
        threshold = re.search(
            r"(高于|大于|超过|低于|小于|不少于|不低于|至多|不超过)\s*(\d+(?:\.\d+)?)\s*分?", body
        )
        if threshold:
            op = {
                "高于": "gt",
                "大于": "gt",
                "超过": "gt",
                "低于": "lt",
                "小于": "lt",
                "不少于": "gte",
                "不低于": "gte",
                "至多": "lte",
                "不超过": "lte",
            }[threshold[1]]
            number = float(threshold[2]) if "." in threshold[2] else int(threshold[2])
            put("price", number, op)
    rank = re.search(r"前\s*(\d+)\s*(?:个|名|种)", body)
    if rank:
        put("limit", int(rank[1]))
    if re.search(r"取消分组|不再分组", body):
        delete("group_by")
    if re.search(r"取消(?:分组和)?排序|不再排序", body):
        delete("sort")
    if not any(c.field == "group_by" and c.operation == "delete" for c in changes):
        grouping = re.search(
            r"(?:按(?:商品|客户|订单)?|分组维度改成(?:当前标价)?)(category|city|price_cents|status|城市|类别|状态|月份)(?:分组|统计|汇总|，|$)",
            body,
        )
        if grouping:
            put("group_by", [OUTPUT_FIELDS[grouping[1]]])
        elif "各city" in body or "各城市" in body:
            put("group_by", ["city"])
    sort_labels = {**OUTPUT_FIELDS, "价格": "price_cents"}
    sorting = re.findall(
        r"(" + "|".join(sorted(sort_labels, key=len, reverse=True)) + r")(升序|降序)", body
    )
    if sorting:
        put(
            "sort",
            [
                ("+" if direction == "升序" else "-") + sort_labels[key]
                for key, direction in sorting
            ],
            "describe",
        )
    outputs = re.search(r"(?:只输出|仅返回|只返回|返回|输出)([^，。；]+)", body)
    if outputs:
        labels = re.split(r"、|和|及|与", outputs[1])
        if any(label.strip() not in OUTPUT_FIELDS for label in labels):
            return None
        put("outputs", [OUTPUT_FIELDS[label.strip()] for label in labels], "describe")
    metric = None
    if re.search(r"平均标价|标价均值", body):
        metric = "商品当前平均标价"
    elif re.search(r"营收|成交金额", body):
        metric = "订单项营收合计"
    elif re.search(r"销量|售出件数|总件数", body):
        metric = "商品销量总件数"
    elif re.search(r"从未下单|从未下过任何订单", body):
        metric = "从未下过任何订单的客户人数"
    elif re.search(r"去重客户", body):
        metric = "去重客户人数"
    elif re.search(r"客户.*(?:人数|多少位|数量|总数)|客户人数|多少[位名]客户", body):
        metric = "客户人数"
    elif re.search(r"订单.*(?:笔数|多少笔|数量|总数)|订单笔数|多少笔订单", body):
        metric = "订单笔数"
    elif re.search(
        r"(?:曾售|售出|已售)商品.*(?:种数|多少种|总种数|总数|数量)|订单中有多少种不同商品", body
    ):
        metric = "曾售商品种数"
    elif re.search(r"商品.*(?:种数|多少种|总种数|总数|数量)|商品种数|多少种商品", body):
        metric = "商品种数"
    elif re.search(r"(?:不同的客户城市|不同的城市)", body):
        metric = "客户城市列表"
    elif re.search(r"当前标价最高的前\s*\d+\s*(?:个|种)(?:Electronics|Office|Home)?商品", body):
        metric = "商品目录列表"
    if metric:
        put("metric", metric, "describe")
    if re.search(r"单位分|营收分|金额分|平均值.*分", body):
        put("unit", "分")
    if action == "new" and not changes:
        return None
    return StateIntent(action=action, name=name, version_id=version, changes=changes)


def execution_question(state, include_source=True, current_question=None):
    labels = {
        "city": "客户城市",
        "status": "订单状态",
        "date": "业务日期范围（含起始，不含结束）",
        "category": "商品类别",
        "price": "商品当前标价（分）",
        "metric": "统计指标",
        "group_by": "分组字段",
        "outputs": "仅输出",
        "unit": "输出单位",
        "limit": "结果数量上限",
        "sort": "排序",
    }
    lines = ["执行以下已经解析的完整用户请求。每项最终条件都必须用于SQL，不需要再次确认。"]
    if include_source:
        lines.append("原始统计要求：" + state["original_question"])
    if include_source and state["modification_questions"]:
        lines.append(
            "依次应用的明确修改（后者覆盖前者）：" + "；".join(state["modification_questions"])
        )
    if current_question:
        lines.append("当前用户请求原文（以本轮明确要求为准）：" + current_question)
    lines.append("当前生效字段（已删除的条件不再生效）：")
    for name, item in state["fields"].items():
        operators = {
            "eq": "为",
            "ne": "不等于",
            "gt": "大于",
            "lt": "小于",
            "gte": "不小于",
            "lte": "不大于",
            "describe": "要求",
            "range": "范围（含起始、不含结束）",
            "in": "属于",
            "not_in": "不属于",
        }
        value = (
            "、".join(map(str, item["value"])) if isinstance(item["value"], list) else item["value"]
        )
        lines.append(f"{labels.get(name, name)}{operators[item['operator']]}：{value}")
    for name in state.get("removed_fields", []):
        lines.append(
            f"已取消{labels.get(name, name)}限制：不要添加该字段的过滤或NULL检查。"
            + ("包含所有订单状态。" if name == "status" else "")
        )
    lines.append("不要重新选择历史专题，不要把专题名称用作数据库筛选值。")
    return "\n".join(lines)
