"""Deterministic city discovery and short order questions for the V6 workbench."""

import re
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from difflib import get_close_matches
from zoneinfo import ZoneInfo

from agentserve.agents.answer_policy import order_amount_columns
from agentserve.agents.order_sql import compile_order_sql, sql_literal
from agentserve.domain.models import AgentError
from agentserve.memory.models import ContextPack
from agentserve.memory.state import StateChange, StateIntent, build_state
from agentserve.storage.memory import MemoryRepository

DATABASE = "business_orders_v1"
CITY_SQL = (
    "SELECT c.city, COUNT(o.order_id) > 0 AS has_orders FROM customers c "
    "LEFT JOIN orders o ON o.customer_id = c.customer_id "
    "WHERE c.city IS NOT NULL AND c.city <> '' GROUP BY c.city ORDER BY c.city"
)
ALIASES = {
    "北京": "Beijing", "成都": "Chengdu", "杭州": "Hangzhou",
    "上海": "Shanghai", "深圳": "Shenzhen", "武汉": "Wuhan", "重庆": "Chongqing",
}
CITY = r"(?P<city>[A-Za-z][A-Za-z-]{1,29}|[\u4e00-\u9fff]{2,6}?)"
EXISTS = re.compile(rf"(?:有没有|有)\s*{CITY}\s*(?:的)?订单(?:吗|么)?")
COUNT = re.compile(
    rf"{CITY}\s*(?:的)?(?:订单(?:笔数|数|有多少(?:笔)?)|有(?:多少(?:笔)?|几笔)订单|订购单数)"
    r"(?:是多少|呢)?(?:\s*[，,]?\s*有吗)?"
)
COUNT_PREFIX = re.compile(rf"有(?:多少(?:笔)?|几笔)\s*{CITY}\s*(?:的)?订单")
FOLLOW = re.compile(rf"(?:那|换成|(?:城市)?改[成为])\s*{CITY}\s*(?:呢)?")
DISCOVER = re.compile(r"(?:有|能查|可以查|可查)?哪(?:些|几个)城市(?:的订单|有订单|可以查(?:询)?)?")
CITY_COUNT = re.compile(r"(?:(?:全库|总共|一共)\s*)?有?多少(?:个)?城市(?:有订单(?:详情|明细)?)?")
CAPABILITIES = re.compile(r"(?:^|[？?。！!])\s*(?:有)?哪些城市可以查询(?:[？?。！!,，\s]|$)")
MONTH_RANGE = re.compile(
    r"(?:(?:刚才|上面|上一轮)(?:查的|统计的)?(?:订单)?\s*)?"
    r"(?:(?:是)?(?:从|自)(?:哪|几)(?:个)?月(?:份)?到(?:哪|几)(?:个)?月(?:份)?(?:的)?|月份呢)"
)
COUNT_SELECT = "SELECT COUNT(*) AS order_count"
MONTH = r"(?:(?P<year>[0-9]{4})年)?(?P<month>0?[1-9]|1[0-2])月"
CITY_TOKEN = r"(?:[A-Za-z][A-Za-z-]{1,29}|[\u4e00-\u9fff]{2,6}?)"
CITY_LIST = rf"(?P<cities>{CITY_TOKEN}(?:\s*[和与及、]\s*{CITY_TOKEN})*)"
HOW_MANY = r"有多少(?:笔)?订单"
MONTH_COUNT = [re.compile(pattern) for pattern in (
    rf"{MONTH}\s*{HOW_MANY}\s*{CITY_LIST}",
    rf"{MONTH}\s*{CITY_LIST}\s*{HOW_MANY}",
    rf"{CITY_LIST}\s*{MONTH}\s*{HOW_MANY}",
)]
MONTH_NAMES = "一 二 三 四 五 六 七 八 九 十 十一 十二".split()
MONTH_NUMBER = r"[0-9]{1,2}|十[一二]?|[一二三四五六七八九]"
PERIOD_JOIN = r"\s*(?:到|至|[-—–~～])\s*"
MONTH_SUFFIX = rf"(?:月份?|(?={PERIOD_JOIN}(?:{MONTH_NUMBER})\s*月份?))"
CALENDAR_DATE = re.compile(
    rf"(?<![0-9])(?:(?P<year>[0-9]{{4}}|[0-9]{{2}})\s*年(?:\s*(?P<month>{MONTH_NUMBER})\s*{MONTH_SUFFIX})?"
    rf"|(?P<bare_month>{MONTH_NUMBER})\s*{MONTH_SUFFIX})(?:\s*(?P<day>[0-9]{{1,2}})\s*[日号])?"
)
OTHER_PERIOD = re.compile(r"上个月|下个月|上月|本月|这个月|下月|去年|今年|明年|季度|半年|以来|之前|之后|截至|最近|过去|取消.{0,6}(?:日期|时间|月份)")
SETTLED_EXISTENCE = re.compile(r"(?:有没有(?:成交|交易)(?:吗|么)?|有(?:成交|交易)(?:吗|么))[？?。！!\s]*$")
AMOUNT_RECHECK = re.compile(r"(?:为什么|怎么)(?:月份(?:增加|变多)了?[，,\s]*)?(?:总)?(?:成交额|交易额|营收)(?:反而)?(?:变少|变低|下降|减少)了?")


def calendar_window(match, anchor_year):
    year = int(match["year"]) if match["year"] else anchor_year
    if match["year"] and len(match["year"]) == 2:
        year += anchor_year // 100 * 100
    month = match["month"] or match["bare_month"]
    if month is None:
        start, end = date(year, 1, 1), date(year + 1, 1, 1)
    else:
        month = int(month) if month.isascii() else MONTH_NAMES.index(month) + 1
        start = date(year, month, int(match["day"] or 1))
        end = (start + timedelta(days=1) if match["day"] else
               date(year + (month == 12), month % 12 + 1, 1))
    return [start.isoformat(), end.isoformat()]


def calendar_mentions(text):
    dates = list(CALENDAR_DATE.finditer(text))
    if OTHER_PERIOD.search(text):
        return []
    if len(dates) == 1 and not re.search(r"到|至", text):
        return dates
    if len(dates) == 2 and all((d["month"] or d["bare_month"]) and not d["day"] for d in dates):
        if re.fullmatch(PERIOD_JOIN, text[dates[0].end():dates[1].start()]):
            return dates
    return []


def calendar_bounds(mentions, anchor_year):
    first, last = mentions[0], mentions[-1]
    if len(mentions) == 1:
        return calendar_window(first, anchor_year)
    if last["year"] and not first["year"]:
        anchor_year = int(calendar_window(last, anchor_year)[0][:4])
    start, _ = calendar_window(first, anchor_year)
    last_start, end = calendar_window(last, int(start[:4]))
    if last_start < start:
        raise ValueError("Month range ends before it starts")
    return [start, end]


def calendar_only(text):
    dates = calendar_mentions(text)
    return bool(dates and dates[-1].end() == len(text)
                and re.fullmatch(r"(?:是|改成|改为|换成|改看|只看)?\s*(?:从)?\s*", text[:dates[0].start()]))


def scope_only(question):
    text = question.strip().rstrip("？?。！! ")
    city = FOLLOW.fullmatch(text) or re.fullmatch(CITY, text.removesuffix("呢").strip())
    return bool(calendar_only(text) or (city and (
        city["city"].removesuffix("市") in ALIASES
        or city["city"].casefold() in {name.casefold() for name in ALIASES.values()}
    )))


def calendar_label(period):
    start, end = map(date.fromisoformat, period)
    if start.day != 1 or end.day != 1:
        return start.isoformat() if end - start == timedelta(days=1) else f"{start} 至 {end}（不含结束日）"
    last = end - timedelta(days=1)
    label = f"{start.year}年{start.month}月"
    if (last.year, last.month) != (start.year, start.month):
        last_year = f"{last.year}年" if last.year != start.year else ""
        label += f"至{last_year}{last.month}月"
    if start.month == end.month == 1 and end.year == start.year + 1:
        label = f"{start.year}年"
    return label


def display_calendar_period(answer, period, *, scalar=False):
    start, end = map(date.fromisoformat, period)
    if start.day != 1 or end.day != 1:
        return answer
    label = calendar_label(period)
    first_day = rf"{start.year}年\s*0?{start.month}月\s*0?1[日号]|{period[0]}"
    next_month = rf"(?:{end.year}年\s*)?0?{end.month}月\s*0?1[日号]|{period[1]}"
    formatted = re.sub(rf"(?:{first_day}){PERIOD_JOIN}(?:{next_month})", label, answer)
    if scalar and (end - timedelta(days=1)).replace(day=1) != start and formatted == answer:
        dates = calendar_mentions(answer)
        if dates:
            return answer[:dates[0].start()] + label + answer[dates[-1].end():]
        return f"{label}。{answer}"
    return formatted


def reply(report, answer):
    report.answer = answer
    report.answer_source = "workbench"


def city_label(city):
    chinese = next((name for name, english in ALIASES.items() if english == city), None)
    return f"{chinese}（{city}）" if chinese else city


def short_order_question(question):
    question = question.strip().rstrip("？?。！! ")
    if match := EXISTS.fullmatch(question):
        return match, "exists"
    return COUNT.fullmatch(question) or COUNT_PREFIX.fullmatch(question), "count"


class OrderWorkbench:
    def __init__(self, tool, memory):
        self.tool = tool
        self.memory = memory

    async def cities(self, deadline, journal=None):
        result = await (
            journal.execute_sql(self.tool, DATABASE, CITY_SQL, deadline=deadline)
            if journal else self.tool.execute(DATABASE, CITY_SQL, deadline=deadline)
        )
        if result.status != "ok":
            raise AgentError(result.status, result.error)
        if result.truncated:
            raise AgentError("CITY_CATALOG_LIMIT", "城市列表超过查询限制")
        return result.rows

    @staticmethod
    def city_from_catalog(value, rows, report):
        value = value.strip()
        canonical = ALIASES.get(value.removesuffix("市"), value)
        by_key = {city.casefold(): city for city, _ in rows}
        if city := by_key.get(canonical.casefold()):
            return city
        choices = "、".join(city_label(city) for city in by_key.values())
        similar = get_close_matches(canonical.casefold(), by_key, n=1, cutoff=.82)
        suggestion = f"你是想查{city_label(by_key[similar[0]])}吗？" if similar else ""
        available = f"可查询的城市：{choices}。" if choices else "当前数据还没有可查询的城市。"
        reply(report, f"当前数据里没有「{value}」。{suggestion}{available}")
        return None

    async def resolve_city(self, value, report, deadline, journal):
        return self.city_from_catalog(value, await self.cities(deadline, journal), report)

    async def model_context(self, deadline, journal):
        values = {city for city, _ in await self.cities(deadline, journal)}
        aliases = {city.casefold(): city for city in values}
        aliases.update({name: city for name, city in ALIASES.items() if city in values})
        translations = ", ".join(f"{name} -> {city}" for name, city in ALIASES.items() if city in values)
        context = (
            f"\nAllowed literal values for customers.city: {', '.join(sorted(values))}. "
            f"Translate Chinese labels to stored values: {translations}. "
            "In SQL city predicates, use only the stored value after the arrow, without Chinese or parentheses. "
            "Chinese labels and labels ending in 市 refer to the same city. "
            "For 成交 or 交易, use orders.status = 'completed' by default for both counts and revenue. "
            "Plain 订单 counts still cover all statuses unless explicitly filtered. "
            "有交易吗 asks if completed orders exist. Always filter status = 'completed' "
            "and count orders, not items. "
            "If asked for both existence and amount, return a completed-order count and the "
            "settled revenue in the same query. When joining order_items, use COUNT(DISTINCT "
            "orders.order_id) for order counts. Interpret short years in the current century. "
            "An explicitly named city or month overrides previous turns. "
            "For a complete range of calendar months, name the first and last included month in the answer. "
            "If asked why a past total changed, verify the current range and amount before explaining it. "
            "If the user asks for a total without a time range, use all available dates "
            "unless the question is explicitly a follow-up.\n"
        )
        return context, aliases

    async def describe_order_months(self, report, journal, deadline):
        store = MemoryRepository(journal.store.tasks)
        inputs = await store.inputs(report.task_id, self.memory.settings.recent_turns)
        previous = next((entry for entry in reversed(inputs["entries"]) if entry.get("state")), None)
        if previous is None:
            reply(report, "请先查一个城市的订单，再问这批订单覆盖哪些月份。")
            return
        state = previous["state"]
        fields = state["fields"]
        if (fields.get("metric", {}).get("value") != "订单笔数"
            or fields.keys() - {"metric", "city", "date", "status", "group_by"}
            or fields.get("group_by", {}).get("value", ["city"]) != ["city"]):
            reply(report, "请告诉我你想看哪个城市、哪些订单的月份范围。")
            return

        candidate = compile_order_sql(state)
        last = await journal.store.tasks.get(previous["task_id"])
        previous_sql = last["report"]["sql"] if last and last["report"] else None
        city = fields.get("city", {}).get("value")
        # The earlier V6 model-generated city count used a semantically equivalent IN query.
        legacy_city_count = (
            f"{COUNT_SELECT} FROM orders WHERE customer_id IN "
            f"(SELECT customer_id FROM customers WHERE city = {sql_literal(city)})"
            if fields.keys() == {"city", "metric"} and isinstance(city, str) else None
        )
        legacy_all_statuses = (
            legacy_city_count[:-1] + " AND status IN ('completed', 'pending', 'cancelled'))"
            if legacy_city_count else None
        )
        if not candidate or previous_sql is None or previous_sql not in (candidate, legacy_city_count, legacy_all_statuses):
            reply(report, "这次需要确认查询口径。请告诉我城市和订单条件，我再查月份范围。")
            return

        count_sql = compile_order_sql({
            **state, "fields": {name: value for name, value in fields.items() if name != "group_by"},
        })
        if not count_sql or not count_sql.startswith(COUNT_SELECT + " FROM "):
            reply(report, "请告诉我你想看哪个城市、哪些订单的月份范围。")
            return
        sql = ("SELECT MIN(o.order_date) AS first_order, MAX(o.order_date) AS last_order"
               + count_sql.removeprefix(COUNT_SELECT))
        result = await journal.execute_sql(self.tool, DATABASE, sql, deadline=deadline)
        if result.status != "ok":
            raise AgentError(result.status, result.error)
        if result.truncated:
            raise AgentError("SQL_RESULT_LIMIT", "日期结果超过查询限制")
        report.sql, report.query_result = sql, result
        label = ("和".join(map(city_label, city)) if isinstance(city, list)
                 else city_label(city) if city else "上一轮查询范围内")
        date_filter = fields.get("date", {})
        if date_filter.get("operator") == "range":
            start, end = date_filter["value"]
            scope = f"上一轮筛选范围为 {start} 至 {end}（不含结束日）。"
        elif date_filter:
            scope = "上一轮设有日期筛选。"
        else:
            scope = "上一轮未限定日期。"
        first, last = result.rows[0]
        if first is None:
            reply(report, f"{scope}{label}没有符合条件的订单。")
        else:
            reply(report, f"{scope}{label}的订单实际覆盖 {first[:7]} 至 {last[:7]}。"
                  f"首单 {first}，末单 {last}。")

    async def prepare(self, task, report, journal, deadline):
        if task.database_id != DATABASE:
            return task, None
        window = task.analysis or task.metric
        if window and window.city:
            city = await self.resolve_city(window.city, report, deadline, journal)
            if city and city != window.city:
                task = task.model_copy(update={task.mode: window.model_copy(update={"city": city})})
            return task, None
        if not task.session_id or not self.memory or self.memory.settings.policy not in {
            "state", "state_on_demand",
        }:
            return task, None
        question = task.question.strip().rstrip("？?。！! ")
        date_edit = calendar_only(question)
        if date_edit or AMOUNT_RECHECK.fullmatch(question):
            return task, await self.prepare_scope_update(task, report, journal, date_edit)
        if MONTH_RANGE.fullmatch(question):
            await self.describe_order_months(report, journal, deadline)
            return task, None
        capabilities = CAPABILITIES.search(question)
        city_count = CITY_COUNT.fullmatch(question)
        if DISCOVER.fullmatch(question) or capabilities or city_count:
            rows = await self.cities(deadline, journal)
            names = [city_label(city) for city, has_orders in rows
                     if has_orders or (not city_count and ("订单" not in question or capabilities))]
            if city_count:
                answer = f"当前订单库共有 {len(names)} 个城市有订单。"
                if names:
                    answer += f"分别是{'、'.join(names)}。"
            else:
                answer = f"可以查这些城市：{'、'.join(names)}。" if names else "当前范围内没有可查询的城市。"
            if "修改" in question:
                answer += "可以查订单有无、笔数和明细，也可以调整查询条件。订单数据是只读的。"
            reply(report, answer)
            report.sql = CITY_SQL
            return task, None

        match, kind = short_order_question(question)
        monthly = next((item for pattern in MONTH_COUNT if (item := pattern.fullmatch(question))), None)
        bare_city = question.removesuffix("市") in ALIASES or question.casefold() in {
            city.casefold() for city in ALIASES.values()
        }
        follow = None if match or monthly else FOLLOW.fullmatch(question) or (re.fullmatch(CITY, question) if bare_city else None)
        if match is None and follow is None and monthly is None:
            return task, None
        store = MemoryRepository(journal.store.tasks)
        saved = await store.context(report.task_id)
        inputs = await store.inputs(report.task_id, self.memory.settings.recent_turns)
        entries = [entry for entry in inputs["entries"] if entry.get("state")]
        prior = entries[-1]["state"] if entries else None
        refine = False
        if follow:
            last_state_turn = entries[-1]["turn_no"] if entries else 0
            for turn in sorted(inputs["recent"], key=lambda item: item["turn_no"], reverse=True):
                if turn["status"] == "COMPLETED" and turn["turn_no"] > last_state_turn:
                    short, previous_kind = short_order_question(turn["question"])
                    if short:
                        kind = previous_kind
                        break
            else:
                refine = True
        if refine:
            fields = prior["fields"] if prior else {}
            if (fields.get("metric", {}).get("value") != "订单笔数"
                or fields.keys() - {"city", "date", "status", "metric", "group_by"}
                or fields.get("group_by", {}).get("value", ["city"]) != ["city"]
                or not prior.get("complete_order_intent")):
                return task, None
            kind = prior.get("workbench_answer", "count")
        if monthly:
            rows = await self.cities(deadline, journal)
            cities = []
            for name in re.split(r"\s*[和与及、]\s*", monthly["cities"]):
                city = self.city_from_catalog(name, rows, report)
                if city is None:
                    return task, None
                if city not in cities:
                    cities.append(city)
        else:
            city = await self.resolve_city((match or follow)["city"], report, deadline, journal)
            if city is None:
                return task, None
            cities = [city]
        if saved:
            return task, ContextPack(**saved)

        changes = [StateChange(
            field="city", operation="set", operator="in" if len(cities) > 1 else "eq",
            value=cities if len(cities) > 1 else cities[0],
            evidence=monthly["cities"] if monthly else (match or follow)["city"],
        )]
        if not refine:
            changes.append(StateChange(
                field="metric", operation="set", operator="describe", value="订单笔数",
                evidence=task.question,
            ))
        elif "group_by" in prior["fields"]:
            changes.append(StateChange(field="group_by", operation="delete", evidence=task.question))
        if monthly:
            year = int(monthly["year"] or datetime.fromtimestamp(
                inputs["accepted_at"], ZoneInfo("Asia/Shanghai"),
            ).year)
            month = int(monthly["month"])
            if year < 1 or (year == 9999 and month == 12):
                reply(report, "请提供有效的查询年份。")
                return task, None
            start = f"{year:04d}-{month:02d}-01"
            end = f"{year + (month == 12):04d}-{month % 12 + 1:02d}-01"
            changes.append(StateChange(field="date", operation="set", operator="range",
                                       value=[start, end], evidence=task.question))
            if len(cities) > 1:
                changes.append(StateChange(field="group_by", operation="set", value=["city"],
                                           evidence=monthly["cities"]))
        intent = StateIntent(action="refine" if refine else "new", changes=changes)
        state, target = build_state(
            intent, entries, prior["name"] if prior else None, task.question,
            inputs["cutoff_turn"] + 1, inputs["accepted_at"], True,
        )
        state["workbench_answer"] = kind
        pack = ContextPack(
            "", [target["entry_id"]] if target else [], inputs["cutoff_turn"], inputs["revision"],
            self.memory.policy.version, 0, {
                "state": state, "action": intent.action, "can_execute": True,
                "resolution": "deterministic", "target_id": target["entry_id"] if target else None,
            },
        )
        await store.save_context(report.task_id, journal.version, asdict(pack))
        return task, pack

    async def prepare_scope_update(self, task, report, journal, date_edit):
        store = MemoryRepository(journal.store.tasks)
        saved = await store.context(report.task_id)
        if saved and (saved.get("plan") or {}).get("resolution") == "scope_update":
            return ContextPack(**saved)
        inputs = await store.inputs(report.task_id, self.memory.settings.recent_turns)
        entries = [entry for entry in inputs["entries"] if entry.get("state")]
        if not entries:
            return None
        prior = entries[-1]["state"]
        fields = prior["fields"].keys()
        has_scope = bool(fields & {"city", "metric"}) if date_edit else {"city", "date"} <= fields
        if not has_scope:
            return None
        changes = [StateChange(field="date", operation="set", value=task.question, evidence=task.question)] if date_edit else []
        intent = StateIntent(action="refine", changes=changes)
        state, target = build_state(intent, entries, prior["name"], task.question,
                                    inputs["cutoff_turn"] + 1, inputs["accepted_at"], False)
        pack = ContextPack("", [target["entry_id"]], inputs["cutoff_turn"], inputs["revision"],
                           self.memory.policy.version, 0, {
                               "state": state, "action": "refine", "can_execute": True,
                               "resolution": "scope_update", "target_id": target["entry_id"],
                           })
        await store.save_context(report.task_id, journal.version, asdict(pack))
        return pack

    def ground_period(self, task, pack):
        """Bind a single calendar period to user evidence, including in old saved conversations."""
        plan = pack.plan or {}
        state = plan.get("state")
        if task.database_id != DATABASE or not state:
            return pack
        fields = state["fields"]
        question = task.question.strip().rstrip("？?。！! ")
        texts = [state["original_question"], *state["modification_questions"]]
        if texts[-1] != task.question:
            texts.append(task.question)
        if "date" in state.get("removed_fields", []) and not calendar_mentions(question):
            return pack
        anchor_year = datetime.fromtimestamp(state["accepted_at"], ZoneInfo("Asia/Shanghai")).year
        window = None
        evidence = None
        for text in texts:
            mentions = calendar_mentions(text)
            if mentions:
                mention = mentions[0]
                try:
                    window = calendar_bounds(mentions, anchor_year)
                except (ValueError, OverflowError):
                    if text == task.question:
                        raise AgentError("MEMORY_CONTEXT_REQUIRED", "日期范围无效，请明确开始和结束月份。") from None
                    window = None
                    continue
                if mentions[-1]["year"]:
                    anchor_year = (date.fromisoformat(window[1]) - timedelta(days=1)).year
                elif mention["year"]:
                    anchor_year = int(window[0][:4])
                evidence = text[mention.start():mentions[-1].end()]
            elif CALENDAR_DATE.search(text) or OTHER_PERIOD.search(text):
                window = None
        if window is None:
            return pack
        previous = fields.get("date", {})
        binding = {"operator": "range", "value": window, "source_task_id": previous.get("source_task_id")}
        state = {**state, "fields": {**fields, "date": binding}, "workbench_period": window}
        if previous.get("value") != window and calendar_mentions(question):
            binding["source_task_id"] = None
            state["changed_fields"] = [change for change in state["changed_fields"] if change["field"] != "date"]
            state["changed_fields"].append(StateChange(
                field="date", operation="set", operator="range", value=window, evidence=evidence,
            ).model_dump())
        if scope_only(question) and fields.keys() <= {"city", "date", "status", "metric"}:
            last_request = next((text for text in reversed(texts[:-1]) if not scope_only(text)), "")
            if SETTLED_EXISTENCE.search(last_request) and fields.get("status", {}).get("value", "completed") == "completed":
                state["fields"].update({
                    "metric": {"operator": "describe", "value": "订单笔数", "source_task_id": None},
                    "status": {"operator": "eq", "value": "completed", "source_task_id": None},
                })
                state["complete_order_intent"] = True
                plan = {**plan, "render_calendar_count": True}
        return replace(pack, plan={**plan, "state": state, "can_execute": True})

    @staticmethod
    def finish(task, report, pack):
        result = report.query_result
        state = (pack.plan or {}).get("state") if pack else None
        period = state.get("workbench_period") if state else None
        scalar = bool(result and len(result.rows) == len(result.columns) == 1 and not result.truncated)
        if period and report.answer_source == "model":
            report.answer = display_calendar_period(report.answer, period, scalar=scalar)
        if (period and scalar and report.answer_source == "tool_values"
            and type(result.rows[0][0]) is int
            and result.columns[0].lower() in {"revenue", "total_revenue", "revenue_cents"}):
            city = state["fields"].get("city", {}).get("value")
            scope = city_label(city) if isinstance(city, str) else "所查范围"
            reply(report, f"{scope}{calendar_label(period)}的成交额为 {Decimal(result.rows[0][0]) / 100:,.2f} 元。")
            return
        if (pack and result and result.rows and not result.truncated
            and all(cell is None for row in result.rows for cell in row)
            and order_amount_columns(report.sql, result)):
            reply(report, "所查范围内没有可汇总的金额。")
            return
        if task.mode == "analysis" and result and not result.rows:
            city = task.analysis.city
            label = city_label(city) if city else "所选范围"
            reply(report, f"{label}在所选时段没有已完成订单。可调整日期范围后再查。")
        elif state and not task.answer_mode:
            fields = state["fields"]
            kind = state.get("workbench_answer")
            if (result is None or report.sql != compile_order_sql(state) or result.truncated
                or fields.get("metric", {}).get("value") != "订单笔数"):
                return
            city_filter = fields.get("city", {})
            if fields.keys() - {"city", "date", "metric", "group_by"} or city_filter.get("operator") not in {"eq", "in"}:
                return
            date_filter = fields.get("date")
            prefix = ""
            if date_filter:
                if date_filter["operator"] != "range":
                    return
                start, end = date_filter["value"]
                month = int(start[5:7])
                next_month = f"{int(start[:4]) + (month == 12):04d}-{month % 12 + 1:02d}-01"
                prefix = (f"{start[:4]}年{month}月" if start[8:] == "01" and end == next_month
                          else f"{start} 至 {end}（不含结束日）期间")
            city_value = city_filter["value"]
            if city_filter["operator"] == "in" and result.columns == ["city", "order_count"]:
                values = dict(result.rows)
                counts = [(city_label(city), values.get(city, 0)) for city in city_value]
                if all(count == 0 for _, count in counts):
                    names = "和".join(name for name, _ in counts)
                    reply(report, f"{prefix}{names}的订单笔数均为 0。")
                else:
                    summary = "，".join(f"{name} {count} 笔" for name, count in counts)
                    reply(report, f"{prefix}订单笔数：{summary}。")
                return
            if city_filter["operator"] != "eq" or result.columns != ["order_count"] or not result.rows:
                return
            city = city_label(city_value)
            count = result.rows[0][0]
            if kind == "exists":
                reply(report, f"有，{city}有符合条件的订单。" if count else f"{city}没有符合条件的订单。")
            else:
                reply(report, f"{prefix}{city}有 {count} 笔符合条件的订单。")
