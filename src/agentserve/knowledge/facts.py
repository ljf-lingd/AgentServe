"""Resolve requested properties only from their approved, retrieved witnesses."""

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field


class Property(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    subject: str
    relation: str
    witnesses: dict[str, list[str]]


class PropertyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    properties: dict[str, Property] = Field(min_length=1)


def install_facts(root, index, chunks, purpose):
    if index == "fixture":
        if purpose != "test":
            raise ValueError("Fixture properties require a test snapshot")
        return fixture_facts(chunks)

    path = (root / index).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("Property index must be a JSON file inside the manifest directory")
    properties = PropertyFile.model_validate_json(path.read_text(encoding="utf-8"))
    approved = {chunk["chunk_id"]: chunk for chunk in chunks if chunk["approval_ref"]}
    schema = {}
    for key, field in properties.properties.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or not field.witnesses:
            raise ValueError(f"Invalid property {key}")
        for pattern in (field.subject, field.relation):
            re.compile(pattern)
            if re.search(pattern, ""):
                raise ValueError(f"Empty match for property {key}")
        for source, witnesses in field.witnesses.items():
            chunk = approved.get(source)
            if (
                not chunk
                or not witnesses
                or any(not text or text not in chunk["text"] for text in witnesses)
            ):
                raise ValueError(f"Unapproved witness for property {key} in {source}")
            chunk.setdefault("facts", {})[key] = witnesses
        schema[key] = field.model_dump(exclude={"witnesses"})
    if any(not chunk.get("facts") for chunk in approved.values()):
        raise ValueError("An approved clause is missing property annotations")
    return schema


FIXTURE_FACT = re.compile(
    r"演练中(?:的)?(?P<topic>[^。\n]+?)(?:为 (?P<value>[^。\n]+)|由(?P<registrar>[^。\n]+)登记)。"
)
FIXTURE_RELATIONS = {
    "value": r"多少|几(?:天|单|次|分|小时|个)|什么|哪个|多长|多高|多大|为何|多频繁",
    "registrar": r"谁|哪个(?:部门|单位|项目|岗位)|何人",
}


def fixture_facts(chunks):
    schema = {}
    ids = {}
    for chunk in chunks:
        if not chunk["approval_ref"] or not (fact := FIXTURE_FACT.fullmatch(chunk["text"])):
            continue
        topic = fact["topic"]
        for kind, relation in FIXTURE_RELATIONS.items():
            key = ids.setdefault((topic, kind), f"fixture_{len(ids) + 1}")
            schema[key] = {
                "description": f"{topic}的{'具体值' if kind == 'value' else '登记者'}",
                "subject": re.escape(topic),
                "relation": relation,
            }
            if fact[kind] is not None:
                chunk["facts"] = {key: [fact[0]]}
    return schema


REQUEST_BOUNDARY = re.compile(r"[?？;；\n。]+|(?:，|,)?(?:并且|同时还要|另外还要|以及还要)")
PARALLEL = re.compile(
    r"(.+?)(?:分别|各自)((?:是|为|用)?(?:哪(?:个|一|两|几|种|组|些)?(?:列|字段|项|个|种|组|张表)?|什么|多少))$"
)
COORDINATION = re.compile(r"、|和|以及|与")
UNSUPPORTED = re.compile(
    r"时区|税率|税费|运费|运输费|优惠券|折扣券|新人折扣|退款|退货|支付方式|银行卡|微信支付|"
    r"手机号|电子邮件|邮箱|汇率|自动取消|预警阈值|滞销|海外城市|默认排序(?:列|字段)|"
    r"(?:具体|实际)(?:删除|修改|写入)(?:数|行)|(?:从第几行|截断阈值|多少行后)|"
    r"(?:忽略|跳过|绕过).{0,8}(?:约束|规则|指令)|\b(?:DELETE|DROP|UPDATE|INSERT|ALTER)\b|"
    r"(?:直接查出|直接告诉我当前|实际(?:营收|销量|数量|金额|结果|数值)|实时(?:查|数|结果))|"
    r"(?:新\s*SKU|SKU\s*的?).{0,10}(?:生成|编号前缀|编码规范)",
    re.IGNORECASE,
)
UNREGISTERED_RELATION = re.compile(
    r"接口地址|服务名|维护接口|索引(?:占|大小|空间)|自增(?:序号|起始|步长)|分页(?:参数|令牌)|"
    r"(?:生成|分配|创建).{0,8}(?:序列号|序号)|(?:哪(?:些|几种|类)|什么).{0,8}渠道|"
    r"(?:字段|外键|列|键).{0,6}(?:可空性|可以为空|能否为空|是否可空|允许\s*NULL)|"
    r"(?:我(?:现在)?宣布|以我(?:说的|提供的)为准|替换(?:批准)?文档|以这个假设|把规则改成)",
    re.IGNORECASE,
)
UNREGISTERED_VALUE = re.compile(
    r"币种|货币(?:代码|种类)|返现|代金券|促销码|积分抵扣|净重|毛重|重量|"
    r"(?:分数|分值|评分).{0,10}(?:下限|上限|阈值|门槛|区间|范围)|"
    r"(?:默认|通常|自动).{0,24}(?:哪个|哪一|什么).{0,12}(?:字段|列|键).{0,6}排序|"
    r"(?:按哪个|按什么|用哪).{0,8}(?:字段|列|键).{0,6}(?:默认|自动)排序"
)
CASE_REQUEST = re.compile(r"大小写|大写.{0,10}小写|小写.{0,10}大写|case.sensitiv", re.IGNORECASE)
CASE_SUBJECT = re.compile(
    r"城市|分类|类目|姓名|名字|客户名|商品编码|\bsku\b|(?:哪[些几]|什么)(?:个|种)?字段",
    re.IGNORECASE,
)
COLUMN_REQUEST = re.compile(
    r"哪(?:个|一|几|两)?(?:数据库)?(?:字段|列)|(?:字段|列)(?:是|为|叫)?哪(?:个|一)|列名|字段名"
)
COLUMN_SUBJECTS = {
    "city_column": r"城市|city\b",
    "order_status": r"状态|\bpending\b|\bcompleted\b|\bcancelled\b",
    "customer_key": r"客户(?:主键|标识|ID|编号)|买家(?:ID|标识)|同名|customer_id",
    "order_key": r"订单(?:主键|标识|ID|号)|(?:订单|单据).{0,8}(?:唯一|标识|识别)|order_id",
    "item_links": r"(?:明细|order_items).{0,30}(?:关联|连到|连接|联接|外键)",
    "item_key": r"(?:明细|order_items).{0,25}(?:唯一|标识|主键|哪组|组合)|联合主键",
    "date_column": r"(?:业务|订单)(?:日期|时间)|order_date|(?:^|这(?:个|一)?|该|此)日期",
    "sku_column": r"\bsku\b|商品编码",
    "category_column": r"类目|分类|category\b",
    "transaction_price": r"历史(?:成交)?(?:单)?价|成交(?:单)?价|历史价|unit_price_cents",
    "catalog_price": r"标价|现价|目录(?:当前|现在)?(?:价|的价格)|products\.price_cents",
    "quantity_definition": r"销量|件数|quantity\b",
}
OPEN = re.compile(
    r"吗|么|如何|怎么|怎样|什么|多少|哪些|哪[个些一两几种组张列]|几点|几(?:天|次|单|分)|"
    r"是否|会不会|能否|要不要|需不需要|还是|请说明|请解释|为何|多长|多高"
)
QUESTION_TAIL = re.compile(
    r"^(?:两点|两项|每项|这两点|以上|这些)?(?:都|均)?(?:要|请)?(?:回答|说明)(?:一下)?$"
)
ANAPHORA = re.compile(
    r"^(?:(?:它|该|此|其|这个|那|同一|两列|两个字段|结果|又)|(?:连|联|接|关联|对应)(?:到|上|回))"
)
DATA_LOOKUP = re.compile(
    r"(?:查(?:询|找)?(?:一下|出|出库里)?|统计|数一数|算出|列出|告诉我|给出)"
    r".{0,22}(?:现在|当前|目前|库中|库里|这[个一]月|上[个一]月|今天|本周|[0-9]{1,2}月)"
    r".{0,22}(?:多少|几笔|几条|几个|前[0-9一二三十]+|最新[一1]|金额|收入|总数|结果)",
    re.IGNORECASE,
)
DEFINITION_CUE = re.compile(r"字段|列名|口径|定义|公式|文档|规范|存储|存放|默认规则")
LIVE_RECORD = re.compile(r"(?:最新|最近|最早)(?:一|1|几|前).{0,18}(?:记录|明细|订单|单子)")
WRITE_REQUEST = re.compile(
    r"^(?:请|帮我|立即|现在|直接|务必)*(?:把|将).{0,24}(?:订单|目录|客户|商品|记录)"
    r".{0,24}(?:改为|标记为|更新为|删除|写入|移除)"
)
FOLLOW_UP = re.compile(
    r"^(?:按|靠|用|根据)(?:什么|哪[个一])(?:辨[别识]|区分|分辨|识别|判断)(?:的|呢)?$"
)

# Full fragments with a stable interpretation can bind directly. Any additional
# modifier leaves the fragment to the semantic parser and its candidate check.
EXACT = {
    "month_basis": r"按月(?:统计|汇总)的?月份(?:怎么|如何)(?:取得|确定|取|归属|划分)",
    "date_format": (
        r"(?:原始)?(?:订单)?(?:业务)?日期(?:怎么存储|存成什么样的字符串|"
        r"(?:在库里|在数据库里)(?:是|采用)?(?:何种|什么)格式)"
    ),
    "city_column": r"(?:按城市|城市)(?:汇总|分组)(?:该|应该)?(?:用|取)哪(?:个|一)(?:字段|列)",
    "case_comparison": r"(?:过滤|比较|筛选)城市(?:字段的)?值时(?:是否|会不会)区分大小写",
    "grouping_policy": (
        r"(?=.*(?:没(?:有)?|未|不)(?:明确)?(?:提出|要求|指定|说明))"
        r"(?=.*(?:按月|每月|月份|城市|状态))"
        r"[^？?；;。]{0,75}(?:分项|分组|拆分|分开)(?:吗|么|呢)"
    ),
}


def requests(question):
    pieces = []
    for fragment in REQUEST_BOUNDARY.split(question):
        fragment = fragment.strip(" \t，,！!")
        if not fragment or QUESTION_TAIL.fullmatch(fragment):
            continue
        if pieces and FOLLOW_UP.fullmatch(fragment):
            pieces[-1] += "，" + fragment
            continue
        if parallel := PARALLEL.fullmatch(fragment):
            names = COORDINATION.split(parallel[1])
            # Do not split sentence-internal conjunctions in a comparison.
            if len(names) > 1 and all(name and not OPEN.search(name) for name in names):
                suffix = re.sub(r"哪(?:两|几)(?=列|字段|组|种|张表)", "哪一", parallel[2])
                pieces.extend(name + suffix for name in names)
                continue
        pieces.append(fragment)
    return pieces


def possible_fields(fragment, schema, *, fixture, preceding=""):
    if unsupported_request(fragment):
        return {}
    candidates = {
        key: field["description"]
        for key, field in schema.items()
        if re.search(field["subject"], fragment, re.IGNORECASE)
        and re.search(field["relation"], fragment, re.IGNORECASE)
    }
    if not fixture and not candidates and ANAPHORA.match(fragment):
        candidates = {
            key: field["description"]
            for key, field in schema.items()
            if re.search(field["subject"], preceding, re.IGNORECASE)
            and re.search(field["relation"], fragment, re.IGNORECASE)
        }
    if fixture and re.search(r"由谁|谁(?:负责)?登记|何人|哪个(?:部门|单位|项目|岗位)", fragment):
        return {key: label for key, label in candidates.items() if label.endswith("的登记者")}
    return candidates


def unsupported_request(fragment):
    return bool(
        UNSUPPORTED.search(fragment)
        or UNREGISTERED_RELATION.search(fragment)
        or UNREGISTERED_VALUE.search(fragment)
        or WRITE_REQUEST.search(fragment)
        or (
            (DATA_LOOKUP.search(fragment) or LIVE_RECORD.search(fragment))
            and not DEFINITION_CUE.search(fragment)
        )
    )


def applicable_properties(fragment, preceding, retrieved):
    """Keep explicit attribute requests inside the registered subject and relation."""
    if (
        CASE_REQUEST.search(fragment)
        and not CASE_SUBJECT.search(fragment)
        and (not ANAPHORA.match(fragment) or not CASE_SUBJECT.search(preceding))
    ):
        return []
    if COLUMN_REQUEST.search(fragment):
        text = f"{preceding}，{fragment}" if ANAPHORA.match(fragment) else fragment
        matched = {
            key
            for key, subject in COLUMN_SUBJECTS.items()
            if re.search(subject, text, re.IGNORECASE)
        }
        if "日期" in text and "格式" in fragment:
            matched.add("date_format")
        if {"transaction_price", "catalog_price"} & matched and "单位" in fragment:
            matched.add("price_unit")
        if CASE_REQUEST.search(fragment):
            matched.add("case_comparison")
        return [key for key in retrieved if key in matched]
    return list(retrieved)


def fixed_field(fragment, candidates, *, fixture):
    compact = re.sub(r"\s+", "", fragment)
    if fixture and len(candidates) == 1:
        ((key, label),) = candidates.items()
        topic = re.escape(label.rsplit("的", 1)[0])
        suffix = (
            r"由(?:谁|哪个(?:部门|项目))(?:负责)?登记"
            if label.endswith("的登记者")
            else r"(?:是|为|有)?(?:多少(?:天|单|次|分|小时|%|个)?|什么|哪个|多长(?:时间)?|几(?:天|单|次|分|小时))"
        )
        if re.fullmatch(r"(?:请问)?演练中(?:的)?" + topic + suffix, compact):
            return key
    for key, pattern in EXACT.items():
        if key in candidates and re.fullmatch(pattern, compact):
            return key
    return None


def fact_evidence(chunks):
    return {
        text: f"e{index}"
        for index, text in enumerate(
            dict.fromkeys(
                text
                for chunk in chunks
                for witnesses in chunk.get("facts", {}).values()
                for text in witnesses
            ),
            1,
        )
    }


def resolve(fields, chunks):
    """Preserve every active conflicting source and require support per property."""
    by_field = defaultdict(list)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    conflicts = defaultdict(set)
    for chunk in chunks:
        for key, witnesses in chunk.get("facts", {}).items():
            by_field[key].extend(
                {"chunk_id": chunk["chunk_id"], "witness": text} for text in witnesses
            )
        if group := chunk["conflict_group"]:
            conflicts[group].add(chunk["chunk_id"])

    cited = {}
    checks = []
    disputed = False
    for fragment, keys in fields:
        supports = []
        if keys and all(by_field[key] for key in keys):
            supports = [support for key in keys for support in by_field[key]]
            ids = {support["chunk_id"] for support in supports}
            involved = [group for group in conflicts.values() if group & ids and len(group) > 1]
            if any(not group <= ids for group in involved):
                supports = []
            else:
                disputed |= bool(involved)
                cited.update(
                    (support["chunk_id"], by_id[support["chunk_id"]]) for support in supports
                )
        checks.append({"question": fragment, "properties": keys, "supports": supports})
    if not checks or not all(check["supports"] for check in checks):
        return "insufficient_evidence", {}, checks
    return ("conflicting_evidence" if disputed else "answered"), cited, checks
