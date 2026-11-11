"""Paired rules evaluation with the same prompt and model, toggling BM25 evidence."""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from agentserve.config import Settings
from agentserve.knowledge.snapshot import KnowledgeSnapshot
from agentserve.memory.resources import MemoryResources

CASES = [
    ("发票号 c12345 算取消单吗？", "算。发票号 C 前缀不区分大小写。", "retail-cancellation"),
    ("一张发票如何从发票号码区分是否取消？", "发票号以 C 开头，大小写不限。", "retail-cancellation"),
    ("销售额是否包含取消单和零价格记录？", "两者均排除，只计非取消、正数量、正单价。", "retail-sales"),
    ("计算销售额时单条明细的金额怎么算？", "数量乘单价，仅在非取消且数量和单价均为正时计入。", "retail-sales"),
    ("取消金额用负数还是正数显示？", "以正值表示，使用取消单中负数量正单价的明细。", "retail-cancel-amount"),
    ("取消金额如何筛选数量和单价？", "仅含取消单中数量为负、单价为正的明细。", "retail-cancel-amount"),
    ("一张取消单应归入原始购买月份还是取消单自己的开票月？", "取消单自身开票月。", "retail-dates"),
    ("用哪个日期字段统计每月取消金额？", "invoice_date，按取消单开票时间归月。", "retail-dates"),
    ("单价的原始币种和金额存储单位是什么？", "英镑，导入后金额存为百万分之一英镑。", "retail-currency"),
    ("客户 ID 缺失会计入去重客户数吗？", "空 ID 不计入去重客户数。", "retail-customer"),
    ("没有客户 ID 的交易会参与销售额统计吗？", "参与金额统计，空 ID 仅从去重客户数排除。", "retail-customer"),
    ("零价或异常调整明细会删除吗，是否算销售额？", "保留原始行，单列调整，从销售和取消金额排除。", "retail-adjustment"),
    ("取消单关联原始发票的字段名是什么？", None, None),
    ("取消单提交后几天内必须到账？", None, None),
    ("取消金额可以按退款银行拆分吗？", None, None),
    ("取消单的售后负责人是哪一个部门？", None, None),
    ("销售额使用含税还是不含税单价？", None, None),
    ("销售额是按客户时区还是仓库时区折算的？", None, None),
    ("哪些取消单状态代表银行退款已经完成？", None, None),
    ("系统对顾客缺失 ID 使用哪个外部 CRM 补全？", None, None),
    ("跨币种销售额采用哪个具体汇率源？", None, None),
    ("异常调整明细需要哪个审批角色签字？", None, None),
    ("取消单的仓库回收入库日期字段是什么？", None, None),
    ("客户可以在几天内申请取消？", None, None),
]
PROMPT = (
    "用中文回答 AgentServe 的 UCI Online Retail II 业务规则。只回答所问，最多两句。"
    "当规则无法从可靠信息确定时，回答“无法确认”。不要凭常见电商做法猜测。"
)
AS_OF = "2026-11-05"


async def run(args):
    settings = Settings.load(args.config)
    rng = random.Random(20261111)
    rows = []
    blind = []
    async with (
        httpx.AsyncClient(base_url=args.model_url.rstrip("/") + "/", timeout=45, trust_env=False) as llm,
        httpx.AsyncClient(base_url=args.resources_url, timeout=20, trust_env=False) as cpu,
    ):
        resources = MemoryResources(cpu)
        await resources.connect("")
        snapshot = KnowledgeSnapshot(settings.knowledge.snapshot_path, "bm25", resources)
        for index, (question, expected, required) in enumerate(CASES, 1):
            conditions = [False, True]
            rng.shuffle(conditions)
            for position, use_rag in enumerate(conditions, 1):
                started = time.monotonic()
                chunks = []
                if use_rag:
                    result = await snapshot.select(
                        "online_retail_ii", question, started + 20, as_of=AS_OF,
                    )
                    chunks = result["chunks"]
                context = "\n".join(chunk["text"] for chunk in chunks)
                prompt = f"规则参考日期：{AS_OF}\n问题：{question}\n资料：\n{context}"
                response = await llm.post("chat/completions", json={
                    "model": settings.model, "temperature": 0, "max_tokens": 192,
                    "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": prompt}],
                })
                body = response.raise_for_status().json()
                choice = body["choices"][0]
                identifier = f"{index:02d}-{position}"
                retrieved = [chunk["clause_id"] for chunk in chunks]
                entry = {
                    "id": identifier, "rag": use_rag, "question": question, "expected": expected,
                    "answer": choice["message"]["content"], "finish_reason": choice["finish_reason"],
                    "retrieved": retrieved, "required_retrieved": required in retrieved if required else None,
                    "request_id": body["id"], "elapsed_seconds": round(time.monotonic() - started, 5),
                }
                rows.append(entry)
                blind.append({key: entry[key] for key in ("id", "question", "expected", "answer")})
    for path, data in ((args.output, rows), (args.output.with_name("rag-blind.json"), blind)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--resources-url", required=True)
    asyncio.run(run(parser.parse_args()))

