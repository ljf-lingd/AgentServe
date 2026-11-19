# V5 工程验证与故障复盘

V5 使用独立应用进程、真实 Qwen2.5-7B-Instruct、只读查询库和单独的任务账本。故障注入走正常 Task API。模型响应代理和步骤断点只用于实验。 [环境、冻结负载和命令](v5-reproduction.md#p0-mixed-business-workload-and-recovery) · [故障入口](../scripts/experiments/run_v5_faults.py)

| 故障或约束 | 注入位置 | 观察结果 |
| --- | --- | --- |
| 模型响应超时 | 代理只扣留一次真实模型响应 | 应用在 30 秒记录 `MODEL_TIMEOUT`，首个请求用量记未知。一次重试在原截止时间内完成，结果与健康对照一致。迟到响应未提交。 |
| 应用进程退出 | 业务 SQL 已执行，步骤尚未持久化 | 重启后沿用任务 ID、开始时间及绝对截止时间。目标只读查询总共执行两次，两条已提交的模型输出被复用，最终结果符合金标。 |
| 重复提交及取消 | 同时提交同一幂等键，分别取消排队和运行中任务 | 重复提交得到同一个任务 ID。排队任务没有执行尝试。释放迟到响应并重启后，两个任务仍为取消状态。 |
| SQL 和知识权限 | 提交写 SQL，并用真实知识问答核查工具与引用 | 写 SQL 返回 `POLICY_REJECTED`。知识请求使用结构化回答，暴露工具数和 SQL 步骤数均为 0。引用可追溯到批准原文，查询库指纹一致且 SQLite `quick_check` 通过。 |

运行中取消用例验证的是应用终态和迟到响应处理。模型在代理扣留响应前已完成生成，GPU 中止未获得验证。

恢复实验的首轮计数混入了城市元数据查询。定点复验改为定位业务 SQL 的步骤号，分别核对执行次数、已提交模型步骤和恢复后的结果。独立运行保留每次尝试的状态账本，以便在不改变业务判定的情况下复核注入点。 [故障用例实现](../scripts/experiments/v5_fault_cases.py)

## 正常执行但业务结果错误

UCI 查询曾用 `COUNT(DISTINCT invoice)` 回答去重客户数，得到 40。按交易明细独立计算的客户数为 29。修复只在明确的零售去重客户意图和受限 SQL 形状匹配时，将聚合列绑定到 `customer_id`。任务步骤保留模型原 SQL 和执行 SQL。

归档的 384 条 SQL 执行回放从 376/384 提升至 384/384。在含 1,067,371 行明细的 UCI 库上，72 个留出国家月份组合从 24/72 提升至 72/72，72 条发票数对照均通过，库指纹一致。这部分为 CPU SQL 回放，新增模型请求数为 0。原始推理实验仍按原分数保留。

先按 [UCI 数据准备](getting-started.md#4-可选-uci-零售数据) 生成 SQLite。在仓库根目录执行公开留出回放：

```bash
PYTHONPATH=src python scripts/experiments/retail_customer_count_replay.py \
  data/databases/online_retail_ii/online_retail_ii.sqlite retail-customer-replay.json
```

## 并发定位

固定 vLLM 容量为 4，提高应用并发从 4 到 8。C8 正式阶段共有 956 次提交，168 次真实模型请求均能归因。吞吐接近平台期，应用队列和模型槽等待增加，vLLM 平均排队低于 4 毫秒。该实验定位了等待增量，CPU 与 SQLite 各自的贡献留待独立干预测量。 [并发配置、分母和命令](v5-reproduction.md#expanded-business-workload)
