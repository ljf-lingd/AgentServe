# AgentServe

AgentServe 围绕一个可运行的 SQL Agent，研究**多轮分析如何记住正确的条件、业务问答如何引用正确的规则、模型推理如何降低任务总耗时**。三个问题分别做了基线对照，选出的方案接入同一套 Task API 和工作台。

AgentServe is a research-oriented SQL agent for studying session memory, evidence-grounded retrieval, and end-to-end latency with speculative decoding.

[研究设计与源码](docs/research.md) · [评测与复现](docs/evaluation.md) · [安装与启动](docs/getting-started.md) · [接入自己的模型和数据](docs/customization.md)

## 三条研究主线

| 研究问题 | 对照与实现 | 代表结果 |
| --- | --- | --- |
| **同会话长程记忆** | 对照近期两轮窗口、语义压缩与混合检索、带来源的查询事件。最终将查询条件存成可恢复的版本，区分保留、修改和删除。 | 24 段正式会话每组 672 轮，记忆相关流程 **48/120 → 120/120**。44 轮会话整段通过 **0/12 → 12/12**。 |
| **业务规则 RAG** | 比较固定证据、BM25、稠密检索和 RRF 融合。按数据集和生效日期筛条款，再核对问题所需属性与引用原文。 | 正式集固定证据 → BM25，有据回答 **8/24 → 23/24**。两组缺证据题均 **24/24** 正确弃答。 |
| **全阶段推测解码** | 用后缀自动机从当前请求和同会话已提交的输出中找草稿，由目标模型验证。覆盖记忆规划、SQL 生成与修复、规则问答、最终回答。 | RTX 4090、Qwen2.5-7B、每组 40 个混合任务。仅回答阶段启用 → 全生成阶段启用，服务端 **JCT P95 从 17.286 降至 10.207 秒（约 41%）**。 |

JCT（Job Completion Time）从任务创建计到终态提交，包含排队、模型调用和工具执行。 [研究设计与判分口径](docs/research.md)

## 项目演进

| 阶段 | 主要进展 |
| --- | --- |
| V1 | 建立只读 SQL 问答基线，支持任务持久化、取消和恢复。 |
| V2 | 对比近期窗口、混合检索和查询事件，将条件存成可追溯、可恢复的会话状态。 |
| V3–V4 | 比较业务规则检索方案。V3.2 将版本化口径用于指标查询，V4 用一条参数化 SQL 在同一读快照中计算月度简报的两项指标。 |
| V5–V6 | 从回答阶段扩展到各模型生成阶段的推测解码，用工作台混合任务比较端到端 JCT。 |
| V7–V7.1 | 接入 UCI 零售数据进行跨数据集验证，修复全量库中已复现的日期查询超时。 |

[更新日志](CHANGELOG.md)

## 工作台与应用接口

| 能力 | 使用方式 |
| --- | --- |
| 多轮查数 | 生成受控 SQLite 查询，返回结果与证据。追问在同一会话内继承适用条件。 |
| 业务规则问答 | 检索按数据集和有效日期登记的资料，回答可附出处。证据不足时可以弃答。 |
| 经营指标与简报 | 中文订单示例提供成交营收、完成订单量和月度经营简报，可下载 Markdown、CSV 和 SVG。 |
| 工作台与试用账号 | 数据集切换、历史、取消、失败重跑、报表下载和运行概览。试用模式有登录及任务归属。 |
| 执行控制 | 单进程队列、幂等提交、状态持久化、步骤审计、重启恢复，以及 SQL 时间、行数和结果大小限制。 |
| 模型服务 | 通过 OpenAI 兼容的 Chat Completions 工具协议接入自托管模型。Qwen 是主要验证配置，可选 vLLM 会话推测解码。 |

中文订单数据由种子生成。UCI Online Retail II 是可选公开数据示例。你可以直接登记其他 SQLite 文件和查询 schema。接入 PostgreSQL、MySQL 等数据库需要新增只读执行适配器并验证取消、超时和 SQL 方言。新业务指标需要登记统计口径并实现查询。 [扩展步骤](docs/customization.md)

## 快速开始

需要 Python 3.11 或更高版本。合成数据准备和后端测试可以在 CPU 上完成，实际问答需要兼容的模型服务。工作台构建还需要 Node.js 24，多轮会话需要配套的 tokenizer 资源服务。

~~~bash
git clone https://github.com/ljf-lingd/AgentServe.git
cd AgentServe
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,test]"
python scripts/setup/prepare_public_demo.py
python -m pytest -q
~~~

已运行兼容模型服务时，把 [app.demo.json](configs/app.demo.json) 里的 `base_url` 和 `model` 改成服务返回的地址和模型名，或复制为自己的配置后使用 `--config`。随附示例指向 `http://127.0.0.1:8000/v1`、模型名 `agent-qwen`。

~~~bash
python -m agentserve --config configs/app.demo.json --database demo_orders --question "一共有多少位客户？"
~~~

合成数据库的金标是 **6 位客户**。工作台启动、合成订单和 UCI 示例见 [安装指南](docs/getting-started.md)。模型替换与私有数据接入见 [扩展指南](docs/customization.md)。

## 工作台验收

V7.1 在 RTX 4090 上使用 Qwen2.5-7B-Instruct，UCI 零售固定正式集 **47/48 轮正确**、**8/8 段多轮会话全对**、**零 SQL 超时**。唯一错误是把去重客户数查成了去重发票数。13 个历史超时任务经改写后，在原 1 秒 SQL 预算下全部符合金标。

同系列实验中，UCI 全量数据为 1,067,371 条交易明细。规模对照为 215/216，BIRD Mini-Dev SQLite 的严格结果集正确率为 36/500。模型、数据、查询口径和并发条件会影响这些结果。 [实验条件与复现入口](docs/evaluation.md)

## 目录

| 路径 | 内容 |
| --- | --- |
| [src/agentserve](src/agentserve) | API、任务应用层、模型协议、只读查询、会话、知识和分析 |
| [web](web) | React / TypeScript 浏览器工作台与 Playwright 测试 |
| [configs](configs) | API、模型与示例数据集配置 |
| [scripts](scripts) | 数据准备、模型探针、回放和验收 |
| [serving/vllm](serving/vllm) | 可选 vLLM 启动器与推测解码实现 |
| [tests](tests) | API、执行边界、恢复和业务功能回归 |

维护者为 [ljf-lingd](https://github.com/ljf-lingd)。一般问题可以通过 GitHub Issues 反馈，贡献前请看 [CONTRIBUTING.md](CONTRIBUTING.md)。

项目自有代码采用 [MIT 许可证](LICENSE)。数据、模型和上游资料的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
