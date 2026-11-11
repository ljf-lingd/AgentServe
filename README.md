# AgentServe

面向业务数据的只读分析助手。用自然语言查询 SQLite，在同一会话里追问时间、地区或指标。浏览器工作台同时提供业务规则问答、经营指标和简报。Task API 保存查询证据、取消状态和执行步骤，便于核对答案与定位失败。

AgentServe is a self-hosted, read-only SQL and analytics agent with a persistent task API, configurable model serving, and a browser workbench.

[安装与启动](docs/getting-started.md) · [接入自己的模型和数据](docs/customization.md) · [架构与 API](docs/architecture.md) · [评测和复现](docs/evaluation.md)

## 已实现

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

需要 Python 3.11 或更高版本。工作台另需 Node.js 24。只有模型推理需要连接模型服务，合成数据准备和后端测试可以在 CPU 上完成。

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

## 运行与评测

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
