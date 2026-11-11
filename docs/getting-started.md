# 安装与启动

在仓库根目录执行命令。应用使用 Python 3.11+ 和 SQLite，构建工作台需要 Node.js 24。应用和推理服务可以运行在不同机器上，二者通过 HTTP 连接。

## 1. 准备应用与示例数据

~~~bash
python -m venv .venv
# Bash: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,test]"
python scripts/setup/prepare_public_demo.py
~~~

准备脚本生成两个合成订单数据库、查询 schema 和知识快照。查询库与任务状态库使用不同路径。后端回归可通过 `python -m pytest -q` 运行。测试使用本地 SQLite 和模型协议替身。

## 2. 连接模型并试跑 CLI

可使用已运行的兼容模型端点，或在支持 vLLM 的环境中启动 Qwen。下面命令在**推理环境**执行，模型权重通过模型发布者的渠道准备。

~~~bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name agent-qwen --host 127.0.0.1 --port 8000 --enable-auto-tool-choice --tool-call-parser hermes
~~~

回到应用环境，确认模型名与配置一致，再运行查询。

~~~bash
curl http://127.0.0.1:8000/v1/models
python -m agentserve --config configs/app.demo.json --database demo_orders --question "一共有多少位客户？"
~~~

期望查询结果为 `[[6]]`。若只使用 CLI 单轮查询或非会话 Task API，可以省略配置中的 `memory`、`knowledge`、`metric`。一般模型部署使用常规解码。会话推测解码需要本项目对应的 vLLM 启动配置。 [vLLM 工具调用说明](https://docs.vllm.ai/en/stable/features/tool_calling/)

## 3. 启动工作台

工作台查数历史使用持久会话。先启动与生成模型匹配的 tokenizer 资源服务，再构建网页。资源服务需要在单独的 Python 环境安装 `torch`、`transformers` 和 `huggingface_hub`，并为该环境安装本项目。以下示例使用 Bash。PowerShell 可以用 `$env:变量名 = '值'` 设置同一变量。

~~~bash
# 在资源服务环境执行。提前把生成模型 tokenizer 文件放入 AGENT_TOKENIZER_PATH。
python scripts/setup/prepare_memory_model.py --source huggingface
export AGENT_TOKENIZER_PATH="$PWD/models/Qwen2.5-7B-Instruct"
export AGENT_EMBEDDING_PATH="$PWD/models/Qwen3-Embedding-0.6B"
python -m uvicorn memory_resources:create_app --factory --app-dir scripts/experiments --host 127.0.0.1 --port 8002
~~~

~~~bash
# 在应用环境执行。
npm --prefix web ci
npm --prefix web run build
export AGENTSERVE_CONFIG=configs/app.workbench.example.json
export AGENTSERVE_WEB_MODE=demo
export AGENTSERVE_WEB_STATE_DIR=state/workbench
python -m uvicorn agentserve.api.app:create_app --factory --host 127.0.0.1 --port 8766 --workers 1
~~~

打开 [工作台](http://127.0.0.1:8766/app/)。可先问「2026 年 3 月上海有多少笔订单」，再问「那深圳呢」。业务规则、指标和简报使用示例订单定义。

`demo` 模式用于本地试用。若需独立试用账号，参照 [试用配置与部署边界](architecture.md#部署边界)。更换模型端口可编辑配置，或在工作台启动时设置 `AGENTSERVE_WEB_MODEL_PORT`。资源服务端口对应 `AGENTSERVE_WEB_RESOURCE_PORT`。

## 4. 可选 UCI 零售数据

从 [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) 下载官方 ZIP 或 XLSX。在应用环境生成新的 SQLite 文件。

~~~bash
python -m pip install -e ".[retail]"
python scripts/setup/prepare_online_retail_ii.py /path/to/online-retail-ii.zip data/databases/online_retail_ii
~~~

复制 `configs/app.v7.retail.json` 作为自己的工作台配置。如果服务使用常规解码，将两个 SAM 开关设为 `false`，并将 `model` 改为服务返回的模型名。然后用新配置重启工作台，切换至「UCI 零售交易」。UCI 支持查数和业务规则，问法示例为「英国 2011 年 11 月的取消金额是多少」。

## 排查

| 现象 | 检查 |
| --- | --- |
| 启动时文件缺失 | 先运行数据准备脚本。路径相对于配置文件所在目录。V7 双库配置另需要 UCI SQLite。 |
| 模型连接错误 | 检查 `base_url` 包含服务的 `/v1` 前缀、端口可达，模型名与 `/v1/models` 一致。 |
| 工具调用错误 | 核对该模型的工具模板和解析器。模型协议探针见 [扩展指南](customization.md#替换模型)。 |
| 多轮会话无法创建 | 确认配置了 `memory`，资源服务可达，且 tokenizer 与模型聊天模板一致。 |
| 页面 404 | 运行前端构建，以工作台模式启动 API，然后访问 `/app/`。 |
