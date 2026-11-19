# AgentServe V8

V8 双库工作台默认使用 vLLM 原生跨请求 Suffix。三种推理配置共用 Task API、步骤账本和工作台，支持切换后继续已有会话。

| `AGENTSERVE_SERVING_PROFILE` | 服务模型名 | 草稿来源 |
| --- | --- | --- |
| `suffix` | `agent-qwen-suffix` | V8 默认。原生跨请求 Suffix，最多缓存 256 个请求，每步至多四个草稿 token。 |
| `suffix-agspec` | `agent-qwen-suffix-agspec` | 同一原生候选，使用请求内的实际接受与拒绝反馈裁剪草稿前缀。 |
| `session-sam` | `agent-qwen-sam` | 原有会话后缀自动机与验证反馈基线。 |

## 从公开仓库启动

在仓库根目录按 [安装指南](getting-started.md) 安装应用依赖、准备合成订单与 UCI SQLite、启动兼容 tokenizer 的资源服务并构建工作台。`prepare_public_demo.py` 会生成 V8 使用的知识快照。首次使用可选择独立状态目录，已有工作台可以继续使用原目录。

在 Linux 推理环境准备本地 Qwen2.5-7B-Instruct 权重和 `VLLM_PYTHON`。两个 Suffix 配置使用 vLLM `0.32.0+cu129` 和 ArcticInference `0.3.0`。如果扩展安装在独立目录，将 `ARCTIC_SITE` 指向该目录。 [推理启动器](../serving/vllm/README.md) · [原生扩展构建与验证](vllm-suffix-baseline.md#环境和兼容性)

```bash
AGENTSERVE_SERVING_PROFILE=suffix AGENTSERVE_ARCTIC_SITE="$ARCTIC_SITE" \
  AGENTSERVE_REQUEST_METRICS=1 MAX_NUM_SEQS=4 PORT=8000 bash serving/vllm/launch-qwen.sh
```

在应用环境另开终端，确认配置内的模型和资源端点可达，再启动工作台。

```bash
export AGENTSERVE_CONFIG=configs/app.v8.json
export AGENTSERVE_SERVING_PROFILE=suffix
export AGENTSERVE_WEB_MODE=demo
export AGENTSERVE_WEB_STATE_DIR=state/workbench
python -m uvicorn agentserve.api.app:create_app --factory --host 127.0.0.1 --port 8766 --workers 1
```

打开 [工作台](http://127.0.0.1:8766/app/)。切换方案时，两端选择表中同一个配置，重启后从 `/v1/models` 核对模型名。CLI 使用 `--config configs/app.v8.json`。应用会在任务账本记录实际推理配置。

跨请求 Suffix 的历史缓存属于整个推理进程。仅在缓存可共享的请求范围内使用这套部署配置。

## 验收

三个模型配置在 RTX 4090 上通过 24/24 次四并发协议检查，涵盖文本、JSON Schema、命名工具和自动工具。同一批协议检查中，原生版和反馈版的 8 对输入及输出 token 一致。默认配置还通过实际工作台任务接口完成 UCI 去重客户查询，结果 29 与独立统计一致。性能对照采用原生 Suffix。 [评测分母、结果和复现命令](v8-suffix-evaluation.md) · [任务恢复与业务回归](v5-reliability.md)
