# 架构与接口

~~~mermaid
flowchart LR
    CLI[CLI] --> APP[Application]
    UI[React 工作台] --> API[FastAPI]
    API --> APP
    APP --> Queue[任务队列与步骤账本]
    APP --> Agent[SQL Agent]
    APP --> Rules[知识检索与批准口径]
    Agent --> Model[模型服务]
    Agent --> Query[只读 SQLite]
    Queue --> State[独立 SQLite 状态库]
    Rules --> Query
~~~

CLI 与 API 使用同一应用组装入口。直连 CLI 执行一次任务，持久 API 保存任务和步骤。会话按数据集绑定，回放和恢复根据已提交步骤与执行身份进行。知识检索按数据集和生效时间约束。指标和简报复用批准口径，结果带出处或产物。

| 层 | 主要职责 |
| --- | --- |
| `api` | 请求校验、工作台和账号访问 |
| `application` | 组合模型、查询、记忆、知识和分析服务 |
| `runtime` / `storage` | 队列、恢复、取消、事件与 SQLite 事务 |
| `agents` / `llm` / `tools` | 工具协议、SQL 生成、模型边界、只读执行 |
| `memory` / `knowledge` / `analytics` | 会话状态、版本化规则和经营简报 |

## Task API

后端独立模式在 `/docs` 提供 OpenAPI。启动前显式指定已准备好的配置。

~~~bash
AGENTSERVE_CONFIG=configs/app.demo.json python -m uvicorn agentserve.api.app:create_app --factory --host 127.0.0.1 --port 8080 --workers 1
~~~

~~~bash
curl -X POST http://127.0.0.1:8080/tasks -H 'Content-Type: application/json' -d '{"database_id":"demo_orders","question":"How many customers are there?"}'
curl http://127.0.0.1:8080/tasks/TASK_ID
~~~

| 入口 | 用途 |
| --- | --- |
| `POST /tasks` | 提交查数、知识、指标或简报。可提供 `Idempotency-Key\)。 |
| `GET /tasks/{task_id}` | 读取状态、错误、最终报告与查询证据。 |
| `POST /tasks/{task_id}/cancel` | 请求取消。 |
| `GET /tasks/{task_id}/steps` | 独立 API 模式下查看步骤和尝试。 |
| `POST /sessions` | 为指定数据库创建会话。后续任务携带 `session_id`。 |
| `GET /catalog/datasets` | 工作台模式下查看已开放的数据集能力。 |

状态事务由 storage 管理。外部模型等待和业务 SQL 执行在状态事务外运行。查询使用 SQLite 只读连接、授权限制和执行中断。行数或结果字节触及边界时会标记截断。任务恢复比较执行身份，已保存结果用于后续读取。

## 部署边界

用一个应用 worker 连接一份独立状态库，模型服务可独立部署。例子使用 loopback。试用账号模式将 `AGENTSERVE_WEB_MODE` 设为 `auth`，并使用单独的 `AGENTSERVE_WEB_STATE_DIR`。运行 `agentserve-account --help` 查看账号创建和会话撤销命令。对外试用需在部署层配置 HTTPS、访问限制与日志保留策略。问句、查询结果与步骤属于业务数据，应按使用方的数据权限处理。
