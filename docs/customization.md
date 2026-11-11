# 接入自己的模型和数据

先把 [通用 SQLite 配置](../configs/app.custom.example.json) 复制为 `configs/app.local.json`。它以生成的演示库为默认输入，可直接替换为自己的路径。所有相对路径以配置文件所在目录为基准。

## 替换模型

1. 启动支持 Chat Completions 工具调用的服务，用 `GET /v1/models` 核对服务中的模型名。
2. 设置 `base_url`（包含 `/v1`）和 `model`。当前应用客户端连接无需应用层密钥的受信任端点。需要鉴权的服务应使用部署环境提供的受信任网关或扩展客户端，凭据由部署环境管理。
3. 常规解码将 `session_sam` 和 `session_sam_all_phases` 设为 `false`。默认 `model_tool_choice=auto` 保留模型澄清。若服务只稳定支持指定工具，可用 `named` 验证条件完整的 SQL 问题，自动澄清路径在此模式下不可用。
4. 需要多轮会话时，`memory.resources_url` 指向使用同一生成模型 tokenizer 和聊天模板的资源服务。新部署先清空旧的 `memory.resources_identity`，用新状态目录验证。修改模型、模板、口径或资源身份后重新运行固定问题。

~~~bash
python scripts/experiments/model_protocol_probe.py configs/app.local.json artifacts/model-protocol.json
~~~

探针分别检查 SQL 工具、澄清、JSON 结构和文本。协议通过后，再用自己的金标评估查询正确性。Qwen 与 Vicuna 的历史实测质量差异见 [评测](evaluation.md)。

## 接入其他 SQLite 数据

1. 准备只读 SQLite 文件与对应 schema 文本。可使用 `sqlite3 database.sqlite .schema` 导出，再整理成供模型读取的表、视图、关联与字段说明。业务敏感表应从专用查询副本移除。
2. 在 `databases` 中给数据集分配一个稳定 ID。设置 `path`、`schema_path`、`context`。在 `context` 中说明币种、时间粒度、默认状态口径、空值和聚合单位。用新路径发布数据库版本，用新 ID 分隔需要独立历史的场景。
3. 先走 CLI，再提交 Task API 请求。

~~~bash
python -m agentserve --config configs/app.local.json --database my_data --question "数据里一共有多少条记录？"
~~~

~~~json
{"database_id":"my_data","question":"上个月完成了多少笔交易？"}
~~~

4. 需要在工作台显示时，给数据库添加 `workbench` 描述。字段包括 `label`、`first_date`、`last_date`、`currency`、`modes`、`data_description`、`rule_description`、`query_examples` 和 `rule_examples`。先启用 `["sql"]`，并参考 [工作台样例](../configs/app.workbench.example.json) 配置会话资源。先验证历史、数据集切换和空结果，再开放其他模式。

### 业务知识、指标与简报

- 业务规则通过文档和 manifest 登记有效期、来源、适用数据库与属性。参考 [规则示例](../data/knowledge_v31)。
- 用 `python -m agentserve.knowledge.snapshot path/to/manifest.json cache/my-knowledge.json` 生成快照，再在 `knowledge.snapshot_path` 指向它。对需要强约束的业务属性，复用 `properties.json` 的范围和证据约定。
- 指标使用经批准的规则版本和应用端 SQL 编译。简报复用批准口径、只读结果与产物生成。接入新领域时在 application 层接线，补充对应的查询、口径和独立金标，然后在工作台登记能力。示例规则见 [指标口径](../data/knowledge_v32)。

### 接入其他数据库引擎

现有查询工具和状态账本均使用 SQLite。迁移查询端时从 [只读执行工具](../src/agentserve/tools/sqlite.py) 的职责出发，实现新方言的 schema 提示、只读权限、资源生命周期、可中断执行、结果限制和错误映射，并在 [应用组装](../src/agentserve/application/bootstrap.py) 中明确配置。先以独立状态库验证并发、取消、重启和只读边界。任务状态库可以继续使用 SQLite。
