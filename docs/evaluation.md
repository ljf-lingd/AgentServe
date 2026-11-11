# 评测与复现

| 评测 | 已记录结果 | 条件 |
| --- | --- | --- |
| V7.1 UCI 固定业务题 | 47/48 轮，8/8 段会话全对，0 次 SQL 超时 | RTX 4090，Qwen2.5-7B-Instruct，桌面同款工具与推测解码配置 |
| UCI 历史超时回放 | 13/13 金标、13/13 账本一致 | 保存的模型 SQL 经真实只读 SQLite 回放，SQL 预算 1 秒 |
| UCI 规模对照 | 215/216，0 次超时 | 约 1 万、10 万及全量交易，固定题集，并发 1 和 4 |
| Spider 冒烟 | 15/20 | 项目冻结的小样本 |
| BIRD Mini-Dev 2026 SQLite | 独立金标 500/500，模型 36/500 | 严格结果集口径，失败任务计入分母 |
| 知识配对 | BM25 有据题 8/12，无检索 0/12，两组缺证据均 12/12 弃答 | 同一模型与问题，辅助盲评 |
| V7 SAM 固定负载 | 两臂 32/32，开启组接受 1,055/2,007 个草稿 token | 同一 Qwen/vLLM 推理引擎，配对开关；后续复跑受当时 SQL 超时影响 |

V7.1 的一条错误查询将带 ID 的去重客户数误查为去重发票数。模型替换实测中，指定工具模式下 Qwen 为 48/48，Vicuna 为 4/48。模型协议接通只是可用性的一部分，应按目标库另做金标验证。

## 本地复验

~~~bash
python scripts/setup/prepare_public_demo.py
python -m pytest -q
npm --prefix web ci
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web run build
~~~

浏览器回归需要已启动的示例工作台与兼容模型。配置 `AGENTSERVE_TEST_URL` 后运行 `npm --prefix web test`，按部署选择合成订单、UCI 和试用账号用例。模型替身测试覆盖协议和执行故障边界。真实质量与延迟用独立 GPU 回放测量。

## UCI 金标与 GPU 回放

先按 [安装指南](getting-started.md) 准备全量零售库及工作台。省略 `--base-url` 只生成独立金标。加入地址后会提交实际 Task API 请求。

~~~bash
mkdir -p artifacts
python scripts/experiments/retail_replay.py data/databases/online_retail_ii/online_retail_ii.sqlite artifacts/retail-gold.json
python scripts/experiments/retail_replay.py data/databases/online_retail_ii/online_retail_ii.sqlite artifacts/retail-run.json --base-url http://127.0.0.1:8766
~~~

`scripts/experiments/retail_scale.py`、`retail_rag_pair.py`、`retail_sam_pair.py` 和 `retail_cross_domain.py` 提供进一步复现实验。运行前用 `--help` 核对参数，准备独立输出和状态目录，记录模型、tokenizer、配置、数据哈希、错误分母与并发。复现跨库实验前，分别从 [Spider](https://yale-lily.github.io/spider) 和 [BIRD](https://huggingface.co/datasets/birdsql/bird_mini_dev) 获取官方数据库。
