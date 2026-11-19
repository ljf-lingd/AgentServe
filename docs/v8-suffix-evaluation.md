# V8 Suffix 评测

使用 Qwen2.5-7B-Instruct、单张 RTX 4090、vLLM 0.32.0+cu129 和公开 UCI 全量数据。JCT 从任务入账到终态，包含应用等待、模型推理和工具执行。各组使用独立引擎和任务库，两轮反序运行。 [数据和环境准备](v5-reproduction.md#data-and-runtime) · [V8 启动说明](v8-release.md)

## 四并发 Suffix 独立负载

C4 使用 UCI 全量库的 **24 段三轮会话，共 72 轮**。6 个国别配对覆盖 8 个国家，指标包括销售额、取消金额、销售发票数和取消发票数。所有参考 SQL 与参数组合均与旧开发、正式负载无交集，并且本次组内无重复。数据、问题模板和金标规则继续采用 UCI 原始明细口径，因此结论适用于同零售域的新业务组合。

每波发起 4 个相互独立的会话。同一会话保留先后依赖，每轮等待四个会话各自终态再发起下轮。应用与引擎并发均为 4。按真实模型请求 ID 把并发响应归回已提交步骤，模型 prompt 和输出 token 单独保存。4 道已知开发题用于四组共 16 次小试，要求 SQL、结果和 token 与 AR 对齐。正式四组为 AR、Session-SAM、请求内 Suffix、跨请求 Suffix，反序再重复一次，共 **576 轮**。各组单次独立预热。正式结果记录匹配率、JCT P95、每秒任务和每秒正确任务。

```bash
export PYTHONPATH=src
python scripts/experiments/run_v5_suffix_c4.py freeze "$RUN"
c4_gpu=(--model-dir "$MODEL_DIR" --embedding-dir "$EMBEDDING_DIR" --vllm-python "$VLLM_PYTHON" --gpu-uuid "$GPU_UUID" --arctic-site "$ARCTIC_SITE")
python scripts/experiments/run_v5_suffix_c4.py smoke "$RUN" "${c4_gpu[@]}"
python scripts/experiments/run_v5_suffix_c4.py matrix "$RUN" "${c4_gpu[@]}"
```

负载构建和并发归因位于 `retail_c4.py`，启动复用原 Suffix 运行入口的隔离、重试与收据，数据在运行端由只读 UCI 独立计算。公开检出重新生成的预热、开发小试和正式 72 轮输入及金标与历史清单逐项一致。冻结清单还包含源码和资产指纹，复现时以该次检出生成的清单为准。

### 四并发真机结果

小试 16/16 正确，三个候选与 AR 逐题 token 对齐。正式 576 次提交中 568 次正确，576 次真实模型请求均有完整 token。四组两轮均为 **71/72**，每轮全部 72 道的 SQL、状态、结果及 prompt 和输出 token 逐题一致。

| 方案 | 两轮 JCT P95（秒） | 两轮正确任务吞吐（题/秒） |
| --- | ---: | ---: |
| AR | 4.227 / 4.214 | 0.981 / 0.986 |
| Session-SAM | 2.911 / 2.902 | 1.680 / 1.705 |
| 请求内 Suffix | 2.641 / 2.655 | 1.763 / 1.750 |
| 跨请求 Suffix | 1.759 / 1.761 | 2.280 / 2.292 |

跨请求 Suffix 相对 Session-SAM 的 P95 分别下降 **39.6% / 39.3%**，正确任务吞吐分别提高 **35.8% / 34.4%**。相对纯 AR 的 P95 下降 58.4% / 58.2%。跨请求 Suffix 在每轮正式请求接受 4,156 / 5,264 草稿 token，Session-SAM 为 2,993 / 5,211。该原生配置已接入 V8 工作台默认路径。 [V8 集成](v8-release.md)

四组共有的错误发生在第三轮只改变月份。第二轮已正确从比利时切到荷兰，第三轮模型实际输入也给出荷兰的最新成功 SQL，但输出回退到比利时。销售额 6,315.76 英镑来自错误国家，荷兰独立参考为 25,874.01 英镑。数据库哈希复查一致。这个共同错答计入了各组分母。

## 可选的 AgSpec 风格反馈

`suffix-feedback` 保留原生候选和相同缓存，仅以 AgSpec 风格的请求内拒绝反馈裁剪候选前缀。以上一步实际验证数量和接受数更新尺度，跳过硬上限全接受样本。尺度从 1 开始，候选最多 4。不引入 workspace 或新的语料来源。功能通过独立 worker 接入 vLLM V1。V5 原四组实验仍可按原入口运行，`--feedback` 选择第五组，并导出与原生全局组的直接比较。

在 Linux 屏蔽 GPU，使用已安装的原生 C++ 扩展将两轮 AR 的 **100 条**历史 prompt 和输出 token 按固定顺序重放。无预热。每轮原生与反馈均为 **3,361** 个提出、**2,487** 个接受、**1,109** 个模拟目标步骤。反馈裁剪 **0** 次。合成拒绝序列验证了控制器可以缩短候选并保持请求间状态隔离。后续若换到拒绝密度高或并发大的独立负载，应同时测草稿浪费、配对成功率、TPOT、JCT 与成功吞吐。

```bash
CUDA_VISIBLE_DEVICES="" VLLM_PLUGINS="" \
PYTHONPATH="$ARCTIC_SITE:serving/vllm" \
"$VLLM_PYTHON" scripts/experiments/replay_suffix_feedback.py "$P1_ARCHIVE" "$CPU_RECEIPT"
```
