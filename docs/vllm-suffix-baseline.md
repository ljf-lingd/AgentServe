# vLLM 原生 Suffix 对照

AgentServe 的 Session-SAM 通过请求扩展传入会话草稿。vLLM 原生 Suffix 在服务端管理候选，且可以缓存其他请求的响应。两种后端通过相同的 Chat Completions 接口返回接受量。AgentServe 将服务端返回的计数写入模型步骤账本及运行概览，未提供指标的请求显示为空。 [vLLM 逐请求指标](https://docs.vllm.ai/en/v0.32.0/features/speculative_decoding/acceptance_metrics/)

## 环境和兼容性

已在 Linux 用 vLLM `0.32.0+cu129`、Python 3.11 和 ArcticInference `0.3.0` 验证 Suffix C++ 扩展可以导入。源包需要支持 C++20 `<span>` 的编译器。默认 GCC 9 编译失败，改用 GCC 11 后，构建参数还需显式指定 CMake 查找的 `Python_EXECUTABLE` 和 `Python_ROOT_DIR`，使扩展的 ABI 与运行 vLLM 的解释器一致。依赖和扩展可以安装到单独的 `--target` 目录。原生路径的实验进程设 `VLLM_PLUGINS=`，以排除 Arctic 的额外 vLLM 插件。 [ArcticInference 发布包](https://pypi.org/project/arctic-inference/0.3.0/) · [vLLM Suffix 配置](https://docs.vllm.ai/en/v0.32.0/features/speculative_decoding/suffix/)

CPU 用例先向一个请求追加 `[10, 11, 12, 13]`，再用 `[10, 11]` 查找续写。缓存容量为 0 时，同请求得到 `[12]`，结束请求后新请求得到空候选。容量为 256 时，新请求仍得到 `[12]`。这验证了该版本原生扩展的请求边界。

在 Linux 上先准备 GCC 11、CMake、Ninja 和独立的 vLLM Python 环境。从 ArcticInference `0.3.0` 的公共发行页取得源码包。验证包指纹 `7f22e3e1b1071c6061c0d6951be17c0fb50b08d06d6e543144c8f52faba60a36`。在独立的 `$BUILD_DEPS` 目录安装 `setuptools==84.2.1`、`nanobind==2.9.2`、`grpcio-tools==1.76.0` 和 `protobuf==6.33.6` 及其构建依赖。

```bash
PYTHON_ROOT=$("$VLLM_PYTHON" -c 'import sys; print(sys.prefix)')
CC=gcc-11 CXX=g++-11 PYTHONPATH="$BUILD_DEPS" \
  ARCTIC_INFERENCE_PRECOMPILED_OPS=0 \
  CMAKE_ARGS="-DPython_EXECUTABLE=$VLLM_PYTHON -DPython_ROOT_DIR=$PYTHON_ROOT" \
  "$VLLM_PYTHON" -m pip install --no-build-isolation --no-deps --target "$ARCTIC_SITE" "$ARCTIC_SDIST"
PYTHONPATH="$ARCTIC_SITE" VLLM_PLUGINS= "$VLLM_PYTHON" -c 'from arctic_inference.suffix_decoding import _C; print(_C.__file__)'
```

## 路线图 V5 真机比较

[四组复现实验](v5-reproduction.md#p1-native-suffix-and-session-sam)统一使用四 token 草稿上限，在两次轮换的顺序中使用同一 48 轮 UCI 任务与独立 SQL 金标。启动时校验权重、编译扩展和配置。真实协议小试覆盖文本、JSON schema、强制及自动工具调用。四组的应用任务时间与真机 token 序列均保留。

## 手动 16 token 探针

| 模式 | vLLM 配置 |
| --- | --- |
| `ar` | 无 `--speculative-config` |
| `suffix-local` | `{"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_cached_requests":0}` |
| `suffix-global` | `{"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_cached_requests":256}` |

每组以新引擎进程启动，固定目标权重、GPU、vLLM 版本、并发 1 和工具解析器。初测关闭 prefix caching，所有组启用 `--enable-per-request-metrics`，Suffix 组再启用 `--per-request-spec-decode-metrics detailed`。AgentServe 每组在独立任务库运行相同的 8 段零售会话、16 个用户轮次，答案由原始明细 SQL 独立判定。反转第二轮的组序可以检查顺序影响。工具强制调用、自动调用、JSON schema、长文本的重复探针在零售任务之后运行，避免把专门的重复探针写入零售测试的共享缓存。

```bash
mkdir run-suffix-local
python scripts/experiments/vllm_suffix_probe.py configs/app.v7.retail.json run-suffix-local \
  --model agent-qwen --model-url http://127.0.0.1:8000/v1 \
  --resources-url http://127.0.0.1:8002 --arm suffix-local --block 1
```

`result.json` 包含逐任务 SQL 正确性、数据库金标哈希、JCT、草稿计数与指标覆盖率，以及逐协议探针的 token ID 和延迟。零售账本指标只统计正式会话，Prometheus 差量包含一次预热。`model_replies` 和 `replies_with_draft_metrics` 用于判别缺测和实际的零草稿。小样本中先检查正确性、协议及接受量，再按同一会话配对比较 JCT。
