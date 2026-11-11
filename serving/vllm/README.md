# vLLM serving

See [setup](../../docs/getting-started.md#2-连接模型并试跑-cli) for ordinary Qwen tool calling and [model configuration](../../docs/customization.md#替换模型) for model replacement. The application and model processes communicate through OpenAI-compatible HTTP.

`launch-qwen.sh` is the pinned Linux launcher. Place model files in `models/Qwen2.5-7B-Instruct` and point `VLLM_PYTHON` to the serving environment. The default profile uses ordinary decoding. `AGENTSERVE_SERVING_PROFILE=session-sam` selects the custom vLLM worker and serves `agent-qwen-sam`. Optional request metrics are enabled with `AGENTSERVE_REQUEST_METRICS=1`.

The validated SAM environment used vLLM 0.32.0 and its V1 model runner. Worker internals and tokenizer templates vary across vLLM versions. Validate both protocol behavior and model quality before adopting a different stack.
