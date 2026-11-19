# vLLM serving

See [setup](../../docs/getting-started.md#2-连接模型并试跑-cli) for ordinary Qwen tool calling and [model configuration](../../docs/customization.md#替换模型) for model replacement. The application and model processes communicate through OpenAI-compatible HTTP.

`launch-qwen.sh` is the pinned Linux launcher. Place model files in `models/Qwen2.5-7B-Instruct` and point `VLLM_PYTHON` to the serving environment. The default profile uses ordinary decoding. `AGENTSERVE_SERVING_PROFILE=session-sam` selects the custom vLLM worker and serves `agent-qwen-sam`. Optional request metrics are enabled with `AGENTSERVE_REQUEST_METRICS=1`.

The validated SAM environment used vLLM 0.32.0 and its V1 model runner. Worker internals and tokenizer templates vary across vLLM versions. Validate both protocol behavior and model quality before adopting a different stack.

## V8 profiles

The V8 application config and launcher share `AGENTSERVE_SERVING_PROFILE`. For the V8 workbench set it to `suffix`, `suffix-agspec`, or `session-sam` in both processes. Suffix profiles use up to four draft tokens and cache up to 256 requests in the engine. Set `AGENTSERVE_ARCTIC_SITE` when ArcticInference is installed in a separate directory. [Startup and profile matrix](../../docs/v8-release.md) · [Pinned Suffix build](../../docs/vllm-suffix-baseline.md#环境和兼容性)
