#!/usr/bin/env bash
# Start the validated local model configuration in the foreground.
set -euo pipefail

AGENT_ROOT="${AGENT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
VLLM_PYTHON="${VLLM_PYTHON:-$HOME/miniconda3/envs/agent-vllm/bin/python}"
MODEL_DIR="$AGENT_ROOT/models/Qwen2.5-7B-Instruct"

[[ -x "$VLLM_PYTHON" ]] || { echo "Missing serving Python: $VLLM_PYTHON" >&2; exit 1; }
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || { echo "Missing local model: $MODEL_DIR" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$AGENT_ROOT/cache/huggingface"
export VLLM_CACHE_ROOT="$AGENT_ROOT/cache/vllm"
export TRITON_CACHE_DIR="$AGENT_ROOT/cache/triton"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p -- "$HF_HOME" "$VLLM_CACHE_ROOT"

served_model=agent-qwen
model_len=2048
extra_args=()
case "${AGENTSERVE_SERVING_PROFILE:-ar}" in
  ar) ;;
  session-sam)
    served_model=agent-qwen-sam
    model_len=8192
    export PYTHONPATH="$AGENT_ROOT/serving/vllm${PYTHONPATH:+:$PYTHONPATH}"
    export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0
    extra_args+=(--worker-cls session_sam_worker.SessionSamWorker
      --speculative-config '{"method":"custom_class","model":"session_sam.SamProposer","num_speculative_tokens":4}')
    ;;
  *) echo "Unknown AGENTSERVE_SERVING_PROFILE: $AGENTSERVE_SERVING_PROFILE" >&2; exit 2 ;;
esac
if [[ "${AGENTSERVE_REQUEST_METRICS:-0}" == 1 ]]; then
  extra_args+=(--enable-per-request-metrics)
  if [[ "$served_model" == agent-qwen-sam ]]; then
    extra_args+=(--per-request-spec-decode-metrics detailed)
  fi
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-$model_len}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.45}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes "${extra_args[@]}"
