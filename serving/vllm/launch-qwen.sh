#!/usr/bin/env bash
# Start the validated local model configuration in the foreground.
set -euo pipefail

SERVING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="${AGENT_ROOT:-$(cd -- "$SERVING_DIR/../.." && pwd)}"
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
max_num_seqs=2
profile="${AGENTSERVE_SERVING_PROFILE:-ar}"
extra_args=()
enable_speculation() {
  model_len=8192
  export PYTHONPATH="$SERVING_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0
}

case "$profile" in
  ar) ;;
  session-sam)
    served_model=agent-qwen-sam
    enable_speculation
    extra_args+=(--worker-cls session_sam_worker.SessionSamWorker
      --speculative-config '{"method":"custom_class","model":"session_sam.SamProposer","num_speculative_tokens":4}')
    ;;
  suffix|suffix-agspec)
    served_model="agent-qwen-$profile"
    enable_speculation
    max_num_seqs=4
    export VLLM_PLUGINS=
    if [[ -n "${AGENTSERVE_ARCTIC_SITE:-}" ]]; then
      export PYTHONPATH="$AGENTSERVE_ARCTIC_SITE:$PYTHONPATH"
    fi
    CUDA_VISIBLE_DEVICES= "$VLLM_PYTHON" -c '
from importlib.metadata import version
if version("vllm") != "0.32.0+cu129" or version("arctic-inference") != "0.3.0":
    raise SystemExit("Suffix requires vLLM 0.32.0+cu129 and ArcticInference 0.3.0")
from arctic_inference.suffix_decoding import _C
' || { echo "Verify the pinned Suffix build and AGENTSERVE_ARCTIC_SITE." >&2; exit 1; }
    extra_args+=(--no-enable-prefix-caching
      --speculative-config '{"method":"suffix","num_speculative_tokens":4,"suffix_decoding_max_cached_requests":256}')
    if [[ "$profile" == suffix-agspec ]]; then
      extra_args+=(--worker-cls suffix_feedback_worker.SuffixFeedbackWorker)
    fi
    ;;
  *) echo "Unknown AGENTSERVE_SERVING_PROFILE: $profile" >&2; exit 2 ;;
esac
if [[ "${AGENTSERVE_REQUEST_METRICS:-0}" == 1 ]]; then
  extra_args+=(--enable-per-request-metrics)
  if [[ "$profile" != ar ]]; then
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
  --max-num-seqs "${MAX_NUM_SEQS:-$max_num_seqs}" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes "${extra_args[@]}"
