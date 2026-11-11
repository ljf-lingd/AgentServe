#!/usr/bin/env bash
# Start a loopback V6 workbench and own its temporary API, CPU, and GPU processes.
set -euo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
AGENT_ROOT="${AGENT_ROOT:-$SOURCE_ROOT}"
MODE="${AGENTSERVE_WEB_MODE:-demo}"
API_PORT="${API_PORT:-8766}"
MODEL_PORT="${MODEL_PORT:-8760}"
RESOURCE_PORT="${RESOURCE_PORT:-8762}"
API_PYTHON="${API_PYTHON:-$HOME/miniconda3/envs/agent/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-$HOME/miniconda3/envs/agent-vllm/bin/python}"
RUN_DIR="${1:?Usage: bash scripts/validation/run_v6_demo.sh RUN_DIRECTORY}"
mkdir -p -- "$RUN_DIR"
RUN_DIR="$(cd -- "$RUN_DIR" && pwd)"
[[ -f "$SOURCE_ROOT/web/dist/index.html" ]] || { echo "Build web/dist before launching"; exit 1; }
case "$MODE" in demo|auth) ;; *) echo "Set AGENTSERVE_WEB_MODE=demo or auth"; exit 2;; esac
"$API_PYTHON" - "$API_PORT" "$MODEL_PORT" "$RESOURCE_PORT" <<'PY'
import socket
import sys
from contextlib import ExitStack
with ExitStack() as stack:
    for port in sys.argv[1:]:
        stack.enter_context(socket.socket()).bind(("127.0.0.1", int(port)))
PY

pids=()
stop() {
  trap - EXIT INT TERM HUP
  for ((i=${#pids[@]}-1; i>=0; i--)); do
    kill -TERM -- "-${pids[i]}" 2>/dev/null || true
  done
  for ((i=0; i<20; i++)); do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi
    done
    [[ "$alive" == 0 ]] && break
    sleep 1
  done
  for pid in "${pids[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
    wait "$pid" 2>/dev/null || true
  done
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RUN_DIR/gpu-after.csv"
  printf '%s\n' "${pids[@]}" > "$RUN_DIR/owned-service-pids.txt"
}
trap stop EXIT
trap 'exit 130' INT TERM HUP
printf '%s\n' "$$" > "$RUN_DIR/launcher.pid"

ready() {
  local pid="$1" url="$2" attempts="$3"
  for ((i=0; i<attempts; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
    if curl --max-time 2 -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

setsid env CUDA_VISIBLE_DEVICES= \
  AGENT_EMBEDDING_PATH="$AGENT_ROOT/models/Qwen3-Embedding-0.6B" \
  AGENT_TOKENIZER_PATH="$AGENT_ROOT/models/Qwen2.5-7B-Instruct" \
  "$VLLM_PYTHON" -m uvicorn memory_resources:create_app --factory \
  --app-dir "$AGENT_ROOT/scripts/experiments" --host 127.0.0.1 --port "$RESOURCE_PORT" \
  > "$RUN_DIR/resources.log" 2>&1 &
cpu=$!
pids+=("$cpu")

setsid env AGENT_ROOT="$AGENT_ROOT" VLLM_PYTHON="$VLLM_PYTHON" \
  AGENTSERVE_SERVING_PROFILE=session-sam AGENTSERVE_REQUEST_METRICS="${AGENTSERVE_REQUEST_METRICS:-1}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PORT="$MODEL_PORT" MAX_NUM_SEQS=4 \
  bash "$AGENT_ROOT/serving/vllm/launch-qwen.sh" > "$RUN_DIR/vllm.log" 2>&1 &
model=$!
pids+=("$model")
printf 'run=%s resource_pid=%s model_pid=%s\n' "$RUN_DIR" "$cpu" "$model"
ready "$cpu" "http://127.0.0.1:$RESOURCE_PORT/info" 65
ready "$model" "http://127.0.0.1:$MODEL_PORT/v1/models" 240
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RUN_DIR/gpu-ready.csv"

setsid env PYTHONPATH="$SOURCE_ROOT/src" AGENTSERVE_WEB_MODE="$MODE" \
  AGENTSERVE_CONFIG="$AGENT_ROOT/configs/app.v5.json" \
  AGENTSERVE_WEB_STATE_DIR="${AGENTSERVE_WEB_STATE_DIR:-$RUN_DIR}" \
  AGENTSERVE_WEB_MODEL_PORT="$MODEL_PORT" AGENTSERVE_WEB_RESOURCE_PORT="$RESOURCE_PORT" \
  "$API_PYTHON" -m uvicorn agentserve.api.app:create_app --factory --host 127.0.0.1 \
  --port "$API_PORT" > "$RUN_DIR/api.log" 2>&1 &
api=$!
pids+=("$api")
ready "$api" "http://127.0.0.1:$API_PORT/healthz" 60
printf 'ready http://127.0.0.1:%s/app/ api_pid=%s\n' "$API_PORT" "$api"
wait "$api"
