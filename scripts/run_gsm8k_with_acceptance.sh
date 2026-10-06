#!/usr/bin/env bash
# Run the existing vLLM GSM8K evaluator and collect acceptance counter deltas.
# Usage: bash scripts/run_gsm8k_with_acceptance.sh [run-name]
# Draft sampling follows the running vLLM service configuration.
set -euo pipefail

# ==================== Editable evaluation settings ====================
# Environment variables override these defaults; the first argument overrides
# RUN_NAME. MODEL must match the service's --served-model-name.
RUN_NAME="${1:-qwen38-baseline-lr3e4-step7812-nonthinking-setting2}"
HOST="${EVAL_HOST:-http://127.0.0.1}"
PORT="${EVAL_PORT:-8000}"
MODEL="${EVAL_MODEL:-qwen38-flash-next}"

NUM_QUESTIONS="${EVAL_NUM_QUESTIONS:-1319}"
NUM_SHOTS="${EVAL_NUM_SHOTS:-5}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
TOP_P="${EVAL_TOP_P:-0.8}"
TOP_K="${EVAL_TOP_K:-20}"
MIN_P="${EVAL_MIN_P:-0}"
SEED="${EVAL_SEED:-42}"
MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-256}"
REQUEST_TIMEOUT_SECONDS="${EVAL_REQUEST_TIMEOUT_SECONDS:-600}"
METRICS_WAIT_SECONDS="${EVAL_METRICS_WAIT_SECONDS:-10}"

# Set USE_CHAT_COMPLETIONS=false to use /v1/completions instead.
USE_CHAT_COMPLETIONS="${EVAL_USE_CHAT_COMPLETIONS:-true}"
# Empty values keep the server defaults. These options require chat completions.
# Example: CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
CHAT_TEMPLATE_KWARGS="${EVAL_CHAT_TEMPLATE_KWARGS:-}"
REASONING_EFFORT="${EVAL_REASONING_EFFORT:-}"

# Paths and output location.
ROOT="${DSPARK_ROOT:-/public/workspace/dspark}"
VLLM_DIR="${VLLM_DIR:-$ROOT/vllm-env/vllm}"
SPEC_DIR="${SPECFORGE_DIR:-$ROOT/SpecForge}"
PYTHON_BIN="${PYTHON_BIN:-$VLLM_DIR/../.venv-vllm-custom/bin/python}"
OUT_ROOT="${EVAL_OUTPUT_DIR:-$ROOT/logs/eval-qwen38-flash-next}"
# ==================== Evaluation workflow ====================
STAT_SCRIPT="$SPEC_DIR/scripts/stat_vllm_acceptance.py"
EVAL_SCRIPT="$VLLM_DIR/tests/evals/gsm8k/gsm8k_eval.py"

if [[ $# -gt 1 || ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Usage: bash $0 [run-name: letters, numbers, dots, underscores, hyphens]" >&2
  exit 2
fi
if [[ "$USE_CHAT_COMPLETIONS" != true && "$USE_CHAT_COMPLETIONS" != false ]]; then
  echo "USE_CHAT_COMPLETIONS must be true or false" >&2
  exit 2
fi
if [[ "$USE_CHAT_COMPLETIONS" == false && ( -n "$CHAT_TEMPLATE_KWARGS" || -n "$REASONING_EFFORT" ) ]]; then
  echo "CHAT_TEMPLATE_KWARGS and REASONING_EFFORT require USE_CHAT_COMPLETIONS=true" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$STAT_SCRIPT" || ! -f "$EVAL_SCRIPT" ]]; then
  echo "Check these paths:" >&2
  printf '%s\n' "$PYTHON_BIN" "$STAT_SCRIPT" "$EVAL_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
RUN_DIR="$(mktemp -d "$OUT_ROOT/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
printf 'Results directory: %s\n' "$RUN_DIR"

EVAL_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL"
  --num-questions "$NUM_QUESTIONS"
  --num-shots "$NUM_SHOTS"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --seed "$SEED"
  --max-concurrency "$MAX_CONCURRENCY"
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
  --save-results "$RUN_DIR/gsm8k.json"
)
if [[ "$USE_CHAT_COMPLETIONS" == true ]]; then
  EVAL_ARGS+=(--use-chat-completions)
fi
if [[ -n "$CHAT_TEMPLATE_KWARGS" ]]; then
  EVAL_ARGS+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  EVAL_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

"$PYTHON_BIN" "$STAT_SCRIPT" snapshot \
  --url "${HOST%/}:$PORT" \
  --output "$RUN_DIR/before.json" 2>&1 | tee "$RUN_DIR/before.log"

(
  cd "$VLLM_DIR"
  "$PYTHON_BIN" "$EVAL_SCRIPT" "${EVAL_ARGS[@]}"
) 2>&1 | tee "$RUN_DIR/gsm8k.log"

echo "Waiting $METRICS_WAIT_SECONDS seconds for metrics to update..."
sleep "$METRICS_WAIT_SECONDS"

"$PYTHON_BIN" "$STAT_SCRIPT" snapshot \
  --url "${HOST%/}:$PORT" \
  --output "$RUN_DIR/after.json" 2>&1 | tee "$RUN_DIR/after.log"

"$PYTHON_BIN" "$STAT_SCRIPT" diff \
  --before "$RUN_DIR/before.json" \
  --after "$RUN_DIR/after.json" \
  --output "$RUN_DIR/acceptance.json" 2>&1 | tee "$RUN_DIR/acceptance.log"

printf '\nCompleted. Results and logs: %s\n' "$RUN_DIR"
