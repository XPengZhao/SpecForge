#!/usr/bin/env bash
# Evaluate a fixed random sample of regen prompts and collect acceptance metrics.
# Usage: bash scripts/run_regen_with_acceptance.sh [run-name]
# Draft sampling follows the running vLLM service configuration.
set -euo pipefail

# ==================== Editable evaluation settings ====================
# Environment variables override these defaults; the first argument overrides
# RUN_NAME. MODEL must match the service's --served-model-name.
RUN_NAME="${1:-qwen3-4b-regen}"
HOST="${EVAL_HOST:-http://127.0.0.1}"
PORT="${EVAL_PORT:-8000}"
MODEL="${EVAL_MODEL:-qwen3-4b}"

NUM_QUESTIONS="${EVAL_NUM_QUESTIONS:-256}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
TOP_P="${EVAL_TOP_P:-1.0}"
TOP_K="${EVAL_TOP_K:--1}"
MIN_P="${EVAL_MIN_P:-0}"
# SAMPLE_SEED selects dataset rows; SEED controls generation sampling.
SAMPLE_SEED="${EVAL_SAMPLE_SEED:-42}"
SEED="${EVAL_SEED:-42}"
MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-32}"
REQUEST_TIMEOUT_SECONDS="${EVAL_REQUEST_TIMEOUT_SECONDS:-600}"
METRICS_WAIT_SECONDS="${EVAL_METRICS_WAIT_SECONDS:-10}"
CHAT_TEMPLATE_KWARGS="${EVAL_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$CHAT_TEMPLATE_KWARGS" ]]; then
  CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
fi
SKIP_ACCEPTANCE="${EVAL_SKIP_ACCEPTANCE:-false}"

# Paths and output location.
DATA_PATH="${EVAL_DATA_PATH:-$HOME/datasets/open-perfectblend-qwen3-4b-regen/data/train.jsonl}"
SPEC_DIR="${SPECFORGE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$SPEC_DIR/.venv/bin/python}"
OUT_ROOT="${EVAL_OUTPUT_DIR:-$HOME/logs/qwen3-4b/eval/regen}"
# ==================== Evaluation workflow ====================
EVAL_SCRIPT="$SPEC_DIR/scripts/eval_regen_vllm.py"

if [[ $# -gt 1 || ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Usage: bash $0 [run-name: letters, numbers, dots, underscores, hyphens]" >&2
  exit 2
fi
if [[ "$SKIP_ACCEPTANCE" != true && "$SKIP_ACCEPTANCE" != false ]]; then
  echo "EVAL_SKIP_ACCEPTANCE must be true or false" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$EVAL_SCRIPT" || ! -r "$DATA_PATH" ]]; then
  echo "Check interpreter (requires aiohttp), evaluator and dataset paths:" >&2
  printf '%s\n' "$PYTHON_BIN" "$EVAL_SCRIPT" "$DATA_PATH" >&2
  exit 1
fi

EVAL_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL"
  --data-path "$DATA_PATH"
  --num-questions "$NUM_QUESTIONS"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --sample-seed "$SAMPLE_SEED"
  --seed "$SEED"
  --max-concurrency "$MAX_CONCURRENCY"
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
  --metrics-wait-seconds "$METRICS_WAIT_SECONDS"
  --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"
)
if [[ "$SKIP_ACCEPTANCE" == true ]]; then
  EVAL_ARGS+=(--skip-acceptance)
fi

mkdir -p "$OUT_ROOT"
RUN_DIR="$(mktemp -d "$OUT_ROOT/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
printf 'Results directory: %s\n' "$RUN_DIR"

# The Python evaluator snapshots counters before and after generation itself.
"$PYTHON_BIN" "$EVAL_SCRIPT" "${EVAL_ARGS[@]}" \
  --output-dir "$RUN_DIR/results" 2>&1 | tee "$RUN_DIR/regen.log"

printf '\nCompleted. Results and logs: %s\n' "$RUN_DIR"
