#!/usr/bin/env bash
# Usage: bash scripts/run_humaneval_with_acceptance.sh [run-name]
# Draft sampling comes from the running service, not this client.
set -euo pipefail

RUN_NAME="${1:-qwen3-4b-markov-mla}"
if [[ $# -gt 1 || ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Usage: bash $0 [run-name: letters, numbers, dots, underscores, hyphens]" >&2
  exit 2
fi
SPEC_DIR="${SPECFORGE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$SPEC_DIR/.venv/bin/python}"
OUT_ROOT="${EVAL_OUTPUT_DIR:-$HOME/logs/qwen3-4b/eval/humaneval}"

HOST="${EVAL_HOST:-http://127.0.0.1}"
PORT="${EVAL_PORT:-8000}"
MODEL="${EVAL_MODEL:-qwen3-4b}"
NUM_QUESTIONS="${EVAL_NUM_QUESTIONS:-164}"
MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-32}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
TOP_P="${EVAL_TOP_P:-0.8}"
TOP_K="${EVAL_TOP_K:-20}"
MIN_P="${EVAL_MIN_P:-0}"
SEED="${EVAL_SEED:-42}"
TIMEOUT="${EVAL_REQUEST_TIMEOUT_SECONDS:-600}"
METRICS_WAIT="${EVAL_METRICS_WAIT_SECONDS:-10}"
CHAT_TEMPLATE_KWARGS="${EVAL_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$CHAT_TEMPLATE_KWARGS" ]]; then
  CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Set PYTHON_BIN to an interpreter with aiohttp and evalplus installed." >&2
  exit 1
fi
ARGS=(
  --host "$HOST" --port "$PORT" --model "$MODEL"
  --num-questions "$NUM_QUESTIONS" --max-concurrency "$MAX_CONCURRENCY"
  --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE"
  --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" --seed "$SEED"
  --request-timeout-seconds "$TIMEOUT" --metrics-wait-seconds "$METRICS_WAIT"
  --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"
)
if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
  ARGS+=(--data-path "$EVAL_DATA_PATH")
fi
if [[ "${EVAL_SKIP_ACCEPTANCE:-false}" == true ]]; then
  ARGS+=(--skip-acceptance)
fi

mkdir -p "$OUT_ROOT"
RUN_DIR="$(mktemp -d "$OUT_ROOT/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PYTHON_BIN" "$SPEC_DIR/scripts/eval_humaneval_vllm.py" \
  "${ARGS[@]}" --output-dir "$RUN_DIR/results" 2>&1 | tee "$RUN_DIR/humaneval.log"

printf '\nGeneration finished. Grade separately in an isolated evaluator:\n'
printf 'evalplus.evaluate --dataset humaneval --samples %q\n' "$RUN_DIR/results/samples.jsonl"
