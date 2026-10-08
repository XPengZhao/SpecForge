#!/usr/bin/env bash
# Run the existing vLLM GSM8K evaluator and collect acceptance counter deltas.
# Usage: bash scripts/run_gsm8k_with_acceptance.sh [run-name]
# Thinking and draft sampling follow the running vLLM service configuration.
set -euo pipefail

RUN_NAME="${1:-qwen38-baseline-lr3e4-step7812-nonthinking-setting2}"
ROOT="${DSPARK_ROOT:-/public/workspace/dspark}"
VLLM_DIR="${VLLM_DIR:-$ROOT/vllm-env/vllm}"
SPEC_DIR="${SPECFORGE_DIR:-$ROOT/SpecForge}"
PYTHON_BIN="${PYTHON_BIN:-$VLLM_DIR/../.venv-vllm-custom/bin/python}"
HOST="${EVAL_HOST:-http://127.0.0.1}"
PORT="${EVAL_PORT:-8000}"
MODEL="${EVAL_MODEL:-qwen38-flash-next}"
OUT_ROOT="${EVAL_OUTPUT_DIR:-$ROOT/logs/eval-qwen38-flash-next}"
STAT_SCRIPT="$SPEC_DIR/scripts/stat_vllm_acceptance.py"
EVAL_SCRIPT="$VLLM_DIR/tests/evals/gsm8k/gsm8k_eval.py"

if [[ $# -gt 1 || ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Usage: bash $0 [run-name: letters, numbers, dots, underscores, hyphens]" >&2
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

"$PYTHON_BIN" "$STAT_SCRIPT" snapshot \
  --url "${HOST%/}:$PORT" \
  --output "$RUN_DIR/before.json" 2>&1 | tee "$RUN_DIR/before.log"

(
  cd "$VLLM_DIR"
  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --host "$HOST" \
    --port "$PORT" \
    --use-chat-completions \
    --model "$MODEL" \
    --num-shots 5 \
    --max-tokens 4096 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --seed 42 \
    --max-concurrency 256 \
    --save-results "$RUN_DIR/gsm8k.json"
) 2>&1 | tee "$RUN_DIR/gsm8k.log"

echo "Waiting 10 seconds for metrics to update..."
sleep 10

"$PYTHON_BIN" "$STAT_SCRIPT" snapshot \
  --url "${HOST%/}:$PORT" \
  --output "$RUN_DIR/after.json" 2>&1 | tee "$RUN_DIR/after.log"

"$PYTHON_BIN" "$STAT_SCRIPT" diff \
  --before "$RUN_DIR/before.json" \
  --after "$RUN_DIR/after.json" \
  --output "$RUN_DIR/acceptance.json" 2>&1 | tee "$RUN_DIR/acceptance.log"

printf '\nCompleted. Results and logs: %s\n' "$RUN_DIR"
