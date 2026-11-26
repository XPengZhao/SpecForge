#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_DATA_PATH:?Set EVAL_DATA_PATH to the actual training subset JSONL}"
: "${TRACE_PREFIX:?Set TRACE_PREFIX equal to server VLLM_NGRAM_TRACE_PREFIX}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${1:-ngram-trace-$(date +%Y%m%d-%H%M%S)}"
exec "$PYTHON_BIN" scripts/eval_ngram_trace_vllm.py \
  --trace-prefix "$TRACE_PREFIX" \
  --host "${EVAL_HOST:-http://127.0.0.1}" \
  --port "${EVAL_PORT:-8000}" \
  --model "${EVAL_MODEL:-qwen3-4b}" \
  --data-path "$EVAL_DATA_PATH" \
  --num-questions "${EVAL_NUM_QUESTIONS:-512}" \
  --max-concurrency "${EVAL_MAX_CONCURRENCY:-32}" \
  --sample-seed "${EVAL_SAMPLE_SEED:-42}" \
  --seed "${EVAL_SEED:-42}" \
  --max-tokens "${EVAL_MAX_TOKENS:-4096}" \
  --temperature "${EVAL_TEMPERATURE:-0}" \
  --top-p "${EVAL_TOP_P:-1}" \
  --top-k "${EVAL_TOP_K:--1}" \
  --output-dir "${EVAL_OUTPUT_DIR:-outputs/eval/$RUN_NAME}"
