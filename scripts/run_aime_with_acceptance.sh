#!/usr/bin/env bash
# Usage: bash scripts/run_aime_with_acceptance.sh [run-name]
# AIME 2024 I + II. Run against a dedicated speculative vLLM service.
set -euo pipefail

name="${1:-qwen3-4b-aime2024}"
if [[ $# -gt 1 || ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Usage: bash $0 [run-name]" >&2
  exit 2
fi
repo="${SPECFORGE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
python="${PYTHON_BIN:-$repo/.venv/bin/python}"
root="${EVAL_OUTPUT_DIR:-$HOME/logs/qwen3-4b/eval/aime}"
host="${EVAL_HOST:-http://127.0.0.1}"
mkdir -p "$root"
out="$(mktemp -d "$root/$name-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
args=(--dataset aime)
if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
  args+=(--data-path "$EVAL_DATA_PATH")
fi
echo "Results directory: $out"
"$python" -u "$repo/scripts/eval_math500_vllm.py" "${args[@]}" \
  --server-url "${EVAL_SERVER_URL:-${host%/}:${EVAL_PORT:-8000}}" --model "${EVAL_MODEL:-qwen3-4b}" \
  --temperature "${EVAL_TEMPERATURE:-0.7}" --top-p "${EVAL_TOP_P:-0.8}" --top-k "${EVAL_TOP_K:-20}" \
  --min-p "${EVAL_MIN_P:-0}" --seed "${EVAL_SEED:-42}" --thinking "${EVAL_THINKING:-off}" \
  --limit "${EVAL_NUM_QUESTIONS:-30}" --max-tokens "${EVAL_MAX_TOKENS:-4096}" \
  --concurrency "${EVAL_MAX_CONCURRENCY:-32}" --metrics-wait-seconds "${EVAL_METRICS_WAIT_SECONDS:-10}" \
  --output "$out/aime.json" 2>&1 | tee "$out/aime.log"
