#!/usr/bin/env bash
# Usage: bash scripts/run_math500_with_acceptance.sh [run-name]
# Legacy: [setting: 1/2/3] [thinking: on/off] [run-name]. EVAL_* overrides the preset.
# Draft sampling follows the running vLLM service's configuration.
set -euo pipefail

SETTING=2
THINKING=off
case "${1:-}" in
  1|2|3)
    SETTING="$1"
    shift
    if (( $# )); then
      THINKING="$1"
      shift
    fi
    ;;
esac
case "$SETTING" in
  1) TEMPERATURE=0; TOP_P=1.0; TOP_K=-1 ;;
  2) TEMPERATURE=0.7; TOP_P=0.8; TOP_K=20 ;;
  3) TEMPERATURE=1.0; TOP_P=0.95; TOP_K=20 ;;
esac
TEMPERATURE="${EVAL_TEMPERATURE:-$TEMPERATURE}"
TOP_P="${EVAL_TOP_P:-$TOP_P}"
TOP_K="${EVAL_TOP_K:-$TOP_K}"
case "$THINKING" in
  off) MODE=nonthinking ;;
  on) MODE=thinking ;;
  *) echo "Thinking 必须是 on 或 off" >&2; exit 2 ;;
esac
NAME="${1:-qwen38-math500-$MODE-setting$SETTING}"
if [[ $# -gt 1 || ! "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Usage: bash $0 [run-name]" >&2
  exit 2
fi

ROOT="${DSPARK_ROOT:-/public/workspace/dspark}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/vllm-env/.venv-vllm-custom/bin/python}"
SPEC_DIR="${SPECFORGE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
EVALUATOR="${MATH500_EVAL_SCRIPT:-$SPEC_DIR/scripts/eval_math500_vllm.py}"
HOST="${EVAL_HOST:-http://127.0.0.1}"
SERVER_URL="${EVAL_SERVER_URL:-${HOST%/}:${EVAL_PORT:-8000}}"
MODEL="${EVAL_MODEL:-qwen38-flash-next}"
OUT_ROOT="${EVAL_OUTPUT_DIR:-$ROOT/logs/eval-qwen38-flash-next/math500}"

if [[ ! -x "$PYTHON_BIN" || ! -f "$EVALUATOR" ]]; then
  echo "请检查 Python 环境和评测脚本路径：" >&2
  printf '%s\n' "$PYTHON_BIN" "$EVALUATOR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
OUT="$(mktemp -d "$OUT_ROOT/$NAME-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
echo "本次结果目录：$OUT"
echo "Target: temperature=$TEMPERATURE top_p=$TOP_P top_k=${TOP_K}；thinking=${THINKING}；draft 采样沿用服务设置"

"$PYTHON_BIN" -u "$EVALUATOR" \
  --server-url "$SERVER_URL" \
  --model "$MODEL" \
  --limit "${EVAL_NUM_QUESTIONS:-500}" \
  --thinking "$THINKING" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p "${EVAL_MIN_P:-0}" \
  --seed "${EVAL_SEED:-42}" \
  --max-tokens "${EVAL_MAX_TOKENS:-4096}" \
  --concurrency "${EVAL_MAX_CONCURRENCY:-${EVAL_CONCURRENCY:-256}}" \
  --metrics-wait-seconds "${EVAL_METRICS_WAIT_SECONDS:-10}" \
  --output "$OUT/math500.json" \
  2>&1 | tee "$OUT/math500.log"

echo "完成，结果、原始回复和日志保存在：$OUT"
