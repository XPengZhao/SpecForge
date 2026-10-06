#!/usr/bin/env bash
# Usage: bash run_math500_with_acceptance.sh [setting=2] [thinking=off] [run-name]
# Draft sampling follows the running vLLM service's configuration.
set -euo pipefail

SETTING="${1:-2}"
THINKING="${2:-off}"
case "$SETTING" in
  1) TEMPERATURE=0; TOP_P=1.0; TOP_K=-1 ;;
  2) TEMPERATURE=0.7; TOP_P=0.8; TOP_K=20 ;;
  3) TEMPERATURE=1.0; TOP_P=0.95; TOP_K=20 ;;
  *) echo "Setting 必须是 1、2 或 3" >&2; exit 2 ;;
esac
case "$THINKING" in
  off) MODE=nonthinking ;;
  on) MODE=thinking ;;
  *) echo "Thinking 必须是 on 或 off" >&2; exit 2 ;;
esac
NAME="${3:-qwen38-math500-$MODE-setting$SETTING}"
if [[ $# -gt 3 || ! "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "用法：bash $0 [setting: 1/2/3] [thinking: on/off] [运行名称]" >&2
  exit 2
fi

ROOT="${DSPARK_ROOT:-/public/workspace/dspark}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/vllm-env/.venv-vllm-custom/bin/python}"
EVALUATOR="${MATH500_EVAL_SCRIPT:-$ROOT/SpecForge/scripts/eval_math500_vllm.py}"
SERVER_URL="${EVAL_SERVER_URL:-http://127.0.0.1:8000}"
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
echo "Target: Setting ${SETTING}；thinking=${THINKING}；draft 采样沿用服务设置"

"$PYTHON_BIN" -u "$EVALUATOR" \
  --server-url "$SERVER_URL" \
  --model "$MODEL" \
  --limit 500 \
  --thinking "$THINKING" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p 0 \
  --seed 42 \
  --max-tokens "${EVAL_MAX_TOKENS:-4096}" \
  --concurrency "${EVAL_CONCURRENCY:-256}" \
  --metrics-wait-seconds 10 \
  --output "$OUT/math500.json" \
  2>&1 | tee "$OUT/math500.log"

echo "完成，结果、原始回复和日志保存在：$OUT"
