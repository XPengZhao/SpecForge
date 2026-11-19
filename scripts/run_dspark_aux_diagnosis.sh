#!/usr/bin/env bash
set -euo pipefail

# 在服务器上填写这一区。相对路径从 SpecForge 仓库根目录算起。
TARGET="/path/to/Qwen3-4B"
RESPONSES="/path/to/gsm8k/responses.jsonl"
PERFECTBLEND="/path/to/open-perfectblend-qwen3-4b-regen/data/train.jsonl"  # 尽量填实际的 1/10 训练子集
PERFECTBLEND_TEMPLATE="qwen"                       # 原始对话使用训练时的模板
SMALL_CONFIG="/path/to/small-run/config.json"       # 0.131M 训练时的配置
SMALL_CHECKPOINT="/path/to/small-run/step-2659"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/qwen3-4b-dspark.json}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"       # vanilla DSpark 的 step-2659，留空跳过
LARGE_CONFIG="/path/to/large-run/config.json"       # 1.31M 训练时的配置
LARGE_CHECKPOINT="/path/to/large-run/step-2659"
VLLM_EXPORT="/path/to/small-vllm-export"            # vLLM 已能加载的 DSpark 导出目录
# 大表同一步数的 vLLM 推理导出目录；也可以在启动时通过环境变量传入。
LARGE_VLLM_EXPORT="${LARGE_VLLM_EXPORT:-}"
TRAIN_PYTHON="python"                               # 已激活的训练环境，或其 Python 绝对路径
VLLM_PYTHON="/path/to/vllm/.venv/bin/python"
OUTPUT="outputs/dspark-aux-three-stage-2659"

GPU="0"
BATCH_INVARIANT="1"                                 # 与待复现的 vLLM benchmark 一致
SGLANG_ATTENTION_BACKEND="flashinfer"               # 与 online teacher 一致
QUESTIONS=64
ANCHORS=8
MAX_LENGTH=1024

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU" VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"
LOG="${OUTPUT%/}.log"                                 # 放在输出目录外，支持诊断脚本断点续跑

fail() { echo "诊断未完成：$*" >&2; exit 1; }
run_native_logits() {
    [[ -f "$LARGE_VLLM_EXPORT/config.json" && -n "$LARGE_VLLM_EXPORT" ]] || fail "请设置 LARGE_VLLM_EXPORT 为大表的 vLLM 导出目录"
    "$TRAIN_PYTHON" -u scripts/diagnose_dspark_native_logits.py \
        --output "$OUTPUT" --large-export "$LARGE_VLLM_EXPORT" 2>&1 | tee -a "$LOG"
}

run_baseline() {
    [[ -n "$BASELINE_CHECKPOINT" ]] || fail "请填写 BASELINE_CHECKPOINT"
    "$TRAIN_PYTHON" -u scripts/diagnose_dspark_baseline.py --output "$OUTPUT" \
        --config "$BASELINE_CONFIG" --checkpoint "$BASELINE_CHECKPOINT" 2>&1 | tee -a "$LOG"
}

if [[ ${1:-} == --baseline-only && $# == 1 ]]; then
    run_baseline
    exit
elif [[ ${1:-} == --native-only && $# == 1 ]]; then
    [[ -f "$OUTPUT/summary.json" ]] || fail "找不到之前的三组结果 $OUTPUT/summary.json"
    run_native_logits
    exit
elif (( $# )); then
    fail "用法：bash scripts/run_dspark_aux_diagnosis.sh [--native-only | --baseline-only]"
fi
for file in "$PERFECTBLEND" "$RESPONSES" "$SMALL_CONFIG" "$LARGE_CONFIG" "$VLLM_EXPORT/config.json" \
            "$TARGET/config.json" scripts/diagnose_dspark_prefixes.py scripts/dspark_aux_capture.py; do
    [[ -f "$file" ]] || fail "找不到文件 $file"
done
for checkpoint in "$SMALL_CHECKPOINT" "$LARGE_CHECKPOINT"; do
    [[ -f "$checkpoint" || -f "$checkpoint/training_state.pt" || -f "$checkpoint/config.json" ]] \
        || fail "找不到 checkpoint：$checkpoint"
done
command -v "$TRAIN_PYTHON" >/dev/null || fail "找不到训练环境 Python：$TRAIN_PYTHON"
command -v "$VLLM_PYTHON" >/dev/null || fail "找不到 vLLM 环境 Python：$VLLM_PYTHON"
export PATH="$(dirname -- "$(command -v "$TRAIN_PYTHON")"):$PATH"
mkdir -p -- "$(dirname -- "$LOG")"

if ! "$TRAIN_PYTHON" -u scripts/diagnose_dspark_prefixes.py \
    --target "$TARGET" --responses "$RESPONSES" \
    --perfectblend "$PERFECTBLEND" --perfectblend-template "$PERFECTBLEND_TEMPLATE" --expected-step 2659 \
    --draft small "$SMALL_CONFIG" "$SMALL_CHECKPOINT" \
    --draft large "$LARGE_CONFIG" "$LARGE_CHECKPOINT" \
    --compare-aux "$VLLM_EXPORT" --vllm-python "$VLLM_PYTHON" \
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND" \
    --questions "$QUESTIONS" --anchors "$ANCHORS" --max-length "$MAX_LENGTH" \
    --output "$OUTPUT" 2>&1 | tee -a "$LOG"; then
    fail "请查看 $LOG，修复后用相同配置重新运行即可续跑。"
fi

"$TRAIN_PYTHON" - "$OUTPUT/summary.json" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
aux = report["aux_comparison"]
count = aux["paired_questions"]
if not report["complete"] or count == 0 or count != report["planned_questions"]:
    sys.exit(f"诊断未完成，已有 {count} 道配对题，暂不能判断反转。")

scores = aux["comparisons"]["large"]
if scores["reference"] != "small":
    sys.exit("报告的参考模型与预期的小表不一致。")
pb = report["perfectblend"]
stages = [
    ("perfectblend_trainer", "① Open PerfectBlend + trainer aux", pb, "trainer"),
    ("gsm8k_trainer", "② GSM8K + trainer aux", aux, "trainer"),
    ("gsm8k_vllm", "③ GSM8K + vLLM aux", aux, "vllm"),
]
first = report["first_negative_stage"]["large"]
print("\n首位接受率差（大表减小表，单位为百分点）")
for key, label, group, source in stages:
    item = group["comparisons"]["large"][source]
    ci = item["ci95_pp"]
    interval = f"，按题配对 95% 区间 [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    print(f"{label}：{item['delta_pp']:+.3f}{interval}，{group['paired_questions']} 题")
    if key == first:
        verdict = f"本批样本首次观察到大表落后在 {label}。"
        if not ci or ci[1] >= 0:
            verdict += " 此组区间尚不能确认总体也落后。"
print(verdict if first else "本批样本三组均未观察到大表落后。")
interaction = scores["vllm_minus_trainer_interaction"]
print(f"同一批 GSM8K 换 aux 后，大表相对优势变化 {interaction['delta_pp']:+.3f} 个百分点。")
print(f"完整报告  {path.resolve()}")
PY

if [[ -n "$LARGE_VLLM_EXPORT" ]]; then
    run_native_logits
fi

if [[ -n "$BASELINE_CHECKPOINT" ]]; then
    run_baseline
fi
