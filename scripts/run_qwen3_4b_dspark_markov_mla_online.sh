#!/usr/bin/env bash
# GPU 0: Qwen3-4B capture; GPUs 1-4: DSpark, global batch 512.
set -euo pipefail

RUN_MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$RUN_MODE" in
    smoke)
        MAX_PROMPTS=1024
        MAX_STEPS=1
        SAVE_INTERVAL=1
        LOG_INTERVAL=1
        ;;
    train)
        MAX_PROMPTS=0  # The prompt loader treats zero as no limit.
        MAX_STEPS=2616
        SAVE_INTERVAL=2616
        LOG_INTERVAL=10
        ;;
    *)
        printf 'Usage: bash %s [smoke|train] [section.field=value ...]\n' "$0" >&2
        exit 2
        ;;
esac

DATA_PATH="${DATA_PATH:-$HOME/datasets/open-perfectblend-qwen3-4b-regen/data/train.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-$HOME/outputs/qwen3-4b}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/data1/models/Qwen3-4B}"
SPECFORGE_PATH="${SPECFORGE_PATH:-$HOME/SpecForge}"
ATTEMPT="${ATTEMPT:-dspark-markov-mla-${RUN_MODE}-$(date +%Y%m%d-%H%M%S)}"

cd "$SPECFORGE_PATH"
source .venv/bin/activate

exec specforge train \
    -c examples/configs/qwen3-4b-dspark-markov-mla-online.yaml \
    model.target_model_path="$TARGET_MODEL_PATH" \
    data.train_data_path="$DATA_PATH" \
    data.max_prompts="$MAX_PROMPTS" \
    deployment.trainer.nproc_per_node=4 \
    'deployment.disaggregated.managed_local.trainer_cuda_visible_devices=["1","2","3","4"]' \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.learning_rate=3e-4 \
    training.total_steps=26160 \
    training.max_steps="$MAX_STEPS" \
    training.save_interval="$SAVE_INTERVAL" \
    training.log_interval="$LOG_INTERVAL" \
    runtime.in_flight_high_watermark=640 \
    runtime.in_flight_low_watermark=512 \
    deployment.disaggregated.managed_local.mooncake.global_segment_size_bytes=103079215104 \
    run_id="$ATTEMPT" \
    tracking.report_to=tensorboard \
    output_dir="$OUTPUT_PATH/$ATTEMPT" \
    deployment.disaggregated.control_dir="$OUTPUT_PATH/$ATTEMPT/control" \
    deployment.disaggregated.consumer_state_dir="$OUTPUT_PATH/$ATTEMPT/consumer-state" \
    "$@"
