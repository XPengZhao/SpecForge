#!/usr/bin/env bash
set -euo pipefail

# Paths on the generation server. Rerun with the same input and output to resume.
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/data1/models/Qwen3-4B}"
DATA="${DATA:-$HOME/datasets/open-perfectblend-qwen3-4b-regen-10pct-seed42/data/train.jsonl}"
OUT="${OUT:-$HOME/datasets/open-perfectblend-qwen3-4b-10pct-t1-rollout10}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm/.venv/bin/vllm}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$REPO/.venv/bin/python}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-4096}"  # 8 DP ranks × 512 requests
MAX_LENGTH=32768
MAX_TOKENS=4096
FIRST_ROUND="${FIRST_ROUND:-1}"
ROUNDS="${ROUNDS:-10}"
[[ "$FIRST_ROUND" =~ ^[1-9][0-9]*$ && "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || { echo "FIRST_ROUND and ROUNDS must be positive integers" >&2; exit 2; }

fail() { echo "$*" >&2; exit 1; }
[[ -r "$DATA" && -d "$MODEL" ]] || fail "Check DATA and MODEL at the top of the script."
[[ -x "$TRAIN_PYTHON" && -x "$VLLM_BIN" ]] || fail "Check TRAIN_PYTHON and VLLM_BIN."
mkdir -p "$OUT"
exec 9>>"$OUT/.rollout.lock"
flock -n 9 || fail "Another rollout is writing to $OUT."
url="http://127.0.0.1:$PORT"
if (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    fail "Port $PORT is in use. Stop the previous service or set PORT."
fi
server_log="$OUT/vllm-$(date +%Y%m%d-%H%M%S)-$$.log"
server_pid=
client_pid=
# Give the service and client separate process groups so a shell signal stops both.
set -m
cleanup() {
    trap - EXIT
    trap '' INT TERM HUP
    for pid in "$client_pid" "$server_pid"; do
        if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

echo "Starting Qwen3-4B (DP=8, TP=1). Server log: $server_log"
PATH="$(dirname "$VLLM_BIN"):$PATH" CUDA_VISIBLE_DEVICES="$GPUS" \
    "$VLLM_BIN" serve "$MODEL" --served-model-name qwen3-4b \
    --enable-prefix-caching --data-parallel-size 8 --tensor-parallel-size 1 --dtype bfloat16 \
    --generation-config vllm --max-model-len "$MAX_LENGTH" --max-num-seqs 512 \
    --host 127.0.0.1 --port "$PORT" >"$server_log" 2>&1 9>&- &
server_pid=$!
until curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 30 "$server_log" >&2
        fail "vLLM exited during startup. See $server_log"
    fi
    sleep 2
done

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$TRAIN_PYTHON" \
    "$REPO/scripts/generate_target_rollouts.py" \
    --data-path "$DATA" --output-path "$OUT" --resume \
    --first-round "$FIRST_ROUND" --rounds "$ROUNDS" --seed-offset 42 --dp-ranks 8 \
    --server-url "$url" --model qwen3-4b --tokenizer-path "$MODEL" \
    --temperature 1 --top-p 1 --disable-thinking --end-marker '<|im_end|>' \
    --max-length "$MAX_LENGTH" --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" --log-interval 1000 9>&- &
client_pid=$!
wait "$client_pid"
client_pid=

echo "Completed $ROUNDS verified rounds in $OUT"
