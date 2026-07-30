# DeepSeek-V4 DSpark offline training

The offline path stores target features once and trains only the three DSpark
layers. The trainer additionally loads the frozen target embedding and LM head;
it does not construct the 43-layer target decoder.

## Prepare target features

Request logs whose `prompt` contains multiple preformatted DeepSeek turns can
be expanded into one record per complete assistant turn:

```bash
python scripts/convert_deepseek_request_logs.py \
  --input-path /path/to/request-logs.jsonl \
  --output-path /path/to/dspark-preformatted.jsonl
```

For each non-empty assistant response ending with
`<｜end▁of▁sentence｜>`, the converter writes a `{"text": ...}` record containing
the full conversation prefix through that response. Prefixes repeated in later
request snapshots are deduplicated by SHA-256 and records retain their
first-seen order.

### Generate target rollouts

When the recorded assistant responses were produced by a different model or
agent policy, regenerate them with the exact target model used for DSpark
serving. Start a target-only vLLM server without `--speculative-config`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /path/to/DeepSeek-V4-Flash-DSpark \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name DeepSeek-V4-Flash-DSpark \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-num-seqs 2 \
  --max-model-len 128000 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8
```

Generate one target response for every preformatted prompt:

```bash
uv run --active --no-sync python scripts/generate_target_rollouts.py \
  --server-url http://127.0.0.1:8000 \
  --model DeepSeek-V4-Flash-DSpark \
  --tokenizer-path /path/to/DeepSeek-V4-Flash-DSpark \
  --data-path /path/to/dspark-preformatted.jsonl \
  --output-path /path/to/dspark-target-rollouts.jsonl \
  --max-length 128000 \
  --max-tokens 2048 \
  --temperature 0 \
  --top-p 1.0 \
  --concurrency 2 \
  --trust-remote-code
```

The output remains preformatted JSONL and can be passed directly to feature
capture. If generation is interrupted, rerun with `--resume`; completed source
line numbers are skipped. Responses stopped by the model receive the DeepSeek
end marker. Length-truncated responses remain valid partial target trajectories
and are retained unless `--drop-truncated` is set. Client concurrency should
not exceed the vLLM server's `--max-num-seqs`; start at two for long prompts
and increase both values only after checking KV-cache headroom.

### Collect Draft-OPD traces

Draft-OPD collection must use a vLLM server running the current DSpark draft.
The patched verification path records each proposed block, its accepted prefix,
and the target log probability of every proposed token. Add
`--collect-spec-decode-trace` when generating rollouts:

```bash
uv run --active --no-sync python scripts/generate_target_rollouts.py \
  --server-url http://127.0.0.1:8000 \
  --model DeepSeek-V4-Flash-DSpark \
  --tokenizer-path /path/to/DeepSeek-V4-Flash-DSpark \
  --data-path /path/to/dspark-preformatted.jsonl \
  --output-path /path/to/dspark-opd-rollouts.jsonl \
  --max-length 128000 \
  --max-tokens 2048 \
  --temperature 1.0 \
  --top-p 1.0 \
  --concurrency 2 \
  --collect-spec-decode-trace \
  --trust-remote-code
```

Use a new output path when enabling trace collection; `--resume` treats an
existing source line as complete even if it was generated without a trace.
The vLLM hidden-state dump script copies the trace into each feature file as
`opd_anchor_positions`, `opd_draft_token_ids`, `opd_target_logprobs`,
`opd_accepted_lengths`, and `opd_candidate_mask`.

```bash
python /path/to/vllm/examples/features/speculative_decoding/dspark_dump_hidden_states_server.py \
  --server-url http://127.0.0.1:8000 \
  --model DeepSeek-V4-Flash-DSpark \
  --tokenizer-path /path/to/DeepSeek-V4-Flash-DSpark \
  --data-path /path/to/dspark-opd-rollouts.jsonl \
  --output-path /path/to/dspark-opd-hidden-states \
  --max-length 128000 \
  --response-context-tokens 128 \
  --trust-remote-code
```

Use this server-backed dump path for OPD data; the generic
`prepare_hidden_states.py` path does not propagate rollout trace metadata.

### Capture target features

Pass the target-rollout file with `--is-preformatted` and the DeepSeek
template:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path /path/to/DeepSeek-V4-Flash-DSpark \
  --draft-model-config configs/deepseek-v4-flash-dspark.json \
  --data-path /path/to/dspark-target-rollouts.jsonl \
  --output-path /path/to/dspark-hidden-states \
  --is-preformatted \
  --chat-template deepseek-v3 \
  --max-length 32768 \
  --response-only \
  --response-context-tokens 128 \
  --sglang-context-length 32768 \
  --tp-size 8 \
  --batch-size 1
```

The configured target layers are `40`, `41`, and `42`. Each record contains
`input_ids`, `loss_mask`, their concatenated `aux_hidden_state`, and the target
`hidden_state` before the LM head.

With `--response-only`, the target still processes the complete prompt and
response. Only the last supervised response and the configured number of
preceding target tokens are written. The retained context has `loss_mask=0`;
only response positions can be sampled as training anchors. If the formatted
conversation exceeds `--max-length`, preprocessing drops tokens from the left
so the final supervised response remains in the target input. Use a new output
directory when changing this setting because existing feature files are skipped.
The current offline capture path runs one full prefill per sample. Very long
DeepSeek-V4 inputs can hit SGLang CSA prefill planner limits before feature
capture; use 32k as the smoke/default setting unless the capture path is updated
to reuse SGLang's scheduler-backed chunked prefill.

## Train

An 8-GPU smoke configuration is available at
`examples/configs/deepseek-v4-flash-dspark-offline-smoke.yaml`. Override its
model and hidden-state paths for the local environment.

Use the following fields in the training YAML:

```yaml
model:
  target_model_path: /path/to/DeepSeek-V4-Flash-DSpark
  draft_model_config: configs/deepseek-v4-flash-dspark.json
  draft_checkpoint_path: /path/to/DeepSeek-V4-Flash-DSpark
  torch_dtype: bfloat16

data:
  hidden_states_path: /path/to/dspark-hidden-states
  max_length: 4096

training:
  strategy: dspark
  attention_backend: flex_attention
  batch_size: 1
  num_anchors: 32
  dspark_ce_loss_alpha: 0.1
  dspark_l1_loss_alpha: 0.9
  dspark_confidence_head_alpha: 1.0
  dspark_opd_loss_alpha: 0.0

deployment:
  mode: local_colocated
  trainer:
    nnodes: 1
    nproc_per_node: 8
```

Set `dspark_opd_loss_alpha` to a positive value only for feature files carrying
the five `opd_*` tensors. Verified response tokens use a local Bernoulli
forward KL, while rejected draft suffix tokens use the sampled k3 reverse-KL
estimator. A rejected speculative step contributes its accepted prefix and
Target recovery token to the response stream; later response tokens belong to
subsequent trace anchors and are not counted twice:

```yaml
training:
  dspark_opd_loss_alpha: 1.0
  dspark_opd_forward_weight: 1.0
  dspark_opd_rejected_weight: 1.0
  dspark_opd_rejected_position_decay: 0.8
  dspark_opd_logprob_min_clamp: -80.0
```

The existing CE, distribution L1, and confidence losses remain independently
controlled by their existing alpha values. Set those alphas to zero for a pure
OPD run, or keep them positive for an SFT-plus-OPD run.

Run the 8-GPU smoke training from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run --active --no-sync specforge train \
  --config examples/configs/deepseek-v4-flash-dspark-offline-smoke.yaml \
  model.target_model_path=/path/to/DeepSeek-V4-Flash-DSpark \
  model.draft_checkpoint_path=/path/to/DeepSeek-V4-Flash-DSpark \
  data.hidden_states_path=/path/to/dspark-hidden-states \
  2>&1 | tee dspark-response-offline-train.log
```

For 4-GPU training, restrict the visible devices and override the trainer
process count:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run --active --no-sync specforge train \
  --config examples/configs/deepseek-v4-flash-dspark-offline-smoke.yaml \
  model.target_model_path=/path/to/DeepSeek-V4-Flash-DSpark \
  model.draft_checkpoint_path=/path/to/DeepSeek-V4-Flash-DSpark \
  data.hidden_states_path=/path/to/dspark-hidden-states \
  deployment.trainer.nproc_per_node=4 \
  2>&1 | tee dspark-response-offline-train-4gpu.log
```

The DSv4 loader reads only `mtp.*` tensors from the full HF checkpoint. Packed
FP4 expert tensors and FP8 tensors are dequantized to the configured training
dtype while loading. The checkpoint, tokenizer, captured layer IDs, and hidden
state convention must come from the same model release.

## Export for vLLM

Training checkpoints are not full serving model directories. Merge the trained
draft weights back into a full DeepSeek-V4-Flash-DSpark HF directory before
measuring vLLM acceptance:

```bash
uv run --active --no-sync python scripts/export_deepseek_v4_dspark_checkpoint.py \
  --base-hf-model /public/llm_models/DeepSeek/DeepSeek-V4-Flash-DSpark \
  --checkpoint ./outputs/dsv4-dspark-response-offline-smoke/dsv4-dspark-response-offline-smoke-latest \
  --output-dir ./exports/deepseek-v4-flash-dspark-trained-draft \
  --overwrite
```

The exporter keeps target weights unchanged. By default, each trained tensor is
converted back to the storage format used by the corresponding base tensor:
MXFP4 expert weights are repacked with UE8M0 scales, FP8 weights receive
regenerated block scales, and unquantized tensors retain their base dtype.

Shards unaffected by `mtp.*` are hardlinked from the base model. Shards that
contain replaced `mtp.*` tensors are rewritten so every checkpoint key occurs
only once; this avoids depending on safetensors file iteration order in serving
loaders. Pass `--copy-base-weights` if all unaffected shards must also be
physically copied. For a draft-only training artifact, use
`--draft-format floating --dtype keep`; this preserves the checkpoint's BF16
weights and strict FP32 mHC, attention-sink, router-bias, and RMSNorm tensors.
