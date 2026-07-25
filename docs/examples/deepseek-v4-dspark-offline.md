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
first-seen order. Pass the converted file with `--is-preformatted` and the
DeepSeek template:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path /path/to/DeepSeek-V4-Flash-DSpark \
  --draft-model-config configs/deepseek-v4-flash-dspark.json \
  --data-path /path/to/dspark-preformatted.jsonl \
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

deployment:
  mode: local_colocated
  trainer:
    nnodes: 1
    nproc_per_node: 8
```

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
physically copied. For debugging only,
`--draft-format floating --dtype bfloat16` retains the older floating-point
overlay behavior.
