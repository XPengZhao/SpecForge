# Qwen3-4B DSpark MLA ablation

This branch adds `draft_attention_type: mla` to the draft JSON. Omission keeps
GQA, including its existing parameter names and initialization path. Use
`configs/qwen3-4b-dspark-mla.json` or the standalone launch configuration below.
The target remains Qwen3-4B and the existing DeepSpec v2 hidden-state cache is reused.

The attention uses joint KV down-projection, latent RMSNorm, per-head content
K/V up-projection, and a shared rotary key with per-head rotary queries. Q uses
a direct projection (no query bottleneck). Unlike the Qwen GQA baseline, it does
not apply per-head Q/K RMSNorm; this is a structural MLA ablation, not an isolated
rank-only experiment. The design follows the KV compression and decoupled RoPE
construction in [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

| Draft setting | GQA baseline | MLA experiment |
|---|---:|---:|
| Query heads | 32 | 32 |
| KV heads | 8 | 32 expanded heads |
| QK width per head | 128, all rotary | 64 content + 64 rotary |
| Value width per head | 128 | 128 |
| KV latent width | none | 512 |
| Layers / block size / anchor budget | 5 / 7 / 512 | 5 / 7 / 512 |

MLA fields are `mla_kv_lora_rank`, `mla_qk_nope_head_dim`,
`mla_qk_rope_head_dim`, and `mla_v_head_dim`. The rotary width must be even;
this implementation requires V width to equal total QK width for backend
compatibility. The original `num_key_value_heads=8` field is retained for Qwen
config compatibility but does not control MLA. These settings are serialized
with the model configuration.

## Run

Start a new run from scratch, using the same response-only supervision and
training schedule as the GQA baseline. Do not resume a GQA checkpoint as MLA.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m specforge.cli train \
  -c examples/configs/qwen3-4b-dspark-mla-offline.yaml \
  data.dspark_supervision=response \
  training.max_steps=2 \
  training.log_interval=1 \
  training.save_interval=2 \
  tracking.report_to=tensorboard \
  run_id=qwen3-4b-mla-smoke \
  output_dir=outputs/qwen3-4b-mla-smoke
```

For a one-epoch comparison, use a fresh run/output name and change `max_steps`
to 2616, `log_interval` to 10 and `save_interval` to 2616. Keep `total_steps=26160`
so the learning-rate schedule matches the baseline. Logs remain in
`<output_dir>/runs`.

## Interpretation and limitations

Training and the generic Transformers cache path materialize expanded K/V.
This is not compressed-latent cache serving and cannot establish MLA KV-memory
or inference-speed gains. Training memory may increase relative to GQA because
all 32 heads have expanded K/V. Use the GPU short run to measure memory before
starting the full experiment. No custom serving-engine MLA adapter is included.

Keep dataset, batch, steps, loss coefficients, and response-only evaluation
identical. Parameter counts and initialization differ across attention types;
report them alongside loss, positional acceptance, and inference MAL. Equal
seeds do not imply identical initialization for the shared layers after attention
modules consume different amounts of randomness.

## MLA + SWA 128

Use `examples/configs/qwen3-4b-dspark-mla-swa128-offline.yaml`. Its draft JSON
sets `dspark_context_window=128`; all five layers use the existing DSpark
block mask with history positions `[max(0, anchor-127), anchor)`. All tokens in
the same draft block remain mutually visible, and other draft blocks remain
invisible. This is a block-anchored context-window ablation, not per-query
causal sliding attention. To retain 128 strictly historical tokens, use 129.

Do not enable Qwen's `layer_types=sliding_attention` or backend
`sliding_window` here: the packed training layout does not use contiguous
logical query positions. The explicit SDPA/Flex block mask enforces the window.
Only draft attention is windowed; cached target features still encode the
original full-context target forward. Anchor sampling and the loss are unchanged.

Start a fresh run with the same schedule as the MLA full-context baseline:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m specforge.cli train \
  -c examples/configs/qwen3-4b-dspark-mla-swa128-offline.yaml \
  training.max_steps=2 training.log_interval=1 training.save_interval=2 \
  tracking.report_to=tensorboard \
  run_id=qwen3-4b-mla-swa128-smoke \
  output_dir=outputs/qwen3-4b-mla-swa128-smoke
```

For one epoch, start a fresh run/output and set max_steps=2616,
log_interval=10, save_interval=2616, retaining total_steps=26160.
The new field is persisted in checkpoints. Offline evaluation uses the same
window mask. Generic unmasked generation now raises for windowed drafts;
inference must supply an equivalent mask before comparing MAL. This training
change does not implement cache eviction or a windowed serving adapter.
