# Reuse a DeepSpec v2 target cache for DSpark

SpecForge detects `manifest.json` under `data.hidden_states_path` and reads
`samples.idx` plus the original binary shards directly. No DeepSpec Python
installation, full-cache conversion, or shard rewrite is required. The local
reader scans the index and stats shard files at startup; it reads only requested
sample tensors, truncated to `data.max_length`, during training. Metadata refs
still scale with the sample count in the existing offline runtime, but tensors
are never all loaded into RAM.

The cache must use version 2, 56-byte `<QIIQQQQQ` index records, BF16 hidden
states, int32 token IDs and uint8 masks. Index length, sample ids, shard ids,
field bounds, layer order and model dimensions are checked. Training assembly
also requires the target checkpoint identifier to match the manifest's
`target_model_name_or_path`. A local mirror with a different identifier will
fail this check; point training at the capture identifier backed by your local
Hugging Face cache. Identical identifiers alone do not verify checkpoint bytes.

Mapping:

| DeepSpec | SpecForge raw feature |
| --- | --- |
| `input_ids` | `input_ids` (int64) |
| `loss_mask` | `loss_mask` |
| `target_hidden_states` | `aux_hidden_state` |
| `target_last_hidden_states` | `hidden_state` |

The reader adds a scalar bool tensor `loss_mask_is_token_aligned=True`.
The DSpark normalizer uses this marker to preserve the last real token's loss
mask, including after truncation. Legacy SpecForge files without the marker
keep their existing normalization. Normal batching pads with zero masks.
The marker travels with native FeatureStore writes and `.ckpt` dumps, so
reading these back preserves the same supervision. There is no new DeepSpec
binary writer: original shards are always read-only.

## Four-GPU baseline

The supplied recipe uses five draft layers, target layers `[1,9,17,25,33]`,
block size 7, 512 anchors, CE/L1/confidence coefficients `0.1/0.9/1.0`,
loss decay gamma 4, maximum length 4096, and a global batch of
`4 GPUs × 1 sample × 128 accumulation steps = 512`.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 specforge train \
  -c examples/configs/qwen3-4b-dspark-deepspec-offline.yaml \
  data.hidden_states_path=/public/workspace/dspark/cache/deepspec/qwen3_4b_target_cache
```

The recipe fixes the scheduler horizon at 26160 optimizer steps, matching the
reference 10-epoch schedule for 1,339,875 samples. For an initial 2616-update
comparison, override `training.max_steps=2616` while keeping
`training.total_steps=26160`. Recompute these values if the dataset or global
batch changes.

Data reuse does not establish numerical training equivalence. SpecForge carries
accumulation across epochs whereas DeepSpec drops the incomplete optimizer
window each epoch; this recipe does not change that policy. Compare equal
optimizer updates, starting weights, sampled anchors, logits and component
losses before interpreting curve differences. Same seed alone does not guarantee
identical initial weights or sample order. DeepSpec checkpoints are not native
SpecForge resume checkpoints.

Local colocated training is recommended for large caches. The existing
disaggregated offline ingestion path can also read this format and write native
store samples, but that path copies tensors to the destination store and may
require another cache-sized allocation; it is not the zero-conversion path.
