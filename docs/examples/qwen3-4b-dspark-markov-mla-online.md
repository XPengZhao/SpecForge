# Qwen3-4B backbone MLA + Markov MLA online training

This experiment branches from `dspark-v1.5`. The target remains frozen Qwen3-4B;
SGLang captures the existing auxiliary layers `[1, 9, 17, 25, 33]` and final
hidden state. The Mooncake producer/capture contract is unchanged.

## Architecture

The existing `hidden_norm(fc(concat(target_features)))` produces context `C`
of shape `[batch, sequence, 2560]` once per training forward. Both the backbone
and head consume it, and both contribute gradients to that fusion projection.
All five backbone attention layers use MLA with their own weights. Existing
backbone residuals and SwiGLU FFNs are retained.

The head is exactly:

```text
previous token -> W1 embedding (256)
               -> MLA cross-attention (Q: 256, context KV: 2560, output: 256)
               -> W2 vocabulary projection
               -> add to backbone logits
```

There is no head residual, FFN, query norm, or output norm. Its KV branch has a
trainable 512-dimensional RMSNorm between down- and up-projection. The fused
input `C` is shared, but none of the head's attention weights are shared with
any backbone layer. The confidence head keeps its original backbone hidden +
W1 embedding inputs.

Both backbone and head use 32 query heads, 64 non-rotary + 64 rotary Q/K
channels, 128 V channels per head, and KV rank 512. Head settings are independent
under `dflash_config.markov_mla`. The head retains one KV entry per context
token; rank 512 compresses channels, not sequence length.

Parameter counts (trainable, excluding the frozen target embeddings/LM head):

| Component | Parameters |
| --- | ---: |
| Backbone MLA + vanilla Markov baseline | 612,109,569 |
| Additional Markov MLA, including KV RMSNorm | 6,717,952 |
| Total | 618,827,521 |

The additional BF16 weights occupy 13.44 MB; all trainable BF16 weights occupy
1.238 GB. Gradients, optimizer states, activations and the teacher are additional.

## Visibility and positions

For anchor `a`, every head query in that block sees only target positions `< a`.
It never sees the block's true future target features or other Markov queries as
KV. Query RoPE positions are `a + step`, matching the backbone training slots;
keys retain their original context positions. Backbone visibility remains
pre-anchor context plus its own draft block. Invalid blocks and empty visible
contexts produce zero head correction. Explicit context padding masks are
supported by the head API; the online training input is right-padded and valid
anchors precede the padding.

Training uses teacher-forced previous tokens. The head's sampling method uses
its own generated previous token and reuses prepared context KV across steps.
The generic `spec_generate` reference path also applies this head and retains
its independent context KV between blocks. This path is for reference evaluation;
production serving engines need their own adapter.

## Run

Use the dependencies and patched SGLang installation described in
[qwen3-4b-dspark-online.md](qwen3-4b-dspark-online.md).

```bash
export SPECFORGE_PATH=/path/to/SpecForge
export TARGET_MODEL_PATH=/data1/models/Qwen3-4B
export DATA_PATH=/path/to/train.jsonl
export OUTPUT_PATH=/path/to/outputs/qwen3-4b

# One optimizer step and a checkpoint; uses GPU 0 for capture, 1-4 for training.
bash scripts/run_qwen3_4b_dspark_markov_mla_online.sh smoke

# Fresh run: 2616 optimizer steps, schedule horizon 26160.
bash scripts/run_qwen3_4b_dspark_markov_mla_online.sh train
```

The script preserves the v1.5 recipe's global batch 512 (4 ranks × batch 1 ×
128 accumulation), learning rate `3e-4`, 512 anchors, block size 7, sequence
length 4096 and CE/L1/confidence losses. It selects the new standalone YAML and
JSON and uses a distinct default attempt name. Extra dotted overrides are
forwarded. The YAML alone retains the original 1+7 topology; the script supplies
the 1+4 / accumulation overrides.

For the backbone-MLA-only control, use the same script and override:

```bash
ATTEMPT=backbone-mla-control-001 \
  bash scripts/run_qwen3_4b_dspark_markov_mla_online.sh smoke \
  model.draft_model_config=configs/qwen3-4b-dspark-backbone-mla.json
```

Start the new architecture from scratch. A GQA or vanilla-head checkpoint is
not a compatible resume checkpoint. For a completed-step checkpoint from this
architecture, use the existing managed-local resume flow with a fresh attempt:

```bash
ATTEMPT=markov-mla-resume-001 \
  bash scripts/run_qwen3_4b_dspark_markov_mla_online.sh train \
  training.resume_from=/path/to/checkpoint
```

Checkpoint state records the MLA dimensions and the head's no-residual,
strict-pre-anchor semantics. The general checkpoint loader still enforces
parameter-key/shape compatibility and the normal data/optimizer resume contract.
See the [disaggregated recovery contract](../basic_usage/disaggregated_training.md#resume-and-freshness)
for queue/recovery constraints. Use the same training data, topology and schedule.
Offline checkpoint evaluation can reuse the existing offline DSpark recipe with
`model.draft_model_config=configs/qwen3-4b-dspark-markov-mla.json`.

## Validation and limits

```bash
python -m pytest tests/test_runtime/test_dspark_markov_mla.py -q
```

Tests cover exact parameter counts, no residual bypass, future/padding/block
masking, empty contexts, sequential/parallel agreement, SDPA/Flex head forward
agreement, end-to-end loss gradients, gradient checkpointing, save/load and
optimizer continuation, reference greedy generation, and independent backbone /
head switches. A CUDA-only test exercises compiled Flex BF16 forward/backward.

The implementation materializes expanded K/V. It does not implement compressed
latent KV serving, vLLM support, or claim an inference-memory/speed improvement.
CPU tests are not a substitute for the GPU smoke run: actual SGLang/Mooncake
integration, compiled CUDA kernels, training throughput and peak memory must be
verified on the training server.

Local checks ran on macOS CPU with both the repository-pinned torch 2.11.0 /
transformers 5.8.1 and an existing torch 2.14.1 / transformers 5.20.2 environment.
Review added incremental-cache versus full-context logits parity across unequal
accepted lengths. It also fixed the inherited DFlash sampler used by the vanilla
Markov control: DSpark reference sampling now applies its head and uses training
slots 0 through block_size-2, rather than skipping the head and using slots 1
through block_size-1. Vanilla, gated and RNN controls have regression coverage.
No real Qwen3-4B training or CUDA throughput/memory measurement was performed locally.
