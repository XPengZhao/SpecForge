# Qwen3-4B DSpark shared + sparse MoE

This experiment replaces each of the five draft FFNs with a shared FFN of
intermediate width 4864 and four routed FFNs of width 4864. One routed expert is
active per speculative block and per layer. The dense recipe remains unchanged.

## Training

Use the existing online launcher with the new draft configuration:

```bash
ATTEMPT="dspark-moe-e4-smoke-$(date +%Y%m%d-%H%M%S)" \
  bash scripts/run_qwen3_4b_dspark_online.sh smoke \
  model.draft_model_config=configs/qwen3-4b-dspark-moe.json
```

After the smoke run succeeds:

```bash
ATTEMPT="dspark-moe-e4-$(date +%Y%m%d-%H%M%S)" \
  bash scripts/run_qwen3_4b_dspark_online.sh train \
  model.draft_model_config=configs/qwen3-4b-dspark-moe.json
```

The launcher retains its existing GPU 0 capture / GPUs 1–4 training layout,
128 accumulation steps, learning rate, and dataset settings. Compare dense and
MoE with the same launcher overrides. This starts a new experiment; do not use a
dense checkpoint as a MoE resume checkpoint. Same-architecture MoE checkpoints
use the normal save/resume path, including router and expert weights.
Offline training can use the same `model.draft_model_config` override. Target
capture/cache features do not change.

## Routing and objective

`dflash_config.moe` in the JSON configuration controls:

- `enabled`: opt-in; absent/false preserves the dense architecture.
- `num_experts`: four routed experts; the shared branch is separate.
- `shared_intermediate_size` / `expert_intermediate_size`: sum must equal the
  original `intermediate_size` to preserve the active FFN width.
- `balance_loss_alpha`: 0.01 initially; a hyperparameter, not a proven optimum.

Each layer has its own router. Its input is the projected, normalized target
feature immediately **before** the block anchor, never a whole-sequence pool.
An anchor at position zero uses a zero context. Padding/invalid anchors do not
participate in expert dispatch or balance statistics. All positions in a block
share its expert and gate. Router scores and softmax are computed in FP32.

The output is `shared(x) + p_selected * expert_selected(x)`, where `p_selected`
is taken from the softmax over **all** experts, without renormalizing Top-1 to
one. The same gate is used during inference. Normal inference skips balance loss and
logging statistics; group lengths needed for dispatch are still computed.
Explicit `return_router_stats=True` retains diagnostics, including in evaluation.

For each layer, balance loss is `E * sum(f_e * P_e)`, where `f_e` is the fraction
of valid blocks assigned to an expert and `P_e` is its average softmax
probability. Losses are averaged over layers and added to the existing DSpark
objective. Empty microbatches contribute zero auxiliary loss. Counts are local
to each rank/microbatch; there is no new per-microbatch collective. The existing
DDP and gradient accumulation scaling applies to the combined objective.

All experts remain replicated on every training GPU. Experts receive only their
selected blocks; empty expert operations retain zero-gradient autograd edges so
DDP can safely accumulate and reduce gradients even when ranks route differently.
No capacity cap or token/block dropping is used. The dispatcher stably sorts blocks
by expert and transfers a fixed-size vector of group lengths to the host once per
layer, replacing the previous per-expert dynamic-index synchronizations. This
preserves block order within each expert and does not evaluate unselected
experts on real blocks. One host synchronization per layer remains; this is not
a fused grouped-GEMM serving kernel.

## Metrics and interpretation

Existing `loss`, `loss_weighted`, CE/L1/confidence and acceptance statistics keep
their task-only meanings for comparison with dense runs. Additional metrics:

- `moe_balance_loss`: unweighted auxiliary loss, averaged over layers and then
  over logged microbatches/ranks.
- `moe_layer{L}_expert{E}_fraction`: expert usage over valid blocks.
- `moe_layer{L}_expert{E}_probability`: mean router probability over valid blocks.

These are collected by the existing logging window; logging adds no new
per-microbatch scalar extraction. Watch long-term and per-domain routing, not
just whether each individual sequence uses all experts equally. A strong local
balance loss can inhibit domain specialization.

## Scope and validation

The PyTorch draft forward supports both training blocks and inference context
chunks with absolute positions. Save/load round trips preserve routing. CPU
regressions cover E=1 dense FFN splitting, Top-1 gated output and router gradients,
masking/causal context, auxiliary loss integration, non-reentrant checkpointing,
and two-rank DDP accumulation with different active experts.

This does **not** implement the new architecture in vLLM/SGLang serving draft
kernels. Existing dense DSpark serving loaders must not be used for this
checkpoint without MoE support. The SGLang **target capture server** is unchanged.
GPU memory use, throughput, and online acceptance length still require server
measurements. Active FFN arithmetic is similar to dense, but optimizer memory,
gradient communication and dispatch costs increase.
