# Qwen3-4B DSpark Q/O MoE with shared K/V

This experiment adds four paired Q/O experts to each draft attention layer.
Each valid draft block selects one pair with its own attention router. K/V
projections, Q/K normalization, RoPE, the attention mask and KV cache layout
remain shared. There is no shared Q/O branch. The FFN keeps the existing shared
width 4864 plus four routed width-4864 experts, with one active FFN expert.
Attention and FFN routers are independent.

## Training

Use the original training command and change only the draft JSON and attempt
name:

```bash
model.draft_model_config=configs/qwen3-4b-dspark-qo-moe.json
```

The existing launcher also accepts this override:

```bash
ATTEMPT="dspark-qo-moe-e4-smoke-$(date +%Y%m%d-%H%M%S)" \
  bash scripts/run_qwen3_4b_dspark_online.sh smoke \
  model.draft_model_config=configs/qwen3-4b-dspark-qo-moe.json
```

Use `train` instead of `smoke` for the full run, with the same batch size,
accumulation, learning rate, data and deployment overrides as the FFN-only
experiment. Use new output/control/consumer-state paths. Start this architecture
from scratch; a dense or FFN-only checkpoint cannot be used as an exact Q/O MoE
resume. Same-architecture checkpoints use the existing resume path. Target
capture features and offline caches do not change.

## Routing and objective

The new JSON section is independent of `dflash_config.moe`:

```json
"attention_moe": {
  "enabled": true,
  "num_experts": 4,
  "balance_loss_alpha": 0.01
}
```

Absent/disabled attention MoE preserves the FFN-only parameter layout and output.
The existing FFN section is unchanged. Attention-only ablations can disable
`dflash_config.moe.enabled` while leaving `attention_moe.enabled` true.

Routing uses the same causal context construction as the FFN: the projected
and normalized target feature immediately before the block anchor. An anchor
at zero uses a zero context. Every position in the block shares the selected
expert ID and full-softmax probability. Scores/softmax are computed in FP32.
Padding blocks are excluded from dispatch and load-balancing statistics.

For a selected expert `e`, the layer computes:

```text
q = RoPE(shared_q_norm(Q_e(x)))
z = Attention(q, shared_k, shared_v, original_mask)
y = softmax(router(context))[e] * O_e(z)
```

The gate multiplies the output exactly once, after O. It is not applied to Q
or renormalized to one. The surrounding residual connection stays unchanged.
Q blocks are scattered back into their original order before one attention call;
O reuses the same dispatch indices and gate. All experts retain autograd edges,
even for empty groups, so DDP/no_sync accumulation can reduce their zero gradients.
There is no capacity cap, sample dropping, or additional distributed collective.

Each attention layer uses `E * sum(f_e * P_e)` over valid blocks. Attention and
FFN balance losses are averaged over layers separately, then independently
weighted and added to the task loss. Adding attention MoE does not change the
FFN coefficient. Statistics are local to each rank and microbatch; normal
training gradient accumulation and DDP averaging apply to the combined loss.
Normal inference computes no auxiliary statistics and still applies the gate.

Metrics retain `moe_*` for FFN and add `attention_moe_balance_loss` and
`attention_moe_layer<L>_expert<E>_{fraction,probability}`. Task metrics remain
separate from both auxiliary terms.

## Cost and inference scope

At the configured Qwen widths, four Q/O pairs add 314,572,800 projection weights
across five layers relative to one pair. Each block activates just one Q/O pair;
active head count and K/V size do not grow. Runtime overhead includes routing,
gather/scatter and less efficient grouped projections. FLOPs parity does not
promise latency parity. Each attention router transfers one small group-count
vector to the host per layer, reusing it for both Q and O.

Python model forward, HF save/load and the existing KV-cache interface are
covered by tests. A serving engine such as the separate vLLM DSpark branch needs
corresponding Q/O expert loading and execution support before using this new
checkpoint for serving. Exporting weights alone does not add that support.
