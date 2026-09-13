# GQA DSpark with a differentiable prefix objective

Based on dspark-v1.3: GQA, bidirectional blocks, response-only supervision,
block size 7, and position decay gamma 4. Existing DeepSpec caches are reused.
`training.dspark_tau_loss_alpha` defaults to 0 (baseline); the experiment YAML
sets it to 0.1. This is a heuristic starting point, not a tuned optimum.

For each block's m valid prefix positions, compute distribution overlap
`a_i = clamp(1 - 0.5 * ||softmax(draft_i)-softmax(target_i)||_1, 0, 1)` and
`L_tau_block = sum_i(1 - product_{j<=i} a_j) / m`.
Average over nonempty blocks. For a full block this is `(8-tau)/7`; truncated
blocks use their actual valid prefix length, not 7. Empty blocks are excluded.
No position decay is applied inside this objective. The existing CE, L1 and
confidence objectives retain their previous decay and normalization.

The total objective is `original_loss + alpha * L_tau`. The differentiable
path reuses per-position L1 computations (including gradient checkpointing).
Distributed gradients use the all-rank valid-block denominator for tau,
separately from the original token-weight denominator. As before, gradient
accumulation averages micro-batch objectives, not one pooled denominator over
an entire optimizer step. Logging pools counts over the logging window:
`train/loss` remains the rank-0 mean of micro-batch objectives;
`train/loss_weighted` is the globally pooled weighted objective.

New tags: `train/tau_loss`, `train/tau_loss_weighted`. The existing detached
`train/tau_probabilistic` remains available. For varying block lengths,
`tau_loss` is not recoverable from mean tau with a fixed `(8-tau)/7` formula.
The tau coefficient and objective version are included in the resume contract.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3-4b-dspark-tau-offline.yaml \
  training.max_steps=2 training.log_interval=1 training.save_interval=2 \
  tracking.report_to=tensorboard \
  run_id=qwen3-4b-gqa-tau-smoke \
  output_dir=outputs/qwen3-4b-gqa-tau-smoke
```

For the one-epoch experiment use max_steps=2616 and a fresh run/output name;
omit the log/save overrides to retain defaults 10/2616. Keep total_steps=26160
so the LR schedule matches baseline. Start from scratch rather than resuming
another objective. For the later no-decay experiment set
`training.loss_decay_gamma=null`, using a separate run.

The overlap product is a teacher-forced surrogate, not exact online MAL or
hard greedy prefix accuracy. Evaluate actual accepted lengths at identical
checkpoint steps; raw total losses are not comparable across alpha settings.
GPU memory can increase because the overlap path now retains gradients.
CPU tests cover the formula, derivative, truncation, empty blocks, actual loss
integration and checkpoint recomputation; distributed denominator behavior is
checked with a simulated collective. GPU/Flex and real multi-GPU short runs
remain necessary before a long experiment.
