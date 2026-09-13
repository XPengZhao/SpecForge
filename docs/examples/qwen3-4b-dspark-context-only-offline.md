# GQA block context-only ablation

This experiment starts from the dspark-v1.3 Qwen3-4B GQA baseline, with no MLA
or sliding window. Set `dspark_block_attention` in the draft JSON to
`context_only`; omission or `bidirectional` keeps the original behavior.

Both SDPA and Flex masks preserve the original historical-context visibility.
Only draft K/V visibility changes from the same block to no draft positions. Invalid blocks and cross-block isolation are unchanged. All layers
use this mask. No parameters are added; the Markov/confidence heads, anchor
sampling, loss weights, and response-only supervision stay unchanged. This
ablates attention communication, not every possible dependency between outputs
(the Markov head still operates as before).

The existing DeepSpec v2 cache can be reused without modification. Start from
scratch with the same seed/schedule as GQA; do not resume the bidirectional run.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m specforge.cli train \
  -c examples/configs/qwen3-4b-dspark-context-only-offline.yaml \
  training.max_steps=2 training.log_interval=1 training.save_interval=2 \
  tracking.report_to=tensorboard \
  run_id=qwen3-4b-gqa-context-only-smoke \
  output_dir=outputs/qwen3-4b-gqa-context-only-smoke
```

For one epoch, change max_steps to 2616, log_interval to 10, save_interval to
2616, and use a fresh run_id/output_dir. Keep total_steps=26160 to match the
baseline learning-rate schedule. Compare positional acceptance and
`tau_probabilistic` as well as loss. Offline evaluation uses the same context-only
mask. Checkpoint config retains the switch. For actual inference MAL, the
serving implementation must use the same mask; this change does not provide
an external serving adapter. The generic unmasked generation path raises an
error for context-only checkpoints instead of silently evaluating bidirectionally.
