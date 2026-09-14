# DSpark GQA all-mask input ablation

From dspark-v1.3, retaining GQA, bidirectional attention, full historical
context strictly before the anchor, response-only supervision, and original
loss/position decay. No MLA, SWA, context-only or tau-loss changes are included.

The draft JSON option `dspark_anchor_token_input` defaults to true. Set false
to replace `[anchor, MASK x 6]` with `[MASK x 7]` at the backbone input only.
Position IDs remain a through a+6; target labels remain a+1 through a+7.
The original input_ids are never overwritten. Training Markov/confidence
inputs still use `[anchor, true first response token, ...]` (teacher forcing).
This is not an ablation of all anchor information in the full model.

The flag is serialized in the draft configuration and recorded in the resume
contract when false. Existing hidden-state caches are reused unchanged.
The generic local generation input constructor also respects the flag without
changing block_output_ids used for verification. External serving backends must
apply the same input rule and the DSpark Markov sampling procedure before
comparing actual MAL; full end-to-end serving parity is not validated here.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3-4b-dspark-all-mask-offline.yaml \
  training.max_steps=2 training.log_interval=1 training.save_interval=2 \
  tracking.report_to=tensorboard \
  run_id=qwen3-4b-gqa-all-mask-smoke \
  output_dir=outputs/qwen3-4b-gqa-all-mask-smoke
```

After the smoke test, use max_steps=2616 and a fresh run/output. Omit log/save
overrides to use 10/2616. Keep total_steps=26160 to match the baseline LR
schedule. Start from scratch, not from another ablation's checkpoint.
