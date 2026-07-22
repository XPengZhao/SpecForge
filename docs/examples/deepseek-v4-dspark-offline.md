# DeepSeek-V4 DSpark offline training

The offline path stores target features once and trains only the three DSpark
layers. The trainer additionally loads the frozen target embedding and LM head;
it does not construct the 43-layer target decoder.

## Prepare target features

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path /path/to/DeepSeek-V4-Flash-DSpark \
  --draft-model-config configs/deepseek-v4-flash-dspark.json \
  --data-path /path/to/train.jsonl \
  --output-path /path/to/dspark-hidden-states \
  --max-length 4096 \
  --tp-size 8 \
  --batch-size 8
```

The configured target layers are `40`, `41`, and `42`. Each record contains
`input_ids`, `loss_mask`, their concatenated `aux_hidden_state`, and the target
`hidden_state` before the LM head.

## Train

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
  num_anchors: 512
  dspark_ce_loss_alpha: 0.1
  dspark_l1_loss_alpha: 0.9
  dspark_confidence_head_alpha: 1.0

deployment:
  mode: local_colocated
  trainer:
    nnodes: 1
    nproc_per_node: 8
```

The DSv4 loader reads only `mtp.*` tensors from the full HF checkpoint. Packed
FP4 expert tensors and FP8 tensors are dequantized to the configured training
dtype while loading. The checkpoint, tokenizer, captured layer IDs, and hidden
state convention must come from the same model release.
