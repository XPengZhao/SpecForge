# Qwen3-4B DSpark: three auxiliary target layers

This ablation selects target layers `[1,17,33]` instead of `[1,9,17,25,33]`.
The draft remains a five-layer GQA model with block size 7, response-only
supervision, and the baseline loss and learning-rate schedule.

Use `examples/configs/qwen3-4b-dspark-aux-3layers-offline.yaml` with the existing
DeepSpec v2 five-layer cache. The local colocated loader reads the original
per-token width, then selects layer groups 0, 2, and 4. Offline evaluation uses
the same selection. Tokens, masks, and final target hidden states are unchanged.
Do not edit the cache manifest or regenerate the cache. Disk reads still include
all five layers; the selected tensors passed to training have width 7680.
The input projection has 19,660,800 parameters, versus 32,768,000 in the baseline.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3-4b-dspark-aux-3layers-offline.yaml \
  tracking.report_to=tensorboard \
  2>&1 | tee /public/workspace/dspark/logs/aux_3layers_train.log
```

The example runs 2616 optimizer steps (approximately one epoch), while retaining
`total_steps=26160` for the reference learning-rate schedule. Micro-batch is 1,
accumulation is 128, and four GPUs give global batch 512. Logging/checkpoint
intervals are 10/2616. Outputs go to `outputs/qwen3-4b-dspark-aux-3layers`, with
TensorBoard events in its `runs` directory when enabled.

Start a fresh model: a five-layer-input checkpoint has a different projection
shape. Resume only a matching three-layer-input experiment. The existing resume
contract records target layer IDs. Compare at equal optimizer steps and global
batch, including position acceptance rates and tau, not only total loss.

This example is for local colocated offline training. A separate disaggregated
cache ingestion pipeline must also select the requested layers before publishing
features; the example does not configure such a pipeline.
