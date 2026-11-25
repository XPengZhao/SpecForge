# Qwen3-4B DSpark: per-layer context projection

This ablation replaces the shared target-feature projection with five independent
projections. Each draft layer reads all five target layers `[1, 9, 17, 25, 33]`:

```text
F = concat(target features)                        # 12800
C_l = shared_RMSNorm(context_projections[l](F))     # 2560
K_l = layer[l].k_proj(C_l)
V_l = layer[l].v_proj(C_l)
```

The RMSNorm remains shared. GQA, FFNs, the vanilla Markov head, positions, masks,
and losses use the baseline settings. Context K/V shapes are unchanged. The five
projection matrices add 131,072,000 trainable parameters (250 MiB in BF16), for
746,293,249 trainable parameters in this config. Optimizer state and activations
require additional memory beyond the weights.

## Training

Keep the baseline training command and override its draft JSON:

```bash
model.draft_model_config=configs/qwen3-4b-dspark-layerwise-context-proj.json
```

The new JSON sets `dflash_config.context_projection_mode` to `per_layer`.
Omitting the field, or setting it to `shared`, retains the baseline architecture
and checkpoint keys. Use a new run ID and output directory for this experiment.

Fresh training initializes one projection and copies it into five independent
parameters. With the same seed, the other parameters and initial predictions
match the shared-projection baseline. Each projection then receives its own
gradients. The copies consume no additional random numbers.

To initialize from a trained baseline instead, also pass:

```bash
model.draft_checkpoint_path=/path/to/baseline-step2659
```

This accepts a SpecForge checkpoint directory containing `training_state.pt`,
the state file itself, or an exported HF directory. The baseline `fc.weight` is
copied into `context_projections.0.weight` through
`context_projections.4.weight`. Other weights load normally. This is a new
training run: optimizer, scheduler and counters start fresh. Do not use training
resume to migrate a baseline checkpoint. Resume is supported for checkpoints
already trained with the per-layer architecture.

## HF export

```bash
specforge export --to hf \
  --checkpoint /path/to/layerwise-run-step2659 \
  --draft-config configs/qwen3-4b-dspark-layerwise-context-proj.json \
  --output-dir /path/to/exported-layerwise-draft
```

The export preserves the projection mode and all five matrices. Export does not
perform baseline migration: a shared checkpoint with the per-layer JSON, a
per-layer checkpoint with the shared JSON, or a missing projection is an error.
The exported directory reloads with SpecForge's `AutoDraftModel.from_pretrained`.

vLLM support is a separate follow-up. The existing baseline vLLM branch cannot
load these per-layer weights correctly without that adaptation.
