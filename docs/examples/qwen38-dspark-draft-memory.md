# Qwen3.8 DSpark draft-state memory experiment

This opt-in experiment reuses the previous draft's final normalized hidden
states. It is different from target-state Carryover and does not run a target
backbone during training. Existing DeepSpec caches are reused unchanged.

## Training transition

1. Sample anchors using the baseline response mask. Run the draft without memory
   under `no_grad`.
2. Compute Markov argmax using recorded previous tokens, in bounded vocabulary
   chunks. Before the first mismatch these previous tokens equal greedy draft
   tokens, so the first reference mismatch is the same as sequential greedy
   drafting. Predictions after the mismatch are not used.
3. If `m` of `B` proposals match, shift anchor `a` to `a + m + 1`.
   Drop rows `0..m` (accepted rows plus the first rejected row). If all `B`
   match, use the bonus position as the new anchor, with no memory. Incomplete response
   tails with no mismatch and transitions lacking another supervised token are
   masked out; do not resample them differently on different ranks.
4. The second forward receives cached correct prefix features strictly before
   the new anchor, its recorded token embedding, and detached remaining draft
   states. Only this forward contributes gradients. Original DSpark losses and
   teacher-forced Markov supervision apply at the shifted positions.

This is **reference-rejection simulation**, not true stochastic rejection
sampling. It coincides with greedy target verification only when the reference
tokens agree with that target's greedy continuation under matching conditions.
The first pass has no memory; deployed later rounds may have memory. This
remaining multi-round distribution difference must be evaluated in serving.

## Memory interface and positions

For hidden row `j`, the old query position is `a+j` but its predicted token
position is `a+j+1`. We store the latter as memory RoPE position. After dropping
the first rejection, retained rows start at `new_anchor+1`.

`memory = draft_memory_norm(draft_memory_proj(detach(old_final_hidden)))`.
The shared adapter maps draft width to draft width. Each decoder layer uses its
existing K/V projections on this same memory and reads it in its original
attention call. There is no second attention or token/status embedding.
KV layout during parallel training is `[target context | per-anchor memory |
draft queries]`. A query sees only its own memory and draft block, plus the
correct prefix; other anchors and correct future target states are masked.

The architecture is explicitly recorded in `dflash_config.draft_memory` and
`draft_memory_version` in the saved draft config. Baseline configs/weights do
not gain adapter parameters. Memory changes the attention softmax; zero-valued
memory is not equivalent to disabling it. Use a false memory mask to disable.

## Run on the server

From `/public/workspace/dspark/SpecForge`, with the training environment active:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3.8-flash-next-dspark-draft-memory-offline.yaml \
  training.max_steps=2 training.num_anchors=8 training.accumulation_steps=1 \
  run_id=qwen38-draft-memory-smoke output_dir=outputs/qwen38-draft-memory-smoke
```

The YAML trains from scratch, matching the previous teacher-forced carryover
run. For an optional weights-only warm start, set `model.draft_checkpoint_path`
explicitly. All baseline tensors must match. Only a completely absent memory
adapter may be initialized; partial adapter checkpoints and missing baseline
weights fail. Do not use `resume_from` for a baseline-to-memory architecture change.

After the smoke run, launch the YAML without overrides for 2604 training steps
on a 26040-step learning-rate schedule (4 GPUs, microbatch 4, accumulation 32,
global batch 512, 512 anchors, LR 3e-4, warmup 4%, save every 651 steps).
These common hyperparameters match the previous teacher-forced carryover run;
the two-pass objective and per-step compute still differ.

```bash
mkdir -p /public/workspace/dspark/logs/train-qwen38-flash-next
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3.8-flash-next-dspark-draft-memory-offline.yaml \
  2>&1 | tee /public/workspace/dspark/logs/train-qwen38-flash-next/qwen38-draft-memory.log
```

This stops at step 2604. To train the full schedule, add
`training.max_steps=26040` before the log redirection.

Logged diagnostics: `memory_reference_accepted`, `memory_rows`,
`memory_active_fraction`, `memory_transition_fraction`. These describe local
training transitions, not measured serving acceptance. Keep baseline training
comparisons at matched budgets and measure actual multi-round acceptance and
latency before scaling up.

`dspark_draft_memory_head_chunk_size` bounds temporary first-pass vocabulary
logits. Smaller chunks trade memory for more launches. Whole-memory dropout is
optional via `dspark_draft_memory_keep_prob`; default 1 retains every eligible
suffix. Evaluation disables this random dropout.

## Export / serving boundary

Use `configs/qwen3.8-flash-next-dspark-draft-memory.json` when exporting; the original
baseline JSON does not describe the adapter. The normal strict exporter rejects
that mismatch. This branch implements training and the model memory interface.
The current vLLM target-carry implementation does not implement draft memory:
it must save previous draft output rows per request, align by accepted counts,
and project/write memory into every draft layer using prediction positions.
Do not benchmark this checkpoint as an ordinary DSpark checkpoint and assume
memory is active. CUDA/FlexAttention throughput and real multi-round serving
need validation on the server.
