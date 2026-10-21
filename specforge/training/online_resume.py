"""Small, deterministic cursor contract for rebuilt managed online streams."""

import hashlib
import itertools
import json

PLAN_KEY = "online_prompt_plan"
REBUILD_ENV = "SPECFORGE_ONLINE_REBUILD_CHECKPOINT"
PLAN_SUFFIX = ".prompt-plan.json"


def prompt_plan(prompts, *, seed, epochs, dp_size, batch_size, accumulation_steps):
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = json.dumps(
            prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        digest.update(encoded)
        digest.update(b"\n")
    return dict(
        version=1, sha256=digest.hexdigest(), num_prompts=len(prompts),
        seed=seed, epochs=epochs, dp_size=dp_size, batch_size=batch_size,
        accumulation_steps=accumulation_steps,
    )


def read_producer_checkpoint(path):
    """Pin the trainer's checkpoint selection before publishing the stream plan."""
    import os
    import torch
    from specforge.training.checkpoint import CheckpointManager, STATE_FILE

    directory = os.path.realpath(CheckpointManager.resolve_resume_dir(path))
    state = torch.load(os.path.join(directory, STATE_FILE), map_location="cpu", weights_only=False)
    return directory, state


def resume_cursor(state, plan):
    """Online trainer epoch 0 is one concatenated producer stream."""
    for key, value in (
        ("world_size", plan["dp_size"]),
        ("batch_size", plan["batch_size"]),
        ("accumulation_steps", plan["accumulation_steps"]),
    ):
        if state.get(key) != value:
            raise ValueError(f"online resume {key} mismatch: checkpoint={state.get(key)}, current={value}")
    step = state["global_step"]
    if not isinstance(step, int) or step < 0:
        raise ValueError("online resume requires a nonnegative integer global_step")
    total = plan["num_prompts"] * plan["epochs"]
    quantum = plan["dp_size"] * plan["batch_size"] * plan["accumulation_steps"]
    if state.get("epoch") == 1:
        if (state.get("epoch_batch") != 0 or state.get("epoch_samples") != 0
                or step != total // quantum):
            raise ValueError("completed online checkpoint does not match the full prompt plan")
        # The distributor settles a final incomplete optimizer window without
        # training it. A completed stream must not regenerate that tail either.
        return total
    micro = step * plan["accumulation_steps"]
    per_rank = micro * plan["batch_size"]
    if (
        state.get("epoch") != 0
        or state.get("epoch_batch") != micro
        or state.get("epoch_samples") != per_rank
    ):
        raise ValueError("online resume requires a complete optimizer boundary in the concatenated stream (epoch=0)")
    cursor = per_rank * plan["dp_size"]
    if cursor > plan["num_prompts"] * plan["epochs"]:
        raise ValueError("online resume cursor exceeds the rebuilt prompt plan")
    return cursor


def validate_resume_plan(state, plan, prompts, *, legacy_channel=None):
    cursor = resume_cursor(state, plan)
    saved = state.get(PLAN_KEY)
    if saved is not None:
        if saved != plan:
            raise ValueError("online prompt plan changed: data, preprocessing, seed, epochs or topology differ")
        return cursor
    # Legacy checkpoints have no content fingerprint. Never silently infer their
    # consumed order from a step count; require the original publication journal.
    if not legacy_channel:
        raise ValueError(
            "legacy online checkpoint lacks online_prompt_plan; supply "
            "training.resume_ref_channel pointing to the original refs.jsonl, "
            "and retain the original unchanged dataset/preprocessing"
        )
    from specforge.launch import _epoch_online_prompts

    expected = itertools.chain.from_iterable(
        _epoch_online_prompts(prompts, epoch, plan["epochs"], seed=plan["seed"])
        for epoch in range(plan["epochs"])
    )
    with open(legacy_channel, encoding="utf-8") as stream:
        for index, prompt in enumerate(itertools.islice(expected, cursor)):
            line = stream.readline()
            if not line:
                raise ValueError(f"legacy publication journal ends before consumed sample {index}")
            ref = json.loads(line)
            if not prompt.get("task_id") or ref.get("source_task_id") != prompt["task_id"]:
                raise ValueError(f"legacy publication order differs from rebuilt plan at sample {index}; refusing ambiguous replay")
    return cursor
