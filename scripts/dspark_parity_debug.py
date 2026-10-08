"""Opt-in first-prefill DSpark dump helper installed into vLLM by the installer.

This module is called AFTER propose returns, outside CUDA capture. It only
supports eager diagnostic serving. No weights, token choices, or KV are changed.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def enabled():
    return bool(os.environ.get("DSPARK_PARITY_DIR"))


def propose_with_dump(speculator, parent, args, kwargs):
    if not enabled():
        return parent(*args, **kwargs)
    bound = inspect.signature(parent).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    diagnostic = not values.get("dummy_run") and not values.get("is_profile")
    if diagnostic and not speculator.vllm_config.model_config.enforce_eager:
        raise RuntimeError("DSPARK_PARITY_DIR requires --enforce-eager")
    result = parent(*args, **kwargs)
    if not diagnostic:
        return result
    from vllm.distributed import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank() != 0:
        return result
    import torch
    batch = values["input_batch"]
    aux = values.get("aux_hidden_states")
    hidden = getattr(speculator, "_dspark_parity_hidden", None)
    if not aux or hidden is None:
        raise RuntimeError("Parity hook did not observe aux/head hidden states")
    offsets = batch.query_start_loc[:batch.num_reqs + 1].cpu().tolist()
    output = Path(os.environ["DSPARK_PARITY_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    for i in range(batch.num_reqs):
        start, stop = offsets[i:i + 2]
        positions = batch.positions[start:stop].cpu().tolist()
        anchor = int(speculator.input_buffers.positions[i * speculator.num_query_per_req])
        # Only a complete first prefill: no partial prefix-cache hits, decode,
        # chunked prefills, rejected rows, or padded rows can enter the comparison.
        if anchor <= 0 or positions != list(range(anchor)):
            continue
        prefix = batch.input_ids[start:stop].cpu().tolist()
        digest = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()[:20]
        indices = speculator.sample_indices[
            i * speculator.num_speculative_steps:(i + 1) * speculator.num_speculative_steps
        ].long()
        payload = {
            "schema": 1, "prefix_token_ids": prefix, "anchor_position": anchor,
            "anchor_token_id": int(speculator.input_buffers.input_ids[i * speculator.num_query_per_req]),
            "query_positions": speculator.input_buffers.positions[
                i * speculator.num_query_per_req:(i + 1) * speculator.num_query_per_req
            ].cpu().clone(),
            "aux_hidden": torch.cat([a[start:stop] for a in aux], -1).cpu().clone(),
            "draft_hidden": hidden.index_select(0, indices).cpu().clone(),
            "draft_token_ids": result[i].cpu().tolist(),
            "draft_model": str(speculator.draft_model_config.model),
            "sample_from_anchor": speculator.sample_from_anchor,
            "eager": True,
        }
        destination = output / f"prefix-{digest}-{time.time_ns()}-{os.getpid()}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
    return result
