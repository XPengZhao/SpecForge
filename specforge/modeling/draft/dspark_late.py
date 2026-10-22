"""Dense 4+1 DSpark: pure-MASK preparation, then parallel late conditioning.

Positions denote decoder slots: the output at position p predicts token p+1.
A bank for a verification starting at t contains slots t+2 .. t+2d+1.
"""
import torch
from transformers import DynamicCache

from .dflash import extract_context_feature


def _stage(model, states, context, positions, cutoffs, keep, layers):
    # Reuse block isolation, but keep context visibility independent of RoPE
    # positions. A later-position MASK must still see only the OLD prefix.
    from specforge.algorithms.common.dflash_family_model import (
        create_dflash_block_mask, create_dflash_sdpa_mask,
    )
    batch, blocks, width, hidden = states.shape
    states = states.reshape(batch, blocks * width, hidden)
    context_positions = torch.arange(context.shape[1], device=states.device).expand(batch, -1)
    pos = torch.cat((context_positions, positions.reshape(batch, -1)), dim=1)
    backend = model.config._attn_implementation
    make_mask = create_dflash_block_mask if backend == 'flex_attention' else create_dflash_sdpa_mask
    mask = make_mask(cutoffs, keep, context.shape[1], width, states.device)
    if backend == 'eager':
        mask = torch.zeros_like(mask, dtype=states.dtype).masked_fill(~mask, torch.finfo(states.dtype).min)
    rotary = model.rotary_emb(states, pos)
    for layer in layers:
        states = layer(hidden_states=states, target_hidden=context,
                       attention_mask=mask, position_embeddings=rotary, use_cache=False)
    return states.reshape(batch, blocks, width, hidden)


def prepare_bank(model, target_hidden, old_cutoffs, bank_starts, keep):
    """Compute 2d states using only context positions < old_cutoffs."""
    d = model.block_size - 1
    context = model.hidden_norm(model.fc(target_hidden))
    positions = bank_starts[..., None] + torch.arange(2*d, device=target_hidden.device)
    states = model.future_mask.view(1, 1, 1, -1).expand(*bank_starts.shape, 2*d, -1)
    return _stage(model, states, context, positions, old_cutoffs, keep, model.layers[:-1])


def finish_block(model, bank, target_hidden, anchor_embeddings, anchors, bank_starts, keep):
    """One parallel final layer over [real bonus, d selected bank states]."""
    d = model.block_size - 1
    offsets = anchors + 1 - bank_starts
    indices = offsets[..., None] + torch.arange(d, device=bank.device)
    selected = bank.gather(2, indices[..., None].expand(*indices.shape, bank.shape[-1]))
    anchor = model.anchor_adapter(anchor_embeddings).unsqueeze(2)
    states = torch.cat((anchor, selected), dim=2)
    positions = anchors[..., None] + torch.arange(d+1, device=bank.device)
    context = model.hidden_norm(model.fc(target_hidden))
    states = _stage(model, states, context, positions, anchors, keep, model.layers[-1:])
    return model.norm(states).flatten(1, 2)


def training_forward(model, target_hidden, anchor_embeddings, anchors, keep, old_cutoffs):
    # Position zero is a bootstrap case without an earlier verification.
    starts = torch.where(anchors == 0, torch.ones_like(anchors), old_cutoffs + 2)
    bank = prepare_bank(model, target_hidden, old_cutoffs, starts, keep)
    return finish_block(model, bank, target_hidden, anchor_embeddings, anchors, starts, keep)


@torch.inference_mode()
def generate(model, target, input_ids, max_new_tokens, stop_token_ids, temperature=0.0,
             return_stats=False):
    """Batch-one greedy reference. No overlap and no draft KV caching.

    Verify with the target cache; retain only accepted-prefix target features.
    Prepare the NEXT bank before verification, even though execution is serial.
    """
    if temperature != 0:
        raise ValueError('4+1 reference generation currently supports greedy decoding only')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('4+1 reference generation requires one nonempty prompt')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    model.eval()
    target.eval()
    d = model.block_size - 1
    tokens = input_ids.clone()
    prompt_length = tokens.shape[1]
    limit = prompt_length + max_new_tokens
    rounds, accepted = 0, 0
    def result():
        if return_stats:
            return dict(output_ids=tokens, rounds=rounds, accepted=accepted,
                        drafted=rounds*d, mal=1+accepted/rounds if rounds else None)
        return tokens
    if max_new_tokens == 0:
        return result()
    cache = DynamicCache()
    out = target(input_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    context = extract_context_feature(out.hidden_states, model.target_layer_ids)
    bonus = out.logits[:, -1].argmax(-1, keepdim=True)
    keep = torch.ones((1, 1), dtype=torch.bool, device=tokens.device)
    anchor = torch.full((1, 1), prompt_length, device=tokens.device, dtype=torch.long)
    # Bootstrap: all prompt features exist before the initial bonus is bound.
    starts = anchor + 1
    bank = prepare_bank(model, context, anchor, starts, keep)
    stops = set(stop_token_ids or [])
    while tokens.shape[1] < limit:
        if int(bonus.item()) in stops or tokens.shape[1] + 1 == limit:
            tokens = torch.cat((tokens, bonus), dim=1)
            break
        hidden = finish_block(model, bank, context, target.get_input_embeddings()(bonus),
                              anchor, starts, keep)
        # Slot p predicts p+1. The last slot is auxiliary training supervision.
        proposal_hidden = hidden[:, :d]
        logits = target.lm_head(proposal_hidden)
        if model.markov_head is None:
            proposals = logits.argmax(-1)
        else:
            proposals, _ = model.markov_head.sample_block_tokens(
                logits, first_prev_token_ids=bonus[:, 0], hidden_states=proposal_hidden,
                temperature=0.0)
        block = torch.cat((bonus, proposals), dim=1)
        # No current-block features enter preparation of the next bank.
        next_starts = anchor + 2
        next_bank = prepare_bank(model, context, anchor, next_starts, keep)
        out = target(block, past_key_values=cache, use_cache=True, output_hidden_states=True)
        posterior = out.logits.argmax(-1)
        r = int((proposals == posterior[:, :d]).long().cumprod(-1).sum().item())
        rounds += 1
        accepted += r
        committed = block[:, :r+1]
        # Stop on EOS inside the accepted prefix; never emit following tokens.
        count = min(committed.shape[1], limit-tokens.shape[1])
        for i in range(count):
            if int(committed[0, i].item()) in stops:
                count = i+1
                break
        tokens = torch.cat((tokens, committed[:, :count]), dim=1)
        if count < r+1 or int(tokens[0, -1].item()) in stops or tokens.shape[1] == limit:
            break
        new_context = extract_context_feature(out.hidden_states, model.target_layer_ids)
        context = torch.cat((context, new_context[:, :r+1]), dim=1)
        if r < d:
            cache.crop(-(d-r))
        bonus = posterior[:, r:r+1]
        anchor = anchor + r + 1
        bank, starts = next_bank, next_starts
    return result()
