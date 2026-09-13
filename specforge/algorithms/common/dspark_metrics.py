"""DeepSpec-compatible distribution-overlap metrics for DSpark."""

import torch


@torch.no_grad()
def acceptance_stats(accept_rates, eval_mask):
    # Use token validity, not the position-decay weights used by the loss.
    valid = eval_mask.to(torch.float32)
    rates = accept_rates.detach().float() * valid
    sums = {}
    denoms = {}
    for position in range(rates.shape[-1]):
        name = f"accept_rate@{position}"
        sums[name] = rates[..., position].sum()
        denoms[name] = valid[..., position].sum()
    valid_blocks = eval_mask.any(dim=-1).to(torch.float32)
    tau = 1.0 + rates.cumprod(dim=-1).sum(dim=-1)
    sums["tau_probabilistic"] = (tau * valid_blocks).sum()
    denoms["tau_probabilistic"] = valid_blocks.sum()
    return sums, denoms


def tau_loss_terms(accept_rates, eval_mask):
    """Differentiable normalized prefix deficit, averaged per valid block.

    A block with m valid prefix positions contributes (m - sum(prefix_prob))/m.
    Empty blocks contribute neither numerator nor denominator; no position decay.
    """
    valid = eval_mask.bool().to(torch.float32).cumprod(dim=-1)
    rates = accept_rates.float() * valid
    lengths = valid.sum(dim=-1)
    deficits = ((1.0 - rates.cumprod(dim=-1)) * valid).sum(dim=-1)
    per_block = deficits / lengths.clamp_min(1.0)
    return per_block.sum(), (lengths > 0).float().sum()
