"""Block-routed shared + sparse FFN for the Qwen DSpark backbone."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class SharedSparseMLP(nn.Module):
    """Top-1 routing with the selected probability from the full softmax.

    Experts run on selected blocks only. Empty expert calls deliberately remain
    in the autograd graph: DDP must reduce zero gradients for experts unused on
    this rank, including at the end of a no_sync accumulation window.
    """

    def __init__(self, config, make_mlp):
        super().__init__()
        options = config.dflash_config["moe"]
        self.num_experts = int(options.get("num_experts", 4))
        self.block_size = int(config.block_size)
        shared_width = int(
            options.get("shared_intermediate_size", config.intermediate_size // 2)
        )
        expert_width = int(
            options.get("expert_intermediate_size", config.intermediate_size // 2)
        )
        if self.num_experts < 1 or min(shared_width, expert_width, self.block_size) < 1:
            raise ValueError("MoE expert count, widths and block size must be positive")
        if shared_width + expert_width != config.intermediate_size:
            raise ValueError(
                "shared + routed widths must equal dense intermediate_size"
            )
        shared_config, expert_config = deepcopy(config), deepcopy(config)
        shared_config.intermediate_size = shared_width
        expert_config.intermediate_size = expert_width
        self.shared = make_mlp(shared_config)
        self.experts = nn.ModuleList(
            make_mlp(expert_config) for _ in range(self.num_experts)
        )
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(
        self,
        hidden_states,
        routing_context,
        block_keep_mask,
        *,
        return_router_stats=True,
    ):
        batch, length, width = hidden_states.shape
        if length % self.block_size:
            raise ValueError("MoE query length must contain complete draft blocks")
        blocks = length // self.block_size
        if routing_context.shape != (batch, blocks, width):
            raise ValueError("routing context must have one vector per draft block")
        if block_keep_mask.shape != (batch, blocks):
            raise ValueError("routing mask must have one entry per draft block")
        # Keep router probabilities stable even when FFNs use BF16.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            probs = F.linear(
                routing_context.float(), self.router.weight.float()
            ).softmax(-1)
        ids = probs.argmax(-1)
        gate = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        flat = hidden_states.reshape(-1, self.block_size, width)
        keep = block_keep_mask.reshape(-1).bool()
        flat_ids = ids.reshape(-1)
        # Invalid blocks sort into a trailing sentinel group. Stable sorting
        # preserves the original order within each expert's GEMM input.
        groups = flat_ids.masked_fill(~keep, self.num_experts)
        order = groups.argsort(stable=True)
        group_counts = torch.zeros(
            self.num_experts + 1, dtype=torch.long, device=groups.device
        ).scatter_add_(0, groups, torch.ones_like(groups))
        # One fixed-size host transfer per layer, rather than one dynamic
        # nonzero/where synchronization per expert. Do not skip empty groups:
        # their zero gradients are needed by DDP accumulation.
        sizes = group_counts[: self.num_experts].cpu().tolist()
        routed = torch.zeros_like(flat)
        offset = 0
        for expert_id, expert in enumerate(self.experts):
            size = sizes[expert_id]
            indices = order[offset : offset + size]
            offset += size
            selected = flat.index_select(0, indices)
            values = expert(selected)
            weights = gate.reshape(-1).index_select(0, indices).to(values.dtype)
            routed = routed.index_add(0, indices, values * weights[:, None, None])
        output = self.shared(hidden_states) + routed.reshape_as(hidden_states)
        if not return_router_stats:
            return output, None
        valid = block_keep_mask.to(probs.dtype)
        counts = group_counts[: self.num_experts].to(probs.dtype)
        count = counts.sum()
        probability_sums = (probs * valid.unsqueeze(-1)).sum((0, 1))
        fractions = counts / count.clamp_min(1)
        balance_loss = (
            self.num_experts
            * (fractions.detach() * probability_sums / count.clamp_min(1)).sum()
        )
        return output, {
            "balance_loss": balance_loss,
            "counts": counts.detach(),
            "probability_sums": probability_sums.detach(),
            "count": count.detach(),
        }
