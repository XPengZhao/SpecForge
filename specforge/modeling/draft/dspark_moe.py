"""Block-routed FFN and Q/O experts for the Qwen DSpark backbone."""

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
        gate, indices_by_expert, stats = _route_blocks(
            routing_context, block_keep_mask, self.router,
            return_router_stats=return_router_stats,
        )
        flat = hidden_states.reshape(-1, self.block_size, width)
        routed = torch.zeros_like(flat)
        for expert, indices in zip(self.experts, indices_by_expert):
            selected = flat.index_select(0, indices)
            values = expert(selected)
            weights = gate.reshape(-1).index_select(0, indices).to(values.dtype)
            routed = routed.index_add(0, indices, values * weights[:, None, None])
        output = self.shared(hidden_states) + routed.reshape_as(hidden_states)
        return output, stats


def _route_blocks(context, keep_mask, router, *, return_router_stats):
    """One FP32 Top-1 decision and one grouped dispatch plan per block.

    Invalid blocks sort into a trailing sentinel group. Empty groups remain in
    the plan so all experts get autograd edges under DDP/no_sync accumulation.
    """
    num_experts = router.out_features
    with torch.autocast(device_type=context.device.type, enabled=False):
        probs = F.linear(context.float(), router.weight.float()).softmax(-1)
    ids = probs.argmax(-1)
    gate = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    groups = ids.reshape(-1).masked_fill(~keep_mask.reshape(-1).bool(), num_experts)
    order = groups.argsort(stable=True)
    group_counts = torch.zeros(
        num_experts + 1, dtype=torch.long, device=groups.device
    ).scatter_add_(0, groups, torch.ones_like(groups))
    # One fixed-size host transfer, shared by Q and O when used for attention.
    sizes = group_counts[:num_experts].cpu().tolist()
    indices = order[:sum(sizes)].split(sizes)
    stats = None
    if return_router_stats:
        valid = keep_mask.to(probs.dtype)
        counts = group_counts[:num_experts].to(probs.dtype)
        count = counts.sum()
        probability_sums = (probs * valid.unsqueeze(-1)).sum((0, 1))
        fractions = counts / count.clamp_min(1)
        balance_loss = (
            num_experts
            * (fractions.detach() * probability_sums / count.clamp_min(1)).sum()
        )
        stats = dict(
            balance_loss=balance_loss, counts=counts.detach(),
            probability_sums=probability_sums.detach(), count=count.detach(),
        )
    return gate, indices, stats


class BlockRoutedQO(nn.Module):
    """Paired Top-1 Q/O projections; K/V and the attention kernel stay shared."""

    def __init__(self, config):
        super().__init__()
        options = config.dflash_config["attention_moe"]
        self.num_experts = int(options.get("num_experts", 4))
        self.block_size = int(config.block_size)
        if self.num_experts < 1 or self.block_size < 1:
            raise ValueError("attention MoE expert count and block size must be positive")
        self.hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.query_width = config.num_attention_heads * head_dim
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.q_experts = nn.ModuleList(
            nn.Linear(self.hidden_size, self.query_width, bias=config.attention_bias)
            for _ in range(self.num_experts)
        )
        self.o_experts = nn.ModuleList(
            nn.Linear(self.query_width, self.hidden_size, bias=config.attention_bias)
            for _ in range(self.num_experts)
        )

    def _project(self, x, experts, indices_by_expert, output_width, gate=None):
        batch, length, width = x.shape
        flat = x.reshape(-1, self.block_size, width)
        output = x.new_zeros((flat.shape[0], self.block_size, output_width))
        for expert, indices in zip(experts, indices_by_expert):
            values = expert(flat.index_select(0, indices))
            if gate is not None:
                weights = gate.reshape(-1).index_select(0, indices).to(values.dtype)
                values = values * weights[:, None, None]
            output = output.index_add(0, indices, values)
        return output.reshape(batch, length, output_width)

    def forward(self, hidden_states, routing_context, block_keep_mask, *, return_router_stats):
        batch, length, width = hidden_states.shape
        if length % self.block_size:
            raise ValueError("attention MoE requires complete draft blocks")
        blocks = length // self.block_size
        if routing_context is None or routing_context.shape != (batch, blocks, width):
            raise ValueError("attention routing requires one context vector per block")
        if block_keep_mask is None or block_keep_mask.shape != (batch, blocks):
            raise ValueError("attention routing requires one mask entry per block")
        route = _route_blocks(routing_context, block_keep_mask, self.router,
                              return_router_stats=return_router_stats)
        q = self._project(hidden_states, self.q_experts, route[1], self.query_width)
        return q, route

    def project_output(self, attention_output, route):
        gate, indices, _stats = route
        # Gate exactly once, AFTER attention and O; never scale normalized Q.
        return self._project(attention_output, self.o_experts, indices,
                             self.hidden_size, gate=gate)
