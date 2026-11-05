"""Causal token-history attention with a normalized Markov residual."""

from copy import deepcopy

import torch
from torch import nn

from .markov_gqa import MarkovGQAContext, MarkovGQAHead


class BiasedRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, states):
        values = states.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return values.to(states.dtype) * self.weight + self.bias


class MarkovTokenAttentionHead(MarkovGQAHead):
    """W2(RMSNorm(W1(token) + O(attention))), with token-derived KV."""

    markov_head_type = "token_attention"

    def __init__(self, config, dspark_config, kernels):
        local_config = deepcopy(config)
        local_config.hidden_size = int(dspark_config["markov_rank"])
        options = dict(dspark_config)
        options["markov_gqa"] = dspark_config.get("markov_token_attention", {})
        super().__init__(local_config, options, kernels)
        self.output_norm = BiasedRMSNorm(self.markov_rank, config.rms_norm_eps)

    def context_limits(self, query_position_ids, anchor_positions, steps):
        # Query p consumes token p and predicts p+1: token p itself is visible.
        return query_position_ids.flatten(1) + 1

    def prepare_context(self, token_ids, context_position_ids=None, context_valid_mask=None):
        if token_ids.ndim != 2:
            raise ValueError("Token history must have shape [batch, sequence]")
        return super().prepare_context(
            self.get_prev_embeddings(token_ids), context_position_ids, context_valid_mask
        )

    def compute_latents(self, token_ids, **kwargs):
        attention_output = super().compute_latents(token_ids, **kwargs)
        return self.output_norm(self.get_prev_embeddings(token_ids) + attention_output)

    def sample_block_tokens(
        self, base_logits, *, first_prev_token_ids, context, query_position_ids,
        anchor_positions, temperature=0.0, attention_backend=None, **kwargs,
    ):
        """Decode one block per sample; context contains only pre-anchor tokens.

        Append the known input token before each query. The local context is
        deliberately not persisted: callers must discard rejected suffix tokens.
        """
        from .dspark import _sample

        if base_logits.ndim != 3:
            raise ValueError("Sequential token attention expects [batch, steps, vocab]")
        prev = first_prev_token_ids
        tokens, corrected = [], []
        for step in range(base_logits.size(1)):
            positions = query_position_ids[:, step:step + 1]
            new = self.prepare_context(prev[:, None], positions)
            context = MarkovGQAContext(
                torch.cat((context.key, new.key), dim=-2),
                torch.cat((context.value, new.value), dim=-2),
                torch.cat((context.positions, new.positions), dim=-1),
                torch.cat((context.valid, new.valid), dim=-1),
            )
            logits = self.apply_block_logits(
                base_logits[:, None, step:step + 1],
                token_ids=prev[:, None, None], context=context,
                query_position_ids=positions[:, None],
                anchor_positions=anchor_positions.reshape(-1, 1),
                attention_backend=attention_backend,
            )[:, 0]
            prev = _sample(logits, temperature)[:, 0]
            tokens.append(prev)
            corrected.append(logits)
        if not tokens:
            return first_prev_token_ids[:, None][:, :0], base_logits
        return torch.stack(tokens, dim=1), torch.cat(corrected, dim=1)
