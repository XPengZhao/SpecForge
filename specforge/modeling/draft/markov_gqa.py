"""Context-only GQA between the Markov embedding and vocabulary projection.

The prepared KV is local to a forward/decode block, never stored on the module.
All queries in a block see the same strictly pre-anchor target context.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, rotate_half


@dataclass
class MarkovGQAContext:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True)


class MarkovGQAHead(nn.Module):
    """W1 -> GQA -> W2, with per-head Q/K RMSNorm, no residual or FFN."""

    markov_head_type = "gqa"

    def __init__(self, config, dspark_config, kernels):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.markov_rank = int(dspark_config["markov_rank"])
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
        gqa = dspark_config.get("markov_gqa", {})
        self.num_heads = gqa.get("num_attention_heads", config.num_attention_heads)
        self.num_kv_heads = gqa.get("num_key_value_heads", config.num_key_value_heads)
        self.head_dim = gqa.get("head_dim", config.head_dim)
        for name in ("markov_rank", "num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Markov GQA {name} must be a positive integer")
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError(
                "Markov GQA requires Q heads divisible by KV heads and even head_dim"
            )
        self.q_proj = nn.Linear(
            self.markov_rank, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.markov_rank, bias=False
        )
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        rotary_config = deepcopy(config)
        rotary_config.head_dim = self.head_dim
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)
        self.scaling = self.head_dim**-0.5
        self.backend = getattr(config, "_attn_implementation", None) or "sdpa"

    def get_prev_embeddings(self, token_ids):
        return self.markov_w1(token_ids.long())

    def prepare_context(
        self, target_context, context_position_ids=None, context_valid_mask=None
    ):
        """Project each context token once, independent of the number of anchors."""
        batch, length, _ = target_context.shape
        if context_position_ids is None:
            context_position_ids = torch.arange(length, device=target_context.device)[
                None
            ].expand(batch, -1)
        if context_valid_mask is None:
            context_valid_mask = torch.ones(
                (batch, length), dtype=torch.bool, device=target_context.device
            )
        if context_position_ids.shape != (
            batch,
            length,
        ) or context_valid_mask.shape != (batch, length):
            raise ValueError(
                "Context positions and validity must have shape [batch, context_length]"
            )
        key = self.k_proj(target_context).view(
            batch, length, self.num_kv_heads, self.head_dim
        )
        key = self.k_norm(key).transpose(1, 2)
        value = (
            self.v_proj(target_context)
            .view(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = self.rotary_emb(target_context, context_position_ids)
        key = key * cos[:, None] + rotate_half(key) * sin[:, None]
        return MarkovGQAContext(
            key, value, context_position_ids, context_valid_mask.bool()
        )

    def compute_latents(
        self,
        token_ids,
        *,
        context,
        query_position_ids,
        anchor_positions,
        block_keep_mask=None,
        attention_backend=None,
    ):
        """Token IDs / query positions: [B,N,L]; anchors / keep: [B,N]."""
        if token_ids.ndim != 3 or query_position_ids.shape != token_ids.shape:
            raise ValueError(
                "Markov GQA queries must have shape [batch, blocks, steps]"
            )
        batch, blocks, steps = token_ids.shape
        if anchor_positions.shape != (batch, blocks):
            raise ValueError("anchor_positions must have shape [batch, blocks]")
        if block_keep_mask is None:
            block_keep_mask = torch.ones_like(anchor_positions, dtype=torch.bool)
        if block_keep_mask.shape != anchor_positions.shape:
            raise ValueError("block_keep_mask must match anchor_positions")
        z = self.get_prev_embeddings(token_ids).reshape(
            batch, blocks * steps, self.markov_rank
        )
        q = (
            self.q_proj(z)
            .view(batch, blocks * steps, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.q_norm(q)
        cos, sin = self.rotary_emb(z, query_position_ids.reshape(batch, blocks * steps))
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        anchors = anchor_positions.repeat_interleave(steps, dim=1)
        keep = block_keep_mask.repeat_interleave(steps, dim=1)
        length = context.key.size(-2)
        backend = attention_backend or self.backend
        if backend not in {"eager", "sdpa", "flex_attention"}:
            raise ValueError(
                f"Markov GQA does not support attention backend {backend!r}"
            )
        if length == 0 or blocks * steps == 0:
            # Preserve the graph (including KV parameters) on empty contexts.
            out = q * 0 + (context.key.sum() + context.value.sum()) * 0
        elif backend == "flex_attention":
            from torch.nn.attention.flex_attention import (
                create_block_mask,
                flex_attention,
            )

            def mask_mod(b, h, qi, ki):
                # Flex evaluates the mask on padded tiles as well.
                sq = qi.clamp(max=blocks * steps - 1)
                sk = ki.clamp(max=length - 1)
                return (
                    (qi < blocks * steps)
                    & (ki < length)
                    & keep[b, sq]
                    & context.valid[b, sk]
                    & (context.positions[b, sk] < anchors[b, sq])
                )

            mask = create_block_mask(
                mask_mod,
                B=batch,
                H=None,
                Q_LEN=blocks * steps,
                KV_LEN=length,
                device=q.device,
            )
            fn = _compiled_flex_attention() if q.is_cuda else flex_attention
            out = fn(
                q,
                context.key,
                context.value,
                block_mask=mask,
                scale=self.scaling,
                enable_gqa=True,
            )
        else:
            allowed = context.positions[:, None, :] < anchors[:, :, None]
            allowed = allowed & context.valid[:, None, :] & keep[:, :, None]
            out = F.scaled_dot_product_attention(
                q,
                context.key,
                context.value,
                attn_mask=allowed[:, None],
                scale=self.scaling,
                enable_gqa=True,
            )
        # An all-masked row must contribute zero, including on alternate backends.
        first_valid = (
            context.positions.masked_fill(
                ~context.valid, torch.iinfo(context.positions.dtype).max
            ).amin(dim=-1)
            if length
            else torch.full((batch,), torch.iinfo(torch.long).max, device=q.device)
        )
        visible = keep & (anchors > first_valid[:, None])
        out = torch.where(visible[:, None, :, None], out, 0)
        latent = self.o_proj(
            out.transpose(1, 2).reshape(
                batch, blocks * steps, self.num_heads * self.head_dim
            )
        )
        return latent.reshape(batch, blocks, steps, self.markov_rank)

    def apply_block_logits(
        self, base_logits, *, token_ids, hidden_states=None, **kwargs
    ):
        latent = self.compute_latents(token_ids, **kwargs)
        return base_logits + self.markov_w2(latent)

    def sample_block_tokens(
        self,
        base_logits,
        *,
        first_prev_token_ids,
        context,
        query_position_ids,
        anchor_positions,
        block_keep_mask=None,
        hidden_states=None,
        temperature=0.0,
        attention_backend=None,
    ):
        """Sequential sampling with already prepared KV; supports [B,N,L,V]."""
        from .dspark import _sample

        squeeze = base_logits.ndim == 3
        if squeeze:
            base_logits = base_logits[:, None]
            first_prev_token_ids = first_prev_token_ids[:, None]
            query_position_ids = query_position_ids[:, None]
            anchor_positions = anchor_positions.reshape(-1, 1)
            if block_keep_mask is not None:
                block_keep_mask = block_keep_mask.reshape(-1, 1)
        prev = first_prev_token_ids
        tokens, logits = [], []
        for step in range(base_logits.size(-2)):
            corrected = self.apply_block_logits(
                base_logits[:, :, step : step + 1],
                token_ids=prev.unsqueeze(-1),
                context=context,
                query_position_ids=query_position_ids[:, :, step : step + 1],
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                attention_backend=attention_backend,
            ).squeeze(-2)
            batch, blocks, vocab = corrected.shape
            prev = _sample(
                corrected.reshape(batch * blocks, 1, vocab), temperature
            ).reshape(batch, blocks)
            tokens.append(prev)
            logits.append(corrected)
        sampled = (
            torch.stack(tokens, dim=-1)
            if tokens
            else first_prev_token_ids.unsqueeze(-1)[..., :0]
        )
        corrected = torch.stack(logits, dim=-2) if logits else base_logits
        return (sampled[:, 0], corrected[:, 0]) if squeeze else (sampled, corrected)
