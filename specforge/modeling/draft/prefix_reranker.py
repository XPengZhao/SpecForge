"""Block-local causal candidate reranker. No cross-round state or target rollout."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PrefixReranker(nn.Module):
    """One pre-norm Transformer, followed by a zero-initialized residual scorer.

    Token features reuse the Markov input/output tables (rank-dimensional),
    not a new vocabulary-sized embedding. Leading dimensions are independent
    blocks. The returned KV tuple belongs to one block and is never stored on self.
    """

    def __init__(
        self,
        hidden_size,
        embedding_size,
        block_size,
        *,
        width=256,
        num_heads=4,
        top_k=16,
    ):
        super().__init__()
        if width <= 0 or num_heads <= 0 or width % num_heads:
            raise ValueError(
                "reranker width must be positive and divisible by num_heads"
            )
        if top_k < 1 or block_size < 1:
            raise ValueError("reranker top_k and block_size must be positive")
        self.width, self.num_heads = width, num_heads
        self.block_size, self.top_k = block_size, top_k
        self.token_proj = nn.Linear(embedding_size, width, bias=False)
        self.position_embed = nn.Embedding(block_size, width)
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width)
        )
        self.output_norm = nn.LayerNorm(width)
        self.hidden_proj = nn.Linear(hidden_size, width, bias=False)
        self.query = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.candidate_proj = nn.Linear(embedding_size, width, bias=False)
        self.residual_out = nn.Linear(width, width, bias=False)
        self.residual_out._prefix_reranker_zero_init = True
        self.reset_parameters()

    def reset_parameters(self):
        # Also called after leaving HF no_init_weights when adding to a baseline.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                module.reset_parameters()
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.zeros_(self.residual_out.weight)

    def encode(self, token_embeddings, cache=None):
        """Parallel causal encoding, or one-token incremental encoding with KV."""
        leading, length = token_embeddings.shape[:-2], token_embeddings.shape[-2]
        start = 0 if cache is None else cache[0].shape[-2]
        if start + length > self.block_size:
            raise ValueError("prefix KV exceeds one block; reset cache at every round")
        if cache is not None and length != 1:
            raise ValueError("incremental reranker expects exactly one new token")
        x = self.token_proj(token_embeddings).reshape(-1, length, self.width)
        pos = torch.arange(start, start + length, device=x.device)
        x = x + self.position_embed(pos)
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)

        def split(t):
            return t.reshape(
                -1, length, self.num_heads, self.width // self.num_heads
            ).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        # A one-token cached query may read every cached key. is_causal=True
        # would use an upper-left mask and incorrectly hide most of that prefix.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        out = out.transpose(1, 2).reshape(-1, length, self.width)
        x = x + self.attn_out(out)
        x = x + self.ffn(self.ffn_norm(x))
        return self.output_norm(x).reshape(*leading, length, self.width), (k, v)

    def score(self, baseline_logits, hidden, prefix, candidate_table):
        k = min(self.top_k, baseline_logits.shape[-1])
        _, ids = baseline_logits.topk(k, dim=-1)
        # topk's tie order is unspecified. Always put torch.argmax first so a
        # zero residual exactly preserves baseline greedy choices, including ties.
        best = baseline_logits.argmax(dim=-1, keepdim=True)
        best_slot = (ids == best).long().argmax(dim=-1, keepdim=True)
        present = (ids == best).any(dim=-1, keepdim=True)
        best_slot = torch.where(present, best_slot, torch.full_like(best_slot, k - 1))
        first_id = ids[..., :1].clone()
        ids = ids.scatter(-1, best_slot, first_id)
        ids[..., :1] = best
        # Gather only after finalizing indices, preserving autograd for joint training.
        values = baseline_logits.gather(-1, ids)
        features = F.embedding(ids, candidate_table)
        keys = self.candidate_proj(features)
        query = self.residual_out(
            self.query(torch.cat((self.hidden_proj(hidden), prefix), dim=-1))
        )
        delta = (query.unsqueeze(-2) * keys).sum(dim=-1) / math.sqrt(self.width)
        return values.float() + delta.float(), ids

    def forward(self, baseline_logits, hidden, prev_token_ids, markov_head):
        prefix, _ = self.encode(markov_head.get_prev_embeddings(prev_token_ids))
        return self.score(baseline_logits, hidden, prefix, markov_head.markov_w2.weight)

    @torch.no_grad()
    def sample(self, base_logits, hidden, anchor_ids, markov_head, *, temperature=0.0):
        if temperature != 0:
            raise ValueError("prefix reranker currently supports greedy sampling only")
        cache, tokens = None, []
        previous = anchor_ids
        for i in range(base_logits.shape[-2]):
            prefix, cache = self.encode(
                markov_head.get_prev_embeddings(previous).unsqueeze(-2), cache
            )
            logits = markov_head.apply_step_logits(
                base_logits[..., i, :],
                token_ids=previous,
                hidden_states=hidden[..., i, :],
            )
            scores, ids = self.score(
                logits,
                hidden[..., i, :],
                prefix.squeeze(-2),
                markov_head.markov_w2.weight,
            )
            previous = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
            tokens.append(previous)
        if not tokens:
            return anchor_ids.new_empty((*anchor_ids.shape, 0))
        return torch.stack(tokens, dim=-1)
