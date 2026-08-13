# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.2 MLA attention for fixed-window DSpark draft blocks."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
)

from .flex_attention import compile_friendly_flex_attention

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


def _method_config(config: GlmMoeDsaConfig) -> dict:
    value = getattr(config, "dflash_config", None)
    return dict(value) if value else {}


def _rotate_interleaved(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent RoPE pairs in GPT-J/interleaved order."""
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_interleaved_rope(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply GLM-5.2 interleaved RoPE to a ``(B, H, S, D)`` tensor."""
    half_dim = cos.shape[-1] // 2
    cos = cos[..., :half_dim].repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin[..., :half_dim].repeat_interleave(2, dim=-1).unsqueeze(1)
    return hidden_states * cos + _rotate_interleaved(hidden_states) * sin


class Glm52DSparkAttention(nn.Module):
    """GLM-5.2 MLA over an SWA context and independent DSpark blocks."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        rotary_emb: GlmMoeDsaRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb
        self.num_heads = int(config.num_attention_heads)
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.block_size = int(_method_config(config).get("block_size", 0))
        self.attention_chunk_size = int(
            getattr(config, "attention_chunk_size", 256)
        )

        self.q_proj = (
            nn.Linear(
                config.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
            if self.q_lora_rank is None
            else None
        )
        self.q_a_proj = (
            nn.Linear(
                config.hidden_size,
                self.q_lora_rank,
                bias=config.attention_bias,
            )
            if self.q_lora_rank is not None
            else None
        )
        self.q_a_layernorm = (
            GlmMoeDsaRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            if self.q_lora_rank is not None
            else None
        )
        self.q_b_proj = (
            nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
            if self.q_lora_rank is not None
            else None
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        if self.q_lora_rank is None:
            query = self.q_proj(hidden_states)
        else:
            query = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        return query.view(
            batch_size,
            query_length,
            self.num_heads,
            self.qk_head_dim,
        ).transpose(1, 2)

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size, kv_length, _ = hidden_states.shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, key_rope = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        expanded = self.kv_b_proj(kv_latent).view(
            batch_size,
            kv_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            expanded,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        key_nope = key_nope.transpose(1, 2)
        value = value.transpose(1, 2)
        key_rope = key_rope.view(
            batch_size,
            1,
            kv_length,
            self.qk_rope_head_dim,
        )
        return (key_nope, key_rope), value

    def _apply_rope(
        self,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        query_positions: torch.Tensor,
        kv_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_cos, query_sin = self.rotary_emb(query_rope, query_positions)
        key_cos, key_sin = self.rotary_emb(key_rope, kv_positions)
        query_rope = apply_interleaved_rope(
            query_rope,
            query_cos,
            query_sin,
        )
        key_rope = apply_interleaved_rope(key_rope, key_cos, key_sin)
        return query_rope, key_rope

    def _chunked_dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        key_t = key.float().transpose(-1, -2)
        for start in range(0, query.shape[2], self.attention_chunk_size):
            stop = start + self.attention_chunk_size
            scores = torch.matmul(query[:, :, start:stop].float(), key_t)
            scores = scores * self.scaling
            mask = attention_mask[..., start:stop, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask.float()
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            outputs.append(torch.matmul(probabilities, value.float()))
        return torch.cat(outputs, dim=2).to(query.dtype)

    def _dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SWA context and block-diagonal draft attention exactly."""
        if attention_mask.dtype != torch.bool:
            return self._chunked_dense_attention(
                query,
                key,
                value,
                attention_mask,
            )

        batch_size, num_heads, query_length, key_dim = query.shape
        context_length = key.shape[2] - query_length
        block_size = self.block_size
        if block_size <= 0 or query_length % block_size:
            raise ValueError(
                "GLM-5.2 DSpark query length must be divisible by block_size: "
                f"query_length={query_length}, block_size={block_size}"
            )
        num_blocks = query_length // block_size
        mask = attention_mask.bool()
        key = key.float()
        value = value.float()

        context_key = key[:, :, :context_length]
        context_value = value[:, :, :context_length]
        context_key_t = context_key.transpose(-1, -2)
        context_outputs = []
        context_lse = []
        for start in range(0, query_length, self.attention_chunk_size):
            stop = start + self.attention_chunk_size
            scores = torch.matmul(
                query[:, :, start:stop].float(),
                context_key_t,
            )
            scores = scores * self.scaling
            context_mask = mask[:, :, start:stop, :context_length]
            scores = scores.masked_fill(~context_mask, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            context_outputs.append(torch.matmul(probabilities, context_value))
            context_lse.append(torch.logsumexp(scores, dim=-1))
        context_output = torch.cat(context_outputs, dim=2)
        context_lse = torch.cat(context_lse, dim=2)

        draft_mask = mask[:, :, :, context_length:]
        diagonal_mask = draft_mask.reshape(
            batch_size,
            1,
            num_blocks,
            block_size,
            num_blocks,
            block_size,
        ).diagonal(dim1=2, dim2=4)
        diagonal_mask = diagonal_mask.permute(0, 1, 4, 2, 3).contiguous()
        query_blocks = query.reshape(
            batch_size,
            num_heads,
            num_blocks,
            block_size,
            key_dim,
        ).float()
        draft_key = key[:, :, context_length:].reshape(
            batch_size,
            num_heads,
            num_blocks,
            block_size,
            key_dim,
        )
        draft_value = value[:, :, context_length:].reshape(
            batch_size,
            num_heads,
            num_blocks,
            block_size,
            self.v_head_dim,
        )
        draft_scores = torch.matmul(
            query_blocks,
            draft_key.transpose(-1, -2),
        )
        draft_scores = draft_scores * self.scaling
        draft_scores = draft_scores.masked_fill(
            ~diagonal_mask,
            float("-inf"),
        )
        draft_probabilities = torch.softmax(draft_scores, dim=-1)
        draft_probabilities = torch.nan_to_num(
            draft_probabilities,
            nan=0.0,
        )
        draft_output = torch.matmul(draft_probabilities, draft_value)
        draft_lse = torch.logsumexp(draft_scores, dim=-1)

        context_output = context_output.reshape(
            batch_size,
            num_heads,
            num_blocks,
            block_size,
            self.v_head_dim,
        )
        context_lse = context_lse.reshape(
            batch_size,
            num_heads,
            num_blocks,
            block_size,
        )
        total_lse = torch.logaddexp(context_lse, draft_lse)
        context_weight = torch.exp(context_lse - total_lse)
        draft_weight = torch.exp(draft_lse - total_lse)
        context_weight = torch.nan_to_num(context_weight, nan=0.0)
        draft_weight = torch.nan_to_num(draft_weight, nan=0.0)
        output = (
            context_output * context_weight.unsqueeze(-1)
            + draft_output * draft_weight.unsqueeze(-1)
        )
        return output.reshape(
            batch_size,
            num_heads,
            query_length,
            self.v_head_dim,
        ).to(query.dtype)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        if BlockMask is not None and isinstance(attention_mask, BlockMask):
            return compile_friendly_flex_attention(
                query,
                key,
                value,
                block_mask=attention_mask,
                scale=self.scaling,
            )
        if isinstance(attention_mask, torch.Tensor):
            return self._dense_attention(query, key, value, attention_mask)
        raise ValueError(
            "GLM-5.2 DSpark requires a Flex Attention BlockMask or a dense "
            f"tensor mask; got {type(attention_mask).__name__}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        """Run GLM MLA for draft queries against context plus draft K/V."""
        batch_size, query_length, _ = hidden_states.shape
        context_length = target_hidden.shape[1]
        kv_hidden = torch.cat((target_hidden, hidden_states), dim=1)

        query = self._project_query(hidden_states)
        query_nope, query_rope = torch.split(
            query,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        (key_nope, key_rope), value = self._project_kv(kv_hidden)
        query_rope, key_rope = self._apply_rope(
            query_rope,
            key_rope,
            position_ids[:, context_length:],
            position_ids,
        )
        query = torch.cat((query_nope, query_rope), dim=-1)
        key = torch.cat(
            (key_nope, key_rope.expand(-1, self.num_heads, -1, -1)),
            dim=-1,
        )

        output = self._attention(query, key, value, attention_mask)
        output = output.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.num_heads * self.v_head_dim,
        )
        return self.o_proj(output)


__all__ = ["Glm52DSparkAttention", "apply_interleaved_rope"]
