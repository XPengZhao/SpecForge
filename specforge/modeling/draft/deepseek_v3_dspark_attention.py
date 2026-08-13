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
"""DeepSeek-V3 MLA attention for a sliding-window DSpark draft block.

The surrounding DFlash-family wrapper owns the attention topology: it builds a
mask in which each draft block sees only its target-context sliding window and
its own non-causal draft tokens.  This module owns the DeepSeek-V3 projection
layout and applies that mask through Flex Attention or a dense fallback.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)

from .flex_attention import compile_friendly_flex_attention

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


def _yarn_attention_scale(config: DeepseekV3Config) -> float:
    """Match the YaRN attention scaling used by HF DeepSeek-V3."""
    scaling = float(config.qk_head_dim) ** -0.5
    rope_parameters = config.rope_parameters
    if rope_parameters.get("rope_type", "default") == "default":
        return scaling

    mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
    factor = rope_parameters["factor"]
    if mscale_all_dim and factor > 1:
        mscale = 0.1 * float(mscale_all_dim) * math.log(factor) + 1.0
        scaling *= mscale * mscale
    return scaling


class DeepseekV3DSparkAttention(nn.Module):
    """DeepSeek-V3 MLA over target context and non-causal DSpark draft blocks.

    Parameter names intentionally follow ``DeepseekV3Attention``:

    * ``q_proj`` or ``q_a_proj`` / ``q_a_layernorm`` / ``q_b_proj``
    * ``kv_a_proj_with_mqa`` / ``kv_a_layernorm`` / ``kv_b_proj``
    * ``o_proj``

    Unlike DeepSeek-V4 attention, V3 has separate non-RoPE query/key dimensions
    and value dimensions.  It therefore expands compressed KV to per-head K/V
    before attention and does not use V4's attention sink, shared K=V tensor,
    inverse output rotation, or grouped output projection.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        rotary_emb: DeepseekV3RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb
        self.num_heads = int(config.num_attention_heads)
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = _yarn_attention_scale(config)
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
            DeepseekV3RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
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
        self.kv_a_layernorm = DeepseekV3RMSNorm(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, kv_length, _ = hidden_states.shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, key_rope = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_pass = self.kv_a_layernorm(kv_pass)
        expanded = self.kv_b_proj(kv_pass).view(
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
        rope_fn = (
            apply_rotary_pos_emb_interleave
            if self.config.rope_interleave
            else apply_rotary_pos_emb
        )
        query_rope, _ = rope_fn(
            query_rope,
            query_rope,
            query_cos,
            query_sin,
        )
        _, key_rope = rope_fn(
            key_rope,
            key_rope,
            key_cos,
            key_sin,
        )
        return query_rope, key_rope

    def _dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked eager attention for dense DSpark boolean/additive masks."""
        outputs = []
        key_t = key.transpose(-1, -2)
        mask_is_bool = attention_mask.dtype == torch.bool

        for start in range(0, query.shape[2], self.attention_chunk_size):
            stop = start + self.attention_chunk_size
            query_chunk = query[:, :, start:stop].float()
            scores = torch.matmul(query_chunk, key_t.float()) * self.scaling
            mask = attention_mask[..., start:stop, :]
            if mask_is_bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask.float()

            probabilities = torch.softmax(scores, dim=-1)
            # Invalid/padded DSpark blocks can have fully masked rows.
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            outputs.append(torch.matmul(probabilities, value.float()))

        return torch.cat(outputs, dim=2).to(query.dtype)

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
            "DeepSeek-V3 DSpark requires a Flex Attention BlockMask or a "
            f"dense tensor mask; got {type(attention_mask).__name__}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        """Run MLA for draft queries against context plus draft K/V.

        ``position_ids`` contains context positions followed by the absolute
        positions of every parallel draft block, matching the layout used by
        ``create_dflash_block_mask`` and ``create_dflash_sdpa_mask``.
        """
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


__all__ = ["DeepseekV3DSparkAttention"]
