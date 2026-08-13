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
"""Trainable GLM-5.2 MLA+SWA DSpark draft model."""

from __future__ import annotations

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaPreTrainedModel,
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
)

from .dspark import AcceptRatePredictor, VanillaMarkovHead
from .glm52_dspark_attention import Glm52DSparkAttention
from .registry import register_draft


def _method_config(config: GlmMoeDsaConfig) -> dict:
    value = getattr(config, "dflash_config", None)
    return dict(value) if value else {}


class Glm52DSparkConfig(GlmMoeDsaConfig):
    """GLM-5.2 config extended with fixed-SWA DSpark metadata."""

    model_type = "glm52_dspark"

    def __init__(
        self,
        *,
        dflash_config: dict | None = None,
        draft_vocab_size: int | None = None,
        attention_chunk_size: int = 256,
        sliding_window: int = 128,
        rope_interleave: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dflash_config = dict(dflash_config or {})
        self.draft_vocab_size = (
            int(draft_vocab_size)
            if draft_vocab_size is not None
            else int(self.vocab_size)
        )
        self.attention_chunk_size = int(attention_chunk_size)
        self.sliding_window = int(sliding_window)
        self.rope_interleave = bool(rope_interleave)
        self.qk_head_dim = int(self.qk_nope_head_dim) + int(
            self.qk_rope_head_dim
        )
        # Transformers 5.8 reads ``head_dim`` when constructing RoPE. GLM-5.2
        # rotates only the 64-dimensional PE slice, not the full QK head.
        self.head_dim = int(self.qk_rope_head_dim)
        mlp_type = str(self.dflash_config.get("mlp_type", "dense")).lower()
        if mlp_type != "dense":
            raise ValueError(
                "GLM-5.2 DSpark currently requires "
                f"dflash_config.mlp_type='dense', got {mlp_type!r}"
            )
        self.dflash_config["mlp_type"] = mlp_type
        if self.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive")
        if self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")


class Glm52DSparkMLP(nn.Module):
    """Dense GLM SwiGLU block used by every draft stage."""

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class Glm52DSparkStage(nn.Module):
    """One pre-norm GLM MLA+SWA draft stage."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        rotary_emb: GlmMoeDsaRotaryEmbedding,
        *,
        stage_idx: int,
        num_stages: int,
        target_layer_count: int,
        markov_rank: int,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        if stage_idx == 0:
            self.main_proj = nn.Linear(
                target_layer_count * hidden_size,
                hidden_size,
                bias=False,
            )
            self.main_norm = GlmMoeDsaRMSNorm(
                hidden_size,
                eps=config.rms_norm_eps,
            )

        self.input_layernorm = GlmMoeDsaRMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = Glm52DSparkAttention(config, rotary_emb)
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = Glm52DSparkMLP(config)

        if stage_idx == num_stages - 1:
            self.norm = GlmMoeDsaRMSNorm(
                hidden_size,
                eps=config.rms_norm_eps,
            )
            self.markov_head = VanillaMarkovHead(
                vocab_size=config.vocab_size,
                markov_rank=markov_rank,
            )
            self.confidence_head = AcceptRatePredictor(
                input_dim=hidden_size + markov_rank,
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            position_ids,
            attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return residual + hidden_states


@register_draft
class Glm52DSparkDraftModel(GlmMoeDsaPreTrainedModel):
    """Three-stage GLM-5.2 MLA+SWA draft trained by DSpark."""

    config_class = Glm52DSparkConfig
    # Stage-owned projector/head parameters are also used by the outer model,
    # so only wrap the large MLP that is called exclusively via its forward.
    _no_split_modules = ["Glm52DSparkMLP"]
    _supports_flex_attn = True

    def __init__(self, config: Glm52DSparkConfig) -> None:
        super().__init__(config)
        method_config = _method_config(config)
        self.target_layer_ids = [
            int(layer_id)
            for layer_id in method_config.get(
                "target_layer_ids",
                getattr(config, "dspark_target_layer_ids", ()),
            )
        ]
        if not self.target_layer_ids:
            raise ValueError(
                "Glm52DSparkDraftModel requires target_layer_ids"
            )
        if any(
            layer_id < 0 or layer_id >= int(config.num_hidden_layers)
            for layer_id in self.target_layer_ids
        ):
            raise ValueError(
                "GLM-5.2 DSpark target_layer_ids must refer to target layers: "
                f"{self.target_layer_ids}"
            )

        self.block_size = int(
            method_config.get(
                "block_size",
                getattr(config, "dspark_block_size", 0),
            )
        )
        if self.block_size <= 0:
            raise ValueError("GLM-5.2 DSpark block_size must be positive")
        self.mask_token_id = int(
            method_config.get(
                "mask_token_id",
                getattr(config, "dspark_noise_token_id", -1),
            )
        )
        if not 0 <= self.mask_token_id < int(config.vocab_size):
            raise ValueError(
                "GLM-5.2 DSpark mask_token_id must be inside the vocabulary"
            )

        self.projector_type = "dspark"
        self.context_window = int(config.sliding_window)
        self.include_anchor_context = True
        num_stages = int(
            method_config.get(
                "num_layers",
                getattr(config, "dspark_num_hidden_layers", 0),
            )
        )
        markov_rank = int(
            method_config.get(
                "markov_rank",
                getattr(config, "dspark_markov_rank", 0),
            )
        )
        if num_stages <= 0 or markov_rank <= 0:
            raise ValueError(
                "GLM-5.2 DSpark requires positive num_layers and markov_rank"
            )

        model_dtype = getattr(config, "dtype", torch.bfloat16)
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(model_dtype)
        try:
            self.rotary_emb = GlmMoeDsaRotaryEmbedding(config)
            self.mtp = nn.ModuleList(
                [
                    Glm52DSparkStage(
                        config,
                        self.rotary_emb,
                        stage_idx=stage_idx,
                        num_stages=num_stages,
                        target_layer_count=len(self.target_layer_ids),
                        markov_rank=markov_rank,
                    )
                    for stage_idx in range(num_stages)
                ]
            )
            self.post_init()
        finally:
            torch.set_default_dtype(previous_dtype)

    @property
    def final_stage(self) -> Glm52DSparkStage:
        return self.mtp[-1]

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask=None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        if noise_embedding is None or target_hidden is None:
            raise ValueError("noise_embedding and target_hidden are required")
        expected_width = len(self.target_layer_ids) * int(
            self.config.hidden_size
        )
        if target_hidden.shape[-1] != expected_width:
            raise ValueError(
                "target_hidden width must equal len(target_layer_ids) * "
                f"hidden_size ({expected_width}), got {target_hidden.shape[-1]}"
            )

        first_stage = self.mtp[0]
        target_hidden = first_stage.main_norm(
            first_stage.main_proj(target_hidden)
        )
        hidden_states = noise_embedding
        for stage in self.mtp:
            hidden_states = stage(
                hidden_states,
                target_hidden,
                position_ids,
                attention_mask,
            )
        return self.final_stage.norm(hidden_states)

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor | None = None,
        prev_token_embeddings: torch.Tensor | None = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if prev_token_ids is None:
            raise ValueError("GLM-5.2 DSpark requires prev_token_ids")
        return self.final_stage.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prev_token_ids is None:
            raise ValueError("GLM-5.2 DSpark requires prev_token_ids")
        markov = self.final_stage.markov_head.get_prev_embeddings(
            prev_token_ids
        ).to(hidden_states.dtype)
        features = torch.cat((hidden_states, markov), dim=-1)
        return self.final_stage.confidence_head(features).float()


__all__ = ["Glm52DSparkConfig", "Glm52DSparkDraftModel"]
