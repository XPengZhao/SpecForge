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
"""Trainable DeepSeek-V3 DSpark draft model with sliding-window MLA."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
)

from .deepseek_v3_dspark_attention import DeepseekV3DSparkAttention
from .dspark import AcceptRatePredictor, VanillaMarkovHead
from .registry import register_draft


def _method_config(config: DeepseekV3Config) -> dict:
    value = getattr(config, "dflash_config", None)
    return dict(value) if value else {}


class DeepseekV3DSparkConfig(DeepseekV3Config):
    """DeepSeek-V3 config extended with DSpark-only training metadata."""

    model_type = "deepseek_v3_dspark"

    def __init__(
        self,
        *,
        dflash_config: Optional[dict] = None,
        draft_vocab_size: Optional[int] = None,
        moe_train_group_size: int = 32,
        attention_chunk_size: int = 256,
        sliding_window: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dflash_config = dict(dflash_config or {})
        self.draft_vocab_size = (
            int(draft_vocab_size)
            if draft_vocab_size is not None
            else int(self.vocab_size)
        )
        self.moe_train_group_size = int(moe_train_group_size)
        self.attention_chunk_size = int(attention_chunk_size)
        self.sliding_window = int(sliding_window)
        mlp_type = str(self.dflash_config.get("mlp_type", "moe")).lower()
        if mlp_type not in {"moe", "dense"}:
            raise ValueError(
                "DeepSeek-V3 DSpark dflash_config.mlp_type must be "
                f"'moe' or 'dense', got {mlp_type!r}"
            )
        self.dflash_config["mlp_type"] = mlp_type
        if self.moe_train_group_size <= 0:
            raise ValueError("moe_train_group_size must be positive")
        if self.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive")
        if self.sliding_window < 0:
            raise ValueError("sliding_window must be non-negative")


class DeepseekV3DSparkExpert(nn.Module):
    """One SwiGLU expert with DeepSeek-V3 projection names."""

    def __init__(self, config: DeepseekV3Config, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class DeepseekV3DSparkMLP(DeepseekV3DSparkExpert):
    """Dense DeepSeek-V3 SwiGLU using the full intermediate dimension."""

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config, int(config.intermediate_size))


class DeepseekV3DSparkRouter(nn.Module):
    """DeepSeek-V3 sigmoid router with group-limited expert selection."""

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.n_routed_experts),
            persistent=True,
        )
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.n_routed_experts)
        self.num_groups = int(config.n_group)
        self.topk_groups = int(config.topk_group)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        if self.num_experts % self.num_groups != 0:
            raise ValueError(
                "n_routed_experts must be divisible by n_group: "
                f"{self.num_experts} vs {self.num_groups}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(
            F.linear(hidden_states.float(), self.weight.float())
        )
        choice_scores = scores + self.e_score_correction_bias.float()
        experts_per_group = self.num_experts // self.num_groups
        group_scores = (
            choice_scores.view(-1, self.num_groups, experts_per_group)
            .topk(min(2, experts_per_group), dim=-1)
            .values.sum(dim=-1)
        )
        selected_groups = group_scores.topk(
            self.topk_groups,
            dim=-1,
            sorted=False,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_groups, experts_per_group)
            .reshape(-1, self.num_experts)
        )
        choice_scores = choice_scores.masked_fill(~expert_mask, float("-inf"))
        indices = choice_scores.topk(
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-20)
        return indices, weights * self.routed_scaling_factor


class DeepseekV3DSparkExpertGroup(nn.Module):
    """Consecutive routed experts grouped into one FSDP-friendly unit."""

    def __init__(
        self,
        config: DeepseekV3Config,
        group_size: int,
        global_offset: int,
    ):
        super().__init__()
        self.global_offset = int(global_offset)
        self.experts = nn.ModuleList(
            [
                DeepseekV3DSparkExpert(config, config.moe_intermediate_size)
                for _ in range(group_size)
            ]
        )

    def forward(
        self,
        flat_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        routed: torch.Tensor,
    ) -> torch.Tensor:
        for local_id, expert in enumerate(self.experts):
            expert_id = self.global_offset + local_id
            token_slot, route_slot = torch.where(indices == expert_id)
            # Empty linear calls keep FSDP collective order independent of routing.
            output = expert(flat_states[token_slot])
            output = output * weights[token_slot, route_slot].unsqueeze(-1)
            routed.index_add_(0, token_slot, output.to(routed.dtype))
        return routed


class DeepseekV3DSparkMoE(nn.Module):
    """DeepSeek-V3 MoE split into bounded-size FSDP expert groups."""

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.gate = DeepseekV3DSparkRouter(config)
        num_experts = int(config.n_routed_experts)
        group_size = int(getattr(config, "moe_train_group_size", 32))
        if num_experts % group_size != 0:
            raise ValueError(
                "n_routed_experts must be divisible by moe_train_group_size: "
                f"{num_experts} vs {group_size}"
            )
        self.expert_groups = nn.ModuleList(
            [
                DeepseekV3DSparkExpertGroup(config, group_size, offset)
                for offset in range(0, num_experts, group_size)
            ]
        )
        self.shared_experts = DeepseekV3DSparkExpert(
            config,
            int(config.moe_intermediate_size) * int(config.n_shared_experts),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        indices, weights = self.gate(flat_states)
        routed = torch.zeros_like(flat_states)
        for group in self.expert_groups:
            routed = group(flat_states, indices, weights, routed)
        output = routed + self.shared_experts(flat_states)
        return output.view(original_shape)


class DeepseekV3DSparkStage(nn.Module):
    """One classic pre-norm DeepSeek-V3 decoder stage for DSpark."""

    def __init__(
        self,
        config: DeepseekV3Config,
        rotary_emb: DeepseekV3RotaryEmbedding,
        *,
        stage_idx: int,
        num_stages: int,
        target_layer_count: int,
        markov_rank: int,
    ):
        super().__init__()
        hidden_size = int(config.hidden_size)
        if stage_idx == 0:
            self.main_proj = nn.Linear(
                target_layer_count * hidden_size,
                hidden_size,
                bias=False,
            )
            self.main_norm = DeepseekV3RMSNorm(
                hidden_size,
                eps=config.rms_norm_eps,
            )

        self.input_layernorm = DeepseekV3RMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = DeepseekV3DSparkAttention(config, rotary_emb)
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        mlp_type = str(_method_config(config).get("mlp_type", "moe")).lower()
        if mlp_type == "moe":
            self.mlp = DeepseekV3DSparkMoE(config)
        elif mlp_type == "dense":
            self.mlp = DeepseekV3DSparkMLP(config)
        else:
            raise ValueError(
                "DeepSeek-V3 DSpark dflash_config.mlp_type must be "
                f"'moe' or 'dense', got {mlp_type!r}"
            )

        if stage_idx == num_stages - 1:
            self.norm = DeepseekV3RMSNorm(
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
class DeepseekV3DSparkDraftModel(DeepseekV3PreTrainedModel):
    """Multi-stage DeepSeek-V3 DSpark draft initialized from its config."""

    config_class = DeepseekV3DSparkConfig
    _no_split_modules = [
        "DeepseekV3DSparkMoE",
        "DeepseekV3DSparkExpertGroup",
    ]
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, DeepseekV3DSparkRouter):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=float(self.config.initializer_range),
            )
            nn.init.zeros_(module.e_score_correction_bias)

    def __init__(self, config: DeepseekV3DSparkConfig):
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
                "DeepseekV3DSparkDraftModel requires target_layer_ids"
            )

        self.block_size = int(
            method_config.get(
                "block_size",
                getattr(config, "dspark_block_size", 0),
            )
        )
        if self.block_size <= 0:
            raise ValueError("DeepSeek-V3 DSpark block_size must be positive")
        self.mask_token_id = int(
            method_config.get(
                "mask_token_id",
                getattr(config, "dspark_noise_token_id", -1),
            )
        )
        if self.mask_token_id < 0:
            raise ValueError("DeepSeek-V3 DSpark mask token id is missing")

        self.projector_type = "dspark"
        self.context_window = (
            int(config.sliding_window)
            if int(config.sliding_window) > 0
            else None
        )
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
                "DeepSeek-V3 DSpark requires positive num_layers and markov_rank"
            )

        model_dtype = getattr(config, "dtype", torch.bfloat16)
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(model_dtype)
        try:
            self.rotary_emb = DeepseekV3RotaryEmbedding(config)
            self.mtp = nn.ModuleList(
                [
                    DeepseekV3DSparkStage(
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
    def final_stage(self) -> DeepseekV3DSparkStage:
        return self.mtp[-1]

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if noise_embedding is None or target_hidden is None:
            raise ValueError("noise_embedding and target_hidden are required")
        expected_width = len(self.target_layer_ids) * int(self.config.hidden_size)
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
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if prev_token_ids is None:
            raise ValueError("DeepSeek-V3 DSpark requires prev_token_ids")
        return self.final_stage.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prev_token_ids is None:
            raise ValueError("DeepSeek-V3 DSpark requires prev_token_ids")
        markov = self.final_stage.markov_head.get_prev_embeddings(
            prev_token_ids
        ).to(hidden_states.dtype)
        features = torch.cat((hidden_states, markov), dim=-1)
        return self.final_stage.confidence_head(features).float()


__all__ = [
    "DeepseekV3DSparkConfig",
    "DeepseekV3DSparkDraftModel",
]
