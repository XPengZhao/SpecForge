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
"""Trainable DeepSeek-V4 DSpark draft model backed by HF ``mtp.*`` weights."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4GroupedLinear,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4UnweightedRMSNorm,
    apply_rotary_pos_emb,
)

from .dspark import AcceptRatePredictor, VanillaMarkovHead
from .registry import register_draft

try:
    from torch.nn.attention.flex_attention import AuxRequest, BlockMask

    from .flex_attention import compile_friendly_flex_attention

    _LSE_AUX_REQUEST = AuxRequest(lse=True)
except ImportError:
    BlockMask = None
    compile_friendly_flex_attention = None
    _LSE_AUX_REQUEST = None


def _method_config(config: DeepseekV4Config) -> dict:
    value = getattr(config, "dflash_config", None)
    return dict(value) if value else {}


class DeepseekV4DSparkConfig(DeepseekV4Config):
    """DeepSeek-V4 config extended with draft-only DSpark metadata."""

    model_type = "deepseek_v4_dspark"

    def __init__(
        self,
        *,
        dflash_config: Optional[dict] = None,
        draft_vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dflash_config = dict(dflash_config or {})
        self.draft_vocab_size = (
            int(draft_vocab_size)
            if draft_vocab_size is not None
            else int(self.vocab_size)
        )


class DeepseekV4DSparkRMSNorm(DeepseekV4RMSNorm):
    """DeepSeek-V4 RMSNorm with FP32 affine weights and computation."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps)
        self.weight.data = self.weight.data.float()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * normalized).to(input_dtype)


_FP4_E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _dequantize_fp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    packed_u8 = packed.view(torch.uint8)
    table = torch.tensor(_FP4_E2M1, dtype=torch.float32)
    low = table[(packed_u8 & 0x0F).long()]
    high = table[(packed_u8 >> 4).long()]
    logical = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    if scale.dtype == torch.uint8:
        scale_f32 = torch.pow(2.0, scale.to(torch.int16).float() - 127.0)
    else:
        scale_f32 = scale.float()
    if (
        scale_f32.ndim != 2
        or scale_f32.shape[0] != logical.shape[0]
        or logical.shape[1] % scale_f32.shape[1] != 0
    ):
        raise ValueError(
            "unsupported DeepSeek-V4 FP4 scale geometry: "
            f"weight={tuple(packed.shape)}, scale={tuple(scale.shape)}"
        )
    block_size = logical.shape[1] // scale_f32.shape[1]
    return (logical * scale_f32.repeat_interleave(block_size, dim=1)).to(dtype)


def _dequantize_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight_f32 = weight.float()
    scale_f32 = scale.float()
    if scale_f32.ndim == 1:
        if scale_f32.numel() == weight_f32.shape[0]:
            expanded = scale_f32[:, None]
        else:
            block_size = (
                weight_f32.shape[0] + scale_f32.numel() - 1
            ) // scale_f32.numel()
            expanded = scale_f32.repeat_interleave(block_size)[
                : weight_f32.shape[0], None
            ]
    elif scale_f32.ndim == 2 and scale_f32.shape[0] == weight_f32.shape[0]:
        tile_size = weight_f32.shape[1] // scale_f32.shape[1]
        expanded = scale_f32.repeat_interleave(tile_size, dim=1)
    elif scale_f32.ndim == 2:
        row_block = (weight_f32.shape[0] + scale_f32.shape[0] - 1) // scale_f32.shape[0]
        col_block = (weight_f32.shape[1] + scale_f32.shape[1] - 1) // scale_f32.shape[1]
        expanded = scale_f32.repeat_interleave(row_block, dim=0)
        expanded = expanded.repeat_interleave(col_block, dim=1)
    else:
        raise ValueError(f"unsupported DeepSeek-V4 FP8 scale rank: {scale_f32.ndim}")
    return (weight_f32 * expanded[: weight_f32.shape[0], : weight_f32.shape[1]]).to(
        dtype
    )


def load_deepseek_v4_dspark_hf_weights(
    model: nn.Module,
    source: str,
    *,
    cache_dir: Optional[str] = None,
) -> tuple[int, tuple[str, ...]]:
    """Stream only ``mtp.*`` tensors from a full DSv4 HF checkpoint."""

    local_source = os.path.abspath(os.path.expanduser(source))
    if not os.path.isdir(local_source):
        local_source = snapshot_download(
            repo_id=source,
            cache_dir=cache_dir,
            allow_patterns=("*.json", "*.safetensors"),
        )
    index_path = os.path.join(local_source, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            "DeepSeek-V4 DSpark loading requires model.safetensors.index.json: "
            f"{local_source}"
        )
    with open(index_path, encoding="utf-8") as stream:
        weight_map = json.load(stream).get("weight_map", {})

    destination = model.state_dict(keep_vars=True)
    required = {key for key in destination if key.startswith("mtp.")}
    missing = sorted(required - weight_map.keys())
    if missing:
        return 0, tuple(missing)

    shard_to_keys: dict[str, list[str]] = {}
    for key in sorted(required):
        shard_to_keys.setdefault(weight_map[key], []).append(key)

    loaded = 0

    def load_scale(scale_key: str) -> torch.Tensor:
        shard = weight_map.get(scale_key)
        if shard is None:
            raise KeyError(scale_key)
        with safe_open(
            os.path.join(local_source, shard), framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(scale_key)

    for shard, keys in shard_to_keys.items():
        with safe_open(
            os.path.join(local_source, shard), framework="pt", device="cpu"
        ) as handle:
            shard_names = set(handle.keys())
            for key in keys:
                value = handle.get_tensor(key)
                scale_key = (
                    key[: -len(".weight")] + ".scale" if key.endswith(".weight") else ""
                )
                if value.dtype == torch.int8 and scale_key:
                    scale = (
                        handle.get_tensor(scale_key)
                        if scale_key in shard_names
                        else load_scale(scale_key)
                    )
                    value = _dequantize_fp4(value, scale, destination[key].dtype)
                elif value.dtype == torch.float8_e4m3fn and scale_key:
                    scale = (
                        handle.get_tensor(scale_key)
                        if scale_key in shard_names
                        else load_scale(scale_key)
                    )
                    value = _dequantize_fp8(value, scale, destination[key].dtype)
                else:
                    value = value.to(destination[key].dtype)
                if value.shape != destination[key].shape:
                    raise ValueError(
                        f"DeepSeek-V4 DSpark tensor shape mismatch for {key}: "
                        f"checkpoint={tuple(value.shape)}, model={tuple(destination[key].shape)}"
                    )
                destination[key].data.copy_(value)
                loaded += 1
    return loaded, ()


class DeepseekV4DSparkAttention(nn.Module):
    """DSv4 shared-KV MLA over target context and one non-causal draft block."""

    def __init__(self, config: DeepseekV4Config, rotary_emb: nn.Module):
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4DSparkRMSNorm(
            config.q_lora_rank, eps=config.rms_norm_eps
        )
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.q_head_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4DSparkRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32)
        )

    def _attention(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        if BlockMask is not None and isinstance(attention_mask, BlockMask):
            output, aux = compile_friendly_flex_attention(
                query,
                kv,
                kv,
                block_mask=attention_mask,
                kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
                scale=self.scaling,
                enable_gqa=True,
                return_aux=_LSE_AUX_REQUEST,
            )
            lse = aux.lse
            if lse is None:
                raise RuntimeError("flex attention did not return the requested LSE")
            sink_scale = torch.sigmoid(
                lse.float() - self.attn_sink.float().view(1, -1, 1)
            ).to(output.dtype)
            return output * sink_scale.unsqueeze(-1)

        raise ValueError(
            "DeepSeek-V4 DSpark requires attention_backend=flex_attention "
            "to preserve sparse masking and the learned attention sink"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        context_length = target_hidden.shape[1]
        kv_hidden = torch.cat([target_hidden, hidden_states], dim=1)

        q = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q).view(batch_size, query_length, self.num_heads, self.head_dim)
        q = self.q_head_norm(q).transpose(1, 2)
        kv = self.kv_norm(self.wkv(kv_hidden)).unsqueeze(1)

        kv_positions = position_ids
        query_positions = position_ids[:, context_length:]
        kv_cos, kv_sin = self.rotary_emb(kv, kv_positions, layer_type="main")
        q_cos, q_sin = self.rotary_emb(q, query_positions, layer_type="main")
        q = apply_rotary_pos_emb(q, q_cos, q_sin)
        kv = apply_rotary_pos_emb(kv, kv_cos, kv_sin)

        output = self._attention(q, kv, attention_mask)
        output = apply_rotary_pos_emb(output, q_cos, -q_sin)
        output = output.transpose(1, 2).contiguous()
        grouped = output.reshape(batch_size, query_length, self.config.o_groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2))


class DeepseekV4DSparkExpert(nn.Module):
    """One routed expert with checkpoint-compatible ``w1/w2/w3`` names."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.w2 = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )
        self.w3 = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.act = ACT2FN[config.hidden_act]
        self.limit = float(config.swiglu_limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.w1(hidden_states).clamp(max=self.limit)
        up = self.w3(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.w2(self.act(gate) * up)


class DeepseekV4DSparkRouter(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.register_buffer(
            "bias",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=True,
        )
        self.top_k = int(config.num_experts_per_tok)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.norm_topk_prob = bool(config.norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor):
        scores = self.score_fn(F.linear(hidden_states.float(), self.weight.float()))
        indices = torch.topk(
            scores + self.bias,
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return indices, weights * self.routed_scaling_factor


class DeepseekV4DSparkMoE(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.gate = DeepseekV4DSparkRouter(config)
        self.experts = nn.ModuleList(
            [DeepseekV4DSparkExpert(config) for _ in range(config.n_routed_experts)]
        )
        self.shared_experts = DeepseekV4DSparkExpert(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        indices, weights = self.gate(flat)
        routed = torch.zeros_like(flat)
        with torch.no_grad():
            active_experts = torch.unique(indices).tolist()
        for expert_id in active_experts:
            token_slot, route_slot = torch.where(indices == expert_id)
            expert_output = self.experts[expert_id](flat[token_slot])
            expert_output = expert_output * weights[token_slot, route_slot].unsqueeze(
                -1
            )
            routed.index_add_(0, token_slot, expert_output.to(routed.dtype))
        return (routed + self.shared_experts(flat)).view(shape)


def _hyper_connection(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float,
    norm_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = hidden_streams.shape[-2]
    flat = hidden_streams.flatten(start_dim=2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre_w, post_w, comb_w = F.linear(flat, fn.float()).split(
        [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_b, post_b, comb_b = base.float().split([hc_mult, hc_mult, hc_mult * hc_mult])
    pre_scale, post_scale, comb_scale = scale.float().unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
    post = torch.sigmoid(post_w * post_scale + post_b) * 2.0
    comb_logits = comb_w.view(
        *comb_w.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
    return post, comb, collapsed.to(hidden_streams.dtype)


def _apply_hyper_connection(
    output: torch.Tensor,
    hidden_streams: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.matmul(
        comb.transpose(-1, -2),
        hidden_streams.float(),
    )
    post_term = post.unsqueeze(-1) * output.float().unsqueeze(-2)
    return (mixed_residual + post_term).to(hidden_streams.dtype)


class DeepseekV4DSparkStage(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        rotary_emb: nn.Module,
        *,
        stage_idx: int,
        num_stages: int,
        target_layer_count: int,
        markov_rank: int,
    ):
        super().__init__()
        hidden_size = int(config.hidden_size)
        hc_mult = int(config.hc_mult)
        hc_width = hc_mult * hidden_size
        hc_mapping_width = (2 + hc_mult) * hc_mult

        if stage_idx == 0:
            self.main_proj = nn.Linear(
                target_layer_count * hidden_size, hidden_size, bias=False
            )
            self.main_norm = DeepseekV4DSparkRMSNorm(
                hidden_size, eps=config.rms_norm_eps
            )

        self.attn_norm = DeepseekV4DSparkRMSNorm(
            hidden_size, eps=config.rms_norm_eps
        )
        self.ffn_norm = DeepseekV4DSparkRMSNorm(
            hidden_size, eps=config.rms_norm_eps
        )
        self.attn = DeepseekV4DSparkAttention(config, rotary_emb)
        self.ffn = DeepseekV4DSparkMoE(config)
        self.hc_attn_fn = nn.Parameter(
            torch.empty(hc_mapping_width, hc_width, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(hc_mapping_width, dtype=torch.float32)
        )
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(hc_mapping_width, hc_width, dtype=torch.float32)
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(hc_mapping_width, dtype=torch.float32)
        )
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        if stage_idx == num_stages - 1:
            self.norm = DeepseekV4DSparkRMSNorm(
                hidden_size, eps=config.rms_norm_eps
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_width, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.markov_head = VanillaMarkovHead(
                vocab_size=config.vocab_size,
                markov_rank=markov_rank,
            )
            self.confidence_head = AcceptRatePredictor(
                input_dim=hidden_size + markov_rank,
                bias=False,
            )

        self.hc_eps = float(config.hc_eps)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)

    def forward(
        self,
        hidden_streams: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        post, comb, collapsed = _hyper_connection(
            hidden_streams,
            self.hc_attn_fn,
            self.hc_attn_base,
            self.hc_attn_scale,
            eps=self.hc_eps,
            norm_eps=self.rms_norm_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
        )
        attn_output = self.attn(
            self.attn_norm(collapsed),
            target_hidden,
            position_ids,
            attention_mask,
        )
        hidden_streams = _apply_hyper_connection(
            attn_output,
            hidden_streams,
            post,
            comb,
        )

        post, comb, collapsed = _hyper_connection(
            hidden_streams,
            self.hc_ffn_fn,
            self.hc_ffn_base,
            self.hc_ffn_scale,
            eps=self.hc_eps,
            norm_eps=self.rms_norm_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
        )
        ffn_output = self.ffn(self.ffn_norm(collapsed))
        return _apply_hyper_connection(
            ffn_output,
            hidden_streams,
            post,
            comb,
        )

    def collapse_head(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = hidden_streams.flatten(2).float()
        flat = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
        )
        mixes = F.linear(flat, self.hc_head_fn.float())
        weights = (
            torch.sigmoid(
                mixes * self.hc_head_scale.float() + self.hc_head_base.float()
            )
            + self.hc_eps
        )
        return (
            (weights.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        )


@register_draft
class DeepseekV4DSparkDraftModel(DeepseekV4PreTrainedModel):
    """DeepSeek-V4 DSpark module matching the checkpoint's ``mtp.*`` tree."""

    config_class = DeepseekV4DSparkConfig
    _no_split_modules = ["DeepseekV4DSparkMoE"]
    _keep_in_fp32_modules_strict = [
        "attn_sink",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "main_norm",
        "attn_norm",
        "ffn_norm",
        "q_norm",
        "kv_norm",
        "norm",
    ]
    _keep_in_fp32_buffers_strict = ["ffn.gate.bias"]
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        std = float(self.config.initializer_range)
        if isinstance(module, DeepseekV4DSparkAttention):
            nn.init.zeros_(module.attn_sink)
        elif isinstance(module, DeepseekV4DSparkRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, DeepseekV4DSparkStage):
            for name in ("hc_attn_fn", "hc_ffn_fn"):
                nn.init.normal_(getattr(module, name), mean=0.0, std=std)
            for name in ("hc_attn_base", "hc_ffn_base"):
                nn.init.zeros_(getattr(module, name))
            for name in ("hc_attn_scale", "hc_ffn_scale"):
                nn.init.ones_(getattr(module, name))
            if hasattr(module, "hc_head_fn"):
                nn.init.normal_(module.hc_head_fn, mean=0.0, std=std)
                nn.init.zeros_(module.hc_head_base)
                nn.init.ones_(module.hc_head_scale)

    def __init__(self, config: DeepseekV4DSparkConfig):
        super().__init__(config)
        method_config = _method_config(config)
        self.target_layer_ids = list(
            method_config.get(
                "target_layer_ids",
                getattr(config, "dspark_target_layer_ids", ()),
            )
        )
        if not self.target_layer_ids:
            raise ValueError(
                "DeepseekV4DSparkDraftModel requires dspark_target_layer_ids"
            )
        self.block_size = int(
            method_config.get("block_size", getattr(config, "dspark_block_size", 0))
        )
        if self.block_size <= 0:
            raise ValueError("DeepSeek-V4 DSpark block_size must be positive")
        self.mask_token_id = int(
            method_config.get(
                "mask_token_id", getattr(config, "dspark_noise_token_id", -1)
            )
        )
        if self.mask_token_id < 0:
            raise ValueError("DeepSeek-V4 DSpark mask token id is missing")
        self.projector_type = "dspark"
        self.context_window = int(config.sliding_window)
        # At inference, the sampled anchor has not passed through the target
        # model yet, so its target-derived context KV is unavailable.
        self.include_anchor_context = False

        num_stages = int(
            method_config.get(
                "num_layers",
                getattr(config, "num_hash_layers", len(self.target_layer_ids)),
            )
        )
        markov_rank = int(
            method_config.get("markov_rank", getattr(config, "dspark_markov_rank", 0))
        )
        if num_stages <= 0 or markov_rank <= 0:
            raise ValueError(
                "DeepSeek-V4 DSpark requires positive num_layers and markov_rank"
            )

        model_dtype = getattr(config, "dtype", torch.bfloat16)
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(model_dtype)
        try:
            self.rotary_emb = DeepseekV4RotaryEmbedding(config)
            self.mtp = nn.ModuleList(
                [
                    DeepseekV4DSparkStage(
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
    def final_stage(self) -> DeepseekV4DSparkStage:
        return self.mtp[-1]

    def fsdp_replicated_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return trainable FP32 parameters excluded from BF16 FSDP casting."""
        return tuple(
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.dtype == torch.float32
        )

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
        first_stage = self.mtp[0]
        target_hidden = first_stage.main_norm(first_stage.main_proj(target_hidden))
        hidden_streams = noise_embedding.unsqueeze(-2).expand(
            -1, -1, self.config.hc_mult, -1
        )
        for stage in self.mtp:
            hidden_streams = stage(
                hidden_streams,
                target_hidden,
                position_ids,
                attention_mask,
            )
        hidden_states = self.final_stage.collapse_head(hidden_streams)
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
            raise ValueError("DeepSeek-V4 DSpark requires prev_token_ids")
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
            raise ValueError("DeepSeek-V4 DSpark requires prev_token_ids")
        markov = self.final_stage.markov_head.get_prev_embeddings(prev_token_ids).to(
            hidden_states.dtype
        )
        return self.final_stage.confidence_head(
            torch.cat([hidden_states, markov], dim=-1)
        ).float()


__all__ = ["DeepseekV4DSparkConfig", "DeepseekV4DSparkDraftModel"]
