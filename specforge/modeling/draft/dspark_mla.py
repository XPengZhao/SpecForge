"""DSpark MLA ablation with decoupled RoPE and expanded-KV attention.

This reference training path caches expanded K/V, not compressed latents.
"""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward, rotate_half,
)


class DSparkMLAAttention(nn.Module):
    def __init__(self, config, layer_idx, kernels):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.rank = getattr(config, "mla_kv_lora_rank", 512)
        self.nope_dim = getattr(config, "mla_qk_nope_head_dim", 64)
        self.rope_dim = getattr(config, "mla_qk_rope_head_dim", 64)
        self.v_dim = getattr(config, "mla_v_head_dim", 128)
        for name, value in (("kv_lora_rank", self.rank), ("qk_nope_head_dim", self.nope_dim),
                            ("qk_rope_head_dim", self.rope_dim), ("v_head_dim", self.v_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"mla_{name} must be a positive integer")
            setattr(config, f"mla_{name}", value)
        if self.rope_dim % 2:
            raise ValueError("mla_qk_rope_head_dim must be even")
        self.head_dim = self.nope_dim + self.rope_dim
        # Equal QK/V widths keep the existing Flex/Flash backends portable.
        if self.v_dim != self.head_dim:
            raise ValueError("MLA currently requires v_head_dim = qk_nope_head_dim + qk_rope_head_dim")
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False
        self.attention_dropout = config.attention_dropout
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.kv_a_proj = nn.Linear(config.hidden_size, self.rank + self.rope_dim, bias=bias)
        self.kv_a_norm = kernels.make_rms_norm(self.rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.rank, self.num_heads * (self.nope_dim + self.v_dim), bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.v_dim, config.hidden_size, bias=bias)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        source = torch.cat((target_hidden, hidden_states), dim=1)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_content, q_rope = q.split((self.nope_dim, self.rope_dim), dim=-1)
        latent, k_rope = self.kv_a_proj(source).split((self.rank, self.rope_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_norm(latent)).view(
            bsz, source.size(1), self.num_heads, self.nope_dim + self.v_dim
        ).transpose(1, 2)
        k_content, v = kv.split((self.nope_dim, self.v_dim), dim=-1)
        cos, sin = (x.unsqueeze(1) for x in position_embeddings)
        q_rope = q_rope * cos[..., -q_len:, :] + rotate_half(q_rope) * sin[..., -q_len:, :]
        k_rope = k_rope.unsqueeze(1)
        k_rope = k_rope * cos + rotate_half(k_rope) * sin
        q = torch.cat((q_content, q_rope), dim=-1)
        k = torch.cat((k_content, k_rope.expand(-1, self.num_heads, -1, -1)), dim=-1)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, {"cache_position": cache_position})
        attn_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        output, weights = attn_fn(
            self, q, k, v, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        return self.o_proj(output.reshape(bsz, q_len, -1)), weights
