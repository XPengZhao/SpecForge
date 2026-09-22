# coding=utf-8
"""DSpark draft model entry point and Markov heads."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from .dflash import DFlashDraftModel
from .registry import register_draft


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(batch_size, seq_len)


class _ZeroInitLinear(nn.Linear):
    """Linear whose construction and HF post-init consume no RNG."""

    _dspark_zero_init = True

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class AcceptRatePredictor(nn.Module):
    """Predict target/draft distribution acceptance probability per draft step."""

    def __init__(self, input_dim: int, *, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class VanillaMarkovHead(nn.Module):
    """Low-rank previous-token logit bias used by DSpark."""

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        if self.markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {self.markov_rank}")
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def get_markov_states(
        self,
        token_ids: torch.Tensor,
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ngram_states
        return self.get_prev_embeddings(token_ids)

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_markov_states(token_ids, ngram_states))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states, ngram_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(
            token_ids, hidden_states, ngram_states
        )

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx]
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx],
                token_ids=prev_token_ids,
                hidden_states=step_hidden,
            )
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class NgramMarkovHead(VanillaMarkovHead):
    """Add a causal Engram feature to the low-rank Markov state."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        rms_norm_eps: float,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "ngram"
        self.ngram_hidden_size = int(hidden_size)
        self.ngram_norm = nn.RMSNorm(
            self.ngram_hidden_size,
            eps=float(rms_norm_eps),
            elementwise_affine=False,
        )
        self.ngram_proj = _ZeroInitLinear(
            self.ngram_hidden_size,
            self.markov_rank,
            bias=False,
        )
        # At initialization this head is exactly the vanilla Markov head. Unlike
        # a zero scalar gate, the projection receives gradients immediately.
        nn.init.zeros_(self.ngram_proj.weight)

    def get_markov_states(
        self,
        token_ids: torch.Tensor,
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ngram_states is None:
            raise ValueError("ngram Markov head requires causal ngram_states")
        expected = (*token_ids.shape, self.ngram_hidden_size)
        if ngram_states.shape != expected:
            raise ValueError(
                "ngram_states must align with previous token IDs: "
                f"expected {expected}, got {tuple(ngram_states.shape)}"
            )
        unigram = self.get_prev_embeddings(token_ids)
        ngram = self.ngram_proj(self.ngram_norm(ngram_states.to(unigram.dtype)))
        return unigram + ngram

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        prefix_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_lookup: Callable[[torch.Tensor], torch.Tensor],
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequentially sample while recomputing Engram from generated prefixes.

        ``ngram_lookup(prefix)`` must return the raw Engram feature for the last
        token in each prefix. This keeps inference causal and avoids using the
        teacher-forced cache after the generated path diverges.
        """
        if prefix_token_ids.ndim != 2 or prefix_token_ids.size(0) != base_logits.size(
            0
        ):
            raise ValueError("prefix_token_ids must be [batch, prefix_length]")
        sampled_tokens = []
        corrected_logits = []
        prefix = prefix_token_ids.long()
        for step_idx in range(base_logits.size(-2)):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx]
            ngram_state = ngram_lookup(prefix)
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx],
                token_ids=prefix[:, -1],
                hidden_states=step_hidden,
                ngram_states=ngram_state,
            )
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(
                step_logits.unsqueeze(1), temperature=temperature
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prefix = torch.cat((prefix, next_token_ids.unsqueeze(1)), dim=1)
        if not sampled_tokens:
            empty = prefix.new_empty(prefix.size(0), 0)
            return empty, base_logits
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkovHead):
    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("gated Markov head requires hidden_states")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ngram_states
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNMarkovHead(VanillaMarkovHead):
    """Recurrent DSpark Markov head unrolled inside one draft block."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        return new_state, self.project_bias(torch.tanh(output_raw))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ngram_states
        if hidden_states is None:
            raise ValueError("rnn Markov head requires hidden_states")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ngram_states
        if hidden_states is None:
            raise ValueError("rnn Markov head requires hidden_states")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        state = torch.zeros(
            *base_logits.shape[:-2],
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for step_idx in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., step_idx])
            state, bias = self._rnn_step(
                state,
                prev_emb,
                hidden_states[..., step_idx, :],
            )
            output_logits.append(base_logits[..., step_idx, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("rnn Markov head requires hidden_states")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx])
            step_logits = base_logits[:, step_idx] + bias
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def build_markov_head(config, dspark_config: dict) -> Optional[nn.Module]:
    markov_rank = int(dspark_config.get("markov_rank", 0) or 0)
    if markov_rank < 0:
        raise ValueError(f"markov_rank must be >= 0, got {markov_rank}")
    if markov_rank == 0:
        return None

    markov_head_type = str(dspark_config.get("markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    if markov_head_type == "ngram":
        return NgramMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")


@register_draft
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone with DSpark Markov/confidence heads."""

    expected_projector_type = "dspark"

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "_dspark_zero_init", False):
            module.reset_parameters()
            return
        super()._init_weights(module)

    def __init__(self, config) -> None:
        dflash_config = getattr(config, "dflash_config", None) or {}
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
            config.dflash_config = dflash_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DSparkDraftModel requires " "dflash_config.projector_type='dspark'."
            )
        super().__init__(config)
        # Qwen3PreTrainedModel.post_init() runs inside the parent constructor
        # after _init_draft_head(), so enforce the exact-baseline initialization
        # after that generic initializer has completed.
        if self.ngram_markov_enabled:
            nn.init.zeros_(self.markov_head.ngram_proj.weight)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.markov_head = build_markov_head(config, dflash_config)
        self.ngram_markov_enabled = isinstance(self.markov_head, NgramMarkovHead)
        confidence_alpha = float(dflash_config.get("confidence_head_alpha", 0.0) or 0.0)
        self.enable_confidence_head = bool(
            dflash_config.get("enable_confidence_head", confidence_alpha > 0.0)
        )
        self.confidence_head_with_markov = bool(
            dflash_config.get("confidence_head_with_markov", False)
        )
        if self.confidence_head_with_markov and self.markov_head is None:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0"
            )

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                input_dim += self.markov_head.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
        ngram_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if self.markov_head is None:
            return base_logits
        if prev_token_ids is None:
            raise ValueError("DSparkDraftModel requires prev_token_ids")
        return self.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
            ngram_states=ngram_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        ngram_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            if prev_token_ids is None:
                raise ValueError("prev_token_ids is required for Markov confidence")
            prev_embeddings = self.markov_head.get_markov_states(
                prev_token_ids, ngram_states
            ).to(hidden_states.dtype)
            hidden_states = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(hidden_states).float()


__all__ = [
    "AcceptRatePredictor",
    "DSparkDraftModel",
    "GatedMarkovHead",
    "NgramMarkovHead",
    "RNNMarkovHead",
    "VanillaMarkovHead",
    "build_markov_head",
]
