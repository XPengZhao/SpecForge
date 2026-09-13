# coding=utf-8
"""DFlash-family training models and shared masking helpers."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.algorithms.common.dspark_metrics import acceptance_stats, tau_loss_terms

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


_VALID_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash"}


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_dflash_sdpa_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    context_window=None,
):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if context_window is not None:
        # Match vLLM's sliding-window convention: the configured window
        # includes the current query, leaving window - 1 historical tokens.
        first_context = anchor_expanded - int(context_window) + 1
        mask_context = mask_context & (kv_indices >= first_context)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    context_window: Optional[int] = None,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees the configured context window strictly before its anchor.
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        mask_context = is_context & (kv_idx < anchor_pos)
        if context_window is not None:
            # Match vLLM's sliding-window convention: the configured window
            # includes the current query, leaving window - 1 historical tokens.
            first_context = anchor_pos - int(context_window) + 1
            mask_context = mask_context & (kv_idx >= first_context)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with DFlash and D-PACE losses."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if (anchor_positions is None) != (block_keep_mask is None):
            raise ValueError(
                "anchor_positions and block_keep_mask must be provided together"
            )
        if anchor_positions is None:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
        assert block_keep_mask is not None

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
                context_window=getattr(self.draft_model, "context_window", None),
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
                context_window=getattr(self.draft_model, "context_window", None),
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel block-wise training forward pass; returns
        (loss, accuracy, metrics) — same shape as Domino's forward."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        if self.loss_type == "dflash":
            # Preserve the existing DFlash weighted-mean behavior.
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(self.block_size, device=device).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights

            flat_weights = loss_weights.view(-1)
            valid_token_count = flat_weights.sum() + 1e-6
            loss = (loss_per_token * flat_weights).sum() / valid_token_count
        elif self.loss_type in _DPACE_LOSS_TYPES:
            neg_log_q = loss_per_token.view_as(target_ids)
            with torch.no_grad():
                q = torch.exp(-neg_log_q)
                dpace_weights = self._dpace_weight(
                    q,
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_weights = weight_mask * dpace_weights
            loss = (neg_log_q * loss_weights).sum() / float(bsz)
        else:
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            accuracy = correct.sum().float() / (accuracy_denom + 1e-6)

        return loss, accuracy, {"accuracy_denom": accuracy_denom.detach()}


class OnlineDominoModel(OnlineDFlashModel):
    """Domino online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        shift_label: bool = False,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.shift_label = shift_label

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = max(1, min(self.num_anchors, int(valid_counts.max().item()) - 1))

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(
                0, self.block_size, device=input_ids.device
            ).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
                max=input_ids.size(1) - 1
            )
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )

        return hidden4d, prev_ids

    def _apply_domino_head(
        self,
        base_logits4d: torch.Tensor,
        hidden4d: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        head_token_ids = prev_ids if self.shift_label else target_ids
        head_token_embeddings = self.embed_tokens(head_token_ids)
        return self.draft_model.apply_logits_head(
            base_logits4d,
            hidden_states=hidden4d,
            prev_token_embeddings=head_token_embeddings,
        )

    def _compute_extra_metrics(
        self,
        pred_ids: torch.Tensor,
        flat_base_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        actual_token_count: torch.Tensor,
        target_ids: torch.Tensor,
        eval_weight_mask: torch.Tensor,
        final_loss: torch.Tensor,
        base_loss: torch.Tensor,
        lambda_base: float,
    ) -> Dict[str, torch.Tensor]:
        bsz, n, bs = target_ids.shape

        base_pred_ids = torch.argmax(flat_base_logits, dim=-1)
        base_correct = (base_pred_ids == flat_targets) & (binary_eval_mask > 0.5)
        base_accuracy = base_correct.sum().float() / actual_token_count

        valid_mask_4d = (eval_weight_mask > 0).bool()
        pred_accept_len = compute_accept_len(
            pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )
        base_accept_len = compute_accept_len(
            base_pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )

        valid_block_mask = valid_mask_4d.any(dim=2)
        num_valid_blocks = valid_block_mask.sum().float() + 1e-6
        avg_accept_len = (
            (pred_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks
        base_avg_accept_len = (
            (base_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks

        return {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": base_accuracy.detach(),
            "accept_len": avg_accept_len.detach(),
            "base_accept_len": base_avg_accept_len.detach(),
            "lambda_base": torch.tensor(lambda_base, device=final_loss.device),
        }

    def _compute_weighted_losses(
        self,
        final_logits: torch.Tensor,
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat_logits = final_logits.reshape(-1, final_logits.size(-1))
        flat_base_logits = base_logits.reshape(-1, base_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)

        valid_token_count = flat_weights.sum() + 1e-6

        final_loss_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        final_loss = (final_loss_per_token * flat_weights).sum() / valid_token_count

        base_loss_per_token = F.cross_entropy(
            flat_base_logits, flat_targets, reduction="none"
        )
        base_loss = (base_loss_per_token * flat_weights).sum() / valid_token_count

        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

        return loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ):
        """Parallel Domino training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )

        bsz, n, bs = target_ids.shape
        base_logits = self.lm_head(output_hidden)
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )
        base_logits4d = base_logits.reshape(bsz, n, bs, -1)
        final_logits = self._apply_domino_head(
            base_logits4d=base_logits4d,
            hidden4d=hidden4d,
            prev_ids=prev_ids,
            target_ids=target_ids,
        ).reshape(bsz, n * bs, -1)

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        eval_weight_mask = weight_mask.clone()
        binary_eval_mask = weight_mask.view(-1)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets = (
            self._compute_weighted_losses(
                final_logits=final_logits,
                base_logits=base_logits,
                target_ids=target_ids,
                weight_mask=weight_mask,
                lambda_base=lambda_base,
            )
        )

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            actual_token_count = accuracy_denom + 1e-6
            accuracy = correct.sum().float() / actual_token_count

            metrics = self._compute_extra_metrics(
                pred_ids=pred_ids,
                flat_base_logits=flat_base_logits,
                flat_targets=flat_targets,
                binary_eval_mask=binary_eval_mask,
                actual_token_count=actual_token_count,
                target_ids=target_ids,
                eval_weight_mask=eval_weight_mask,
                final_loss=final_loss,
                base_loss=base_loss,
                lambda_base=lambda_base,
            )
            metrics["accuracy_denom"] = accuracy_denom.detach()

        return loss, accuracy, metrics


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dspark_loss_mode: str = "original",
        dspark_ce_loss_alpha: float = 0.1,
        dspark_l1_loss_alpha: float = 0.9,
        dspark_kl_loss_alpha: float = 1.0,
        dspark_confidence_head_alpha: float = 1.0,
        dspark_tau_loss_alpha: float = 0.0,
        dspark_opd_loss_alpha: float = 0.0,
        dspark_opd_forward_weight: float = 1.0,
        dspark_opd_rejected_weight: float = 1.0,
        dspark_opd_rejected_position_decay: float = 0.8,
        dspark_opd_logprob_min_clamp: float = -80.0,
        dspark_opd_loss_max_clamp: float = 10.0,
        recompute_loss: bool = False,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        if dspark_loss_mode not in {"original", "kl"}:
            raise ValueError("dspark_loss_mode must be 'original' or 'kl'")
        if dspark_ce_loss_alpha < 0:
            raise ValueError("dspark_ce_loss_alpha must be >= 0")
        if dspark_l1_loss_alpha < 0:
            raise ValueError("dspark_l1_loss_alpha must be >= 0")
        if dspark_kl_loss_alpha < 0:
            raise ValueError("dspark_kl_loss_alpha must be >= 0")
        if dspark_confidence_head_alpha < 0:
            raise ValueError("dspark_confidence_head_alpha must be >= 0")
        if dspark_opd_loss_alpha < 0:
            raise ValueError("dspark_opd_loss_alpha must be >= 0")
        if dspark_opd_forward_weight < 0 or dspark_opd_rejected_weight < 0:
            raise ValueError("DSpark OPD stream weights must be >= 0")
        if (
            dspark_opd_loss_alpha > 0
            and dspark_opd_forward_weight == 0
            and dspark_opd_rejected_weight == 0
        ):
            raise ValueError("at least one DSpark OPD stream weight must be positive")
        if not 0 < dspark_opd_rejected_position_decay <= 1:
            raise ValueError("DSpark OPD rejected position decay must be in (0, 1]")
        if dspark_opd_logprob_min_clamp > 0:
            raise ValueError("DSpark OPD logprob minimum clamp must be <= 0")
        if dspark_opd_loss_max_clamp <= 0:
            raise ValueError("DSpark OPD loss maximum clamp must be > 0")

        self.loss_type = "dspark"
        self.dspark_loss_mode = str(dspark_loss_mode)
        self.dspark_ce_loss_alpha = float(dspark_ce_loss_alpha)
        self.dspark_l1_loss_alpha = float(dspark_l1_loss_alpha)
        self.dspark_kl_loss_alpha = float(dspark_kl_loss_alpha)
        if not math.isfinite(dspark_tau_loss_alpha) or dspark_tau_loss_alpha < 0:
            raise ValueError("dspark_tau_loss_alpha must be finite and >= 0")
        self.dspark_tau_loss_alpha = float(dspark_tau_loss_alpha)
        self.dspark_confidence_head_alpha = float(dspark_confidence_head_alpha)
        self.dspark_opd_loss_alpha = float(dspark_opd_loss_alpha)
        self.dspark_opd_forward_weight = float(dspark_opd_forward_weight)
        self.dspark_opd_rejected_weight = float(dspark_opd_rejected_weight)
        self.dspark_opd_rejected_position_decay = float(
            dspark_opd_rejected_position_decay
        )
        self.dspark_opd_logprob_min_clamp = float(dspark_opd_logprob_min_clamp)
        self.dspark_opd_loss_max_clamp = float(dspark_opd_loss_max_clamp)
        self.recompute_loss = bool(recompute_loss)

    def _build_anchor_candidate_mask(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_candidates = max(seq_len - 1, 0)
        if num_candidates == 0:
            return loss_mask[:, :0].bool()
        anchor_valid = loss_mask[:, :num_candidates] > 0.5
        first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
        return anchor_valid & first_target_valid

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample fixed-width DSpark anchors; invalid slots are masked out."""
        valid = self._build_anchor_candidate_mask(seq_len, loss_mask)
        bsz = loss_mask.shape[0]
        num_candidates = valid.shape[1]
        max_n = int(self.num_anchors)
        if num_candidates == 0:
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid_counts = valid.sum(dim=1)
        indices = (
            torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid,
            indices,
            torch.full(indices.shape, seq_len + 1, dtype=indices.dtype, device=indices.device),
        )
        if not self.training:
            sorted_valid = masked_indices.sort(dim=1).values
            slots = torch.arange(max_n, device=device).unsqueeze(0).expand(bsz, -1)
            kept_counts = valid_counts.clamp(max=max_n)
            source_indices = torch.round(
                slots.float()
                * (valid_counts.clamp_min(1).unsqueeze(1) - 1).float()
                / (kept_counts.clamp_min(2).unsqueeze(1) - 1).float()
            ).long()
            source_indices = torch.minimum(
                source_indices,
                valid_counts.clamp_min(1).unsqueeze(1) - 1,
            ).clamp(max=max(num_candidates - 1, 0))
            anchors = torch.gather(sorted_valid, 1, source_indices)
            keep_mask = slots < kept_counts.unsqueeze(1)
            anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
            return anchors, keep_mask

        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(
            valid, random_vals, torch.full(random_vals.shape, 2.0, dtype=random_vals.dtype, device=device)
        )
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        if num_candidates < max_n:
            pad = torch.full(
                (bsz, max_n - num_candidates),
                seq_len + 1,
                dtype=gathered.dtype,
                device=device,
            )
            gathered = torch.cat([gathered, pad], dim=1)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
            valid_counts.unsqueeze(1).clamp(max=max_n)
        )
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _build_dspark_labels_and_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        target_valid = label_indices < seq_len
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
            2,
            safe_label_indices,
        )
        eval_mask = target_valid & (target_loss_mask > 0.5)
        eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
        return target_ids, eval_mask, safe_label_indices

    def _dspark_loss_weight_mask(
        self,
        eval_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight_mask = eval_mask.to(torch.float32)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=eval_mask.device).view(
                1, 1, -1
            )
            decay_weights = torch.exp(-positions.float() / float(self.loss_decay_gamma))
            loss_weight_mask = loss_weight_mask * decay_weights
        return loss_weight_mask

    def _aligned_target_logits(
        self,
        target_last_hidden_states: Optional[torch.Tensor],
        safe_label_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if target_last_hidden_states is None:
            return None
        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        # Gather along the sequence axis only; avoid broadcasting the hidden
        # states across the anchor axis (would force a contiguous copy of
        # (bsz, num_anchors, seq_len, H) — tens of GiB for long sequences).
        bsz, num_anchors, block_size = target_pred_indices.shape
        H = target_last_hidden_states.size(-1)
        flat_idx = target_pred_indices.reshape(bsz, num_anchors * block_size)
        aligned_target_hidden = torch.gather(
            target_last_hidden_states,
            1,
            flat_idx.unsqueeze(-1).expand(-1, -1, H),
        ).view(bsz, num_anchors, block_size, H)
        return self.lm_head(aligned_target_hidden)

    def _select_opd_blocks(
        self,
        *,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_logprobs: torch.Tensor,
        accepted_lengths: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        num_blocks = anchor_positions.size(1)
        if num_blocks == 0:
            raise ValueError("DSpark OPD sample contains no speculative blocks")
        valid_blocks = candidate_mask.any(dim=-1)
        valid_blocks &= anchor_positions >= 0
        valid_blocks &= anchor_positions < input_ids.size(1) - 1
        width = self.num_anchors
        if self.training:
            scores = torch.rand(valid_blocks.shape, device=input_ids.device)
            scores = torch.where(
                valid_blocks, scores, torch.full_like(scores, 2.0)
            )
            selected = scores.argsort(dim=1)
            if num_blocks < width:
                selected = F.pad(selected, (0, width - num_blocks))
            selected = selected[:, :width]
            block_keep_mask = torch.gather(valid_blocks, 1, selected)
            if num_blocks < width:
                block_keep_mask[:, num_blocks:] = False
        else:
            valid_counts = valid_blocks.sum(dim=1)
            indices = torch.arange(num_blocks, device=input_ids.device)
            indices = indices.unsqueeze(0).expand_as(valid_blocks)
            masked_anchor_positions = torch.where(
                valid_blocks,
                anchor_positions,
                torch.full_like(anchor_positions, input_ids.size(1)),
            )
            sorted_valid = torch.gather(
                indices,
                1,
                masked_anchor_positions.argsort(dim=1),
            )
            slots = torch.arange(width, device=input_ids.device).unsqueeze(0)
            slots = slots.expand(valid_blocks.size(0), -1)
            kept_counts = valid_counts.clamp(max=width)
            source_indices = torch.round(
                slots.float()
                * (valid_counts.clamp_min(1).unsqueeze(1) - 1).float()
                / (kept_counts.clamp_min(2).unsqueeze(1) - 1).float()
            ).long()
            source_indices = torch.minimum(
                source_indices,
                valid_counts.clamp_min(1).unsqueeze(1) - 1,
            ).clamp(max=num_blocks - 1)
            selected = torch.gather(sorted_valid, 1, source_indices)
            block_keep_mask = slots < kept_counts.unsqueeze(1)
            selected = torch.where(
                block_keep_mask, selected, torch.zeros_like(selected)
            )

        def gather_blocks(value: torch.Tensor) -> torch.Tensor:
            index = selected
            while index.dim() < value.dim():
                index = index.unsqueeze(-1)
            index = index.expand(-1, -1, *value.shape[2:])
            return torch.gather(value, 1, index)

        return (
            gather_blocks(anchor_positions),
            block_keep_mask,
            gather_blocks(draft_token_ids),
            gather_blocks(target_logprobs),
            gather_blocks(accepted_lengths),
            gather_blocks(candidate_mask),
        )

    def _compute_opd_loss(
        self,
        *,
        input_ids: torch.Tensor,
        base_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        aligned_target_logits: Optional[torch.Tensor],
        output_hidden_4d: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_logprobs: torch.Tensor,
        accepted_lengths: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if aligned_target_logits is None:
            raise ValueError("DSpark OPD response loss requires target logits")

        response_student_logprobs = F.log_softmax(
            draft_logits.float(),
            dim=-1,
        )
        response_student_logprobs = torch.gather(
            response_student_logprobs,
            -1,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
        response_target_logprobs = F.log_softmax(
            aligned_target_logits.float(),
            dim=-1,
        )
        response_target_logprobs = torch.gather(
            response_target_logprobs,
            -1,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)

        min_logprob = self.dspark_opd_logprob_min_clamp
        max_logprob = math.log1p(-torch.finfo(torch.float32).eps)
        response_student_logprobs = response_student_logprobs.clamp(
            min=min_logprob,
            max=max_logprob,
        )
        response_target_logprobs = response_target_logprobs.clamp(
            min=min_logprob,
            max=max_logprob,
        )
        response_target_probs = response_target_logprobs.exp()
        response_losses = response_target_probs * (
            response_target_logprobs - response_student_logprobs
        )
        response_losses += (1.0 - response_target_probs) * (
            torch.log1p(-response_target_probs)
            - torch.log1p(-response_student_logprobs.exp())
        )
        num_candidates = min(draft_token_ids.size(-1), self.block_size)
        draft_token_ids = draft_token_ids[..., :num_candidates]
        target_logprobs = target_logprobs[..., :num_candidates].float()
        candidate_mask = candidate_mask[..., :num_candidates].bool()
        candidate_mask &= block_keep_mask.unsqueeze(-1)
        invalid_token_ids = candidate_mask & (
            (draft_token_ids < 0) | (draft_token_ids >= base_logits.size(-1))
        )
        if bool(invalid_token_ids.any()):
            invalid_id = int(draft_token_ids[invalid_token_ids][0].item())
            raise ValueError(
                f"DSpark OPD draft token ID {invalid_id} is outside the vocabulary"
            )
        if not bool(torch.isfinite(target_logprobs[candidate_mask]).all()):
            raise ValueError("DSpark OPD target logprobs must be finite")
        if bool((target_logprobs[candidate_mask] > 1e-6).any()):
            raise ValueError("DSpark OPD target logprobs must be <= 0")
        draft_token_ids = torch.where(
            candidate_mask,
            draft_token_ids,
            torch.zeros_like(draft_token_ids),
        )
        candidate_counts = candidate_mask.sum(dim=-1)
        accepted_lengths = accepted_lengths.clamp(min=0)
        accepted_lengths = torch.minimum(accepted_lengths, candidate_counts)

        response_losses = response_losses[..., :num_candidates].clamp(
            min=-self.dspark_opd_loss_max_clamp,
            max=self.dspark_opd_loss_max_clamp,
        )
        offsets = torch.arange(num_candidates, device=input_ids.device).view(
            1, 1, -1
        )
        # A rejected verify step contributes its accepted draft prefix plus
        # the target recovery token. A fully accepted block contributes only
        # its draft tokens; the bonus target token is outside this block.
        response_lengths = accepted_lengths + (accepted_lengths < candidate_counts)
        response_mask = eval_mask[..., :num_candidates].bool()
        response_mask &= offsets < response_lengths.unsqueeze(-1)

        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), draft_token_ids[..., :-1]],
            dim=-1,
        )
        opd_logits = base_logits[..., :num_candidates, :]
        opd_logits = self.draft_model.apply_logits_head(
            opd_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=output_hidden_4d[..., :num_candidates, :],
        )
        student_logprobs = F.log_softmax(opd_logits.float(), dim=-1)
        student_logprobs = torch.gather(
            student_logprobs,
            -1,
            draft_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        accepted_mask = candidate_mask & (offsets < accepted_lengths.unsqueeze(-1))
        rejected_mask = candidate_mask & (offsets >= accepted_lengths.unsqueeze(-1))

        log_ratio = (target_logprobs - student_logprobs).clamp(min=-20, max=20)
        rejected_losses = (log_ratio.exp() - log_ratio - 1.0).clamp(
            min=-self.dspark_opd_loss_max_clamp,
            max=self.dspark_opd_loss_max_clamp,
        )
        # Draft-OPD decays by absolute candidate position within the block.
        rejected_weights = self.dspark_opd_rejected_position_decay ** offsets.float()
        rejected_weights = rejected_weights * rejected_mask

        response_count = response_mask.sum().to(torch.float32)
        accepted_count = accepted_mask.sum().to(torch.float32)
        rejected_count = rejected_mask.sum().to(torch.float32)
        response_sum = (response_losses * response_mask).sum()
        rejected_sum = (rejected_losses * rejected_mask).sum()
        rejected_weighted_sum = (rejected_losses * rejected_weights).sum()
        denominator = (
            self.dspark_opd_forward_weight * response_count
            + self.dspark_opd_rejected_weight * rejected_weights.sum()
        )
        numerator = self.dspark_opd_forward_weight * response_sum
        numerator += self.dspark_opd_rejected_weight * rejected_weighted_sum

        global_stats = torch.stack(
            (
                denominator.detach(),
                numerator.detach(),
                response_count.detach(),
                accepted_count.detach(),
                rejected_count.detach(),
                response_sum.detach(),
                rejected_sum.detach(),
            )
        )
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_stats)
            world_size = torch.distributed.get_world_size()
        global_denominator = global_stats[0].clamp_min(1.0)
        # FSDP averages gradients, so compensate after using the global denominator.
        opd_loss = numerator * world_size / global_denominator
        metrics = {
            "opd_loss": global_stats[1] / global_denominator,
            "opd_response_loss": (
                global_stats[5] / global_stats[2].clamp_min(1.0)
            ),
            "opd_rejected_loss": (
                global_stats[6] / global_stats[4].clamp_min(1.0)
            ),
            "opd_response_tokens": global_stats[2] / world_size,
            "opd_accepted_tokens": global_stats[3] / world_size,
            "opd_rejected_tokens": global_stats[4] / world_size,
            "_eval_opd_loss_sum": numerator.detach(),
            "_eval_opd_loss_denom": denominator.detach(),
        }
        return opd_loss, metrics

    def _pos_loss(
        self,
        dl_p: torch.Tensor,        # (bsz, num_anchors, vocab)
        tl_p: Optional[torch.Tensor],   # (bsz, num_anchors, vocab) or None
        tids_p: torch.Tensor,     # (bsz, num_anchors) long
        cconf_p: Optional[torch.Tensor],  # (bsz, num_anchors) or None
    ):
        """Per-position DSpark losses. Designed to run under
        ``torch.utils.checkpoint`` so the (bsz, num_anchors, vocab) fp32
        softmaxes / CE upcast are recomputed in backward instead of being saved.
        """
        bsz, num_anchors = tids_p.shape
        vocab = dl_p.size(-1)

        zeros = dl_p[..., 0].float().new_zeros((bsz, num_anchors))
        if self.dspark_loss_mode == "original":
            # Flatten to 2D for cross_entropy: some PyTorch versions reject a
            # 3D-logit / 2D-target pair with "expected target size [.., V]".
            ce = F.cross_entropy(
                dl_p.reshape(bsz * num_anchors, vocab),
                tids_p.reshape(-1),
                reduction="none",
            ).view(bsz, num_anchors)
        else:
            ce = zeros

        needs_l1 = (
            tl_p is not None
            and (
                (self.dspark_loss_mode == "original" and self.dspark_l1_loss_alpha > 0)
                or self.dspark_tau_loss_alpha > 0
                or cconf_p is not None
                or not self.training
            )
        )
        if needs_l1:
            l1 = (
                torch.softmax(dl_p.float(), dim=-1)
                - torch.softmax(tl_p.float(), dim=-1)
            ).abs().sum(dim=-1)  # (bsz, num_anchors)
        else:
            l1 = zeros

        if self.dspark_loss_mode == "kl" and tl_p is not None:
            teacher_probs = torch.softmax(tl_p.float(), dim=-1)
            student_log_probs = F.log_softmax(dl_p.float(), dim=-1)
            kl = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
        else:
            kl = zeros

        if cconf_p is not None:
            accept = (1.0 - 0.5 * l1).clamp(0.0, 1.0).detach()
            conf_err = F.binary_cross_entropy_with_logits(
                cconf_p.float(), accept, reduction="none"
            )  # (bsz, num_anchors)
        else:
            conf_err = zeros
        return ce, l1, kl, conf_err

    def _compute_dspark_loss(
        self,
        *,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        confidence_pred: Optional[torch.Tensor],
        aligned_target_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_weight_mask = self._dspark_loss_weight_mask(eval_mask)
        ce_loss_den = loss_weight_mask.sum()
        ce_loss_sum = loss_weight_mask.new_zeros(())
        l1_loss_sum = loss_weight_mask.new_zeros(())
        kl_loss_sum = loss_weight_mask.new_zeros(())
        confidence_loss_sum = loss_weight_mask.new_zeros(())
        confidence_abs_error_sum = loss_weight_mask.new_zeros(())
        position_sums = []
        position_denoms = []
        eval_metric_sums = {}
        eval_metric_denoms = {}
        accept_rates = []
        tau_accept_rates = []
        use_confidence_loss = (
            confidence_pred is not None and self.dspark_confidence_head_alpha > 0
        )
        needs_target_distribution = (
            self.dspark_loss_mode == "kl"
            or self.dspark_l1_loss_alpha > 0
            or use_confidence_loss
            or self.dspark_tau_loss_alpha > 0
            or not self.training
        )
        if (
            aligned_target_logits is None
            and (
                self.dspark_loss_mode == "kl"
                or self.dspark_l1_loss_alpha > 0
                or use_confidence_loss
                or self.dspark_tau_loss_alpha > 0
            )
        ):
            raise ValueError(
                "DSpark distribution/confidence loss requires target_last_hidden_states. "
                "Use the disaggregated DSpark server-capture path so the "
                "consumer receives target_last_hidden_states."
            )

        for position in range(self.block_size):
            dl_p = draft_logits[:, :, position, :]
            # Only compute the target distribution when it's actually needed
            # (L1, KL, or confidence); otherwise skip softmax entirely.
            tl_p = (
                None
                if aligned_target_logits is None or not needs_target_distribution
                else aligned_target_logits[:, :, position, :]
            )
            tids_p = target_ids[:, :, position]
            cconf_p = (
                None if not use_confidence_loss else confidence_pred[..., position]
            )
            wmask_p = loss_weight_mask[..., position]

            if self.recompute_loss:
                ce_p, l1_p, kl_p, conf_err_p = _grad_checkpoint(
                    self._pos_loss,
                    dl_p,
                    tl_p,
                    tids_p,
                    cconf_p,
                    use_reentrant=False,
                )
            else:
                ce_p, l1_p, kl_p, conf_err_p = self._pos_loss(
                    dl_p, tl_p, tids_p, cconf_p
                )

            if self.dspark_tau_loss_alpha > 0:
                tau_accept_rates.append((1.0 - 0.5 * l1_p).clamp(0.0, 1.0))

            ce_position_sum = (ce_p * wmask_p).sum()
            position_denom = wmask_p.sum()
            ce_loss_sum = ce_loss_sum + ce_position_sum
            eval_metric_sums[f"mtp_{position + 1}_ce"] = (
                ce_position_sum.detach()
            )
            eval_metric_denoms[f"mtp_{position + 1}_ce"] = (
                position_denom.detach()
            )

            if tl_p is not None:
                l1_position_sum = (l1_p * wmask_p).sum()
                kl_position_sum = (kl_p * wmask_p).sum()
                l1_loss_sum = l1_loss_sum + l1_position_sum
                kl_loss_sum = kl_loss_sum + kl_position_sum
                l1_name = f"mtp_{position + 1}_l1"
                kl_name = f"mtp_{position + 1}_kl"
                eval_metric_sums[l1_name] = l1_position_sum.detach()
                eval_metric_denoms[l1_name] = position_denom.detach()
                eval_metric_sums[kl_name] = kl_position_sum.detach()
                eval_metric_denoms[kl_name] = position_denom.detach()
            else:
                kl_position_sum = kl_p.new_zeros(())

            if aligned_target_logits is not None:
                with torch.no_grad():
                    # Reuse the loss's L1 when available; KL/CE-only runs
                    # still need actual distribution overlap for this metric.
                    if tl_p is not None and (
                        (self.dspark_loss_mode == "original" and self.dspark_l1_loss_alpha > 0)
                        or cconf_p is not None
                        or not self.training
                    ):
                        metric_l1 = l1_p.detach()
                    else:
                        metric_l1 = (
                            dl_p.float().softmax(dim=-1)
                            - aligned_target_logits[:, :, position, :].float().softmax(dim=-1)
                        ).abs().sum(dim=-1)
                    accept_rates.append((1.0 - 0.5 * metric_l1).clamp(0.0, 1.0))

            position_sum = (
                ce_position_sum
                if self.dspark_loss_mode == "original"
                else kl_position_sum
            )
            position_sums.append(position_sum)
            position_denoms.append(position_denom)

            if use_confidence_loss:
                confidence_loss_sum = confidence_loss_sum + (
                    conf_err_p * wmask_p
                ).sum()
                with torch.no_grad():
                    accept_p = (1.0 - 0.5 * l1_p).clamp(0.0, 1.0)
                    confidence_abs_error_sum = confidence_abs_error_sum + (
                        (cconf_p.float().sigmoid() - accept_p).abs() * wmask_p
                    ).sum()

        if accept_rates:
            accept_sums, accept_denoms = acceptance_stats(
                torch.stack(accept_rates, dim=-1), eval_mask
            )
            eval_metric_sums.update(accept_sums)
            eval_metric_denoms.update(accept_denoms)

        if self.dspark_loss_mode == "original":
            objective_sum = (
                self.dspark_ce_loss_alpha * ce_loss_sum
                + self.dspark_l1_loss_alpha * l1_loss_sum
                + self.dspark_confidence_head_alpha * confidence_loss_sum
            )
        else:
            objective_sum = (
                self.dspark_kl_loss_alpha * kl_loss_sum
                + self.dspark_confidence_head_alpha * confidence_loss_sum
            )
        if self.dspark_tau_loss_alpha > 0:
            tau_num, tau_den = tau_loss_terms(torch.stack(tau_accept_rates, dim=-1), eval_mask)
            eval_metric_sums["tau_loss"] = tau_num.detach()
            eval_metric_denoms["tau_loss"] = tau_den.detach()

        global_stats = torch.stack(
            (
                ce_loss_den.detach(),
                ce_loss_sum.detach(),
                l1_loss_sum.detach(),
                kl_loss_sum.detach(),
                confidence_loss_sum.detach(),
                confidence_abs_error_sum.detach(),
                *(value.detach() for value in position_sums),
                *(value.detach() for value in position_denoms),
                *([tau_den.detach()] if self.dspark_tau_loss_alpha > 0 else []),
            )
        )
        world_size = 1
        if (
            self.training
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(global_stats)
            world_size = torch.distributed.get_world_size()

        global_denominator = global_stats[0].clamp_min(1e-6)
        # FSDP averages gradients across ranks. Scale each rank's local
        # numerator so the resulting gradient is sum(N_r) / sum(D_r).
        loss = objective_sum * world_size / global_denominator
        if self.dspark_tau_loss_alpha > 0:
            loss = loss + self.dspark_tau_loss_alpha * tau_num * world_size / global_stats[-1].clamp_min(1.0)
        position_offset = 6
        global_position_sums = global_stats[
            position_offset : position_offset + self.block_size
        ]
        global_position_denoms = global_stats[
            position_offset + self.block_size : position_offset + 2 * self.block_size
        ]
        position_metrics = {
            f"mtp_{position + 1}_loss": (
                global_position_sums[position]
                / global_position_denoms[position].clamp_min(1e-6)
            )
            for position in range(self.block_size)
        }
        position_metrics.update(
            {
                f"mtp_{position + 1}_minus_{position}_loss": (
                    position_metrics[f"mtp_{position + 1}_loss"]
                    - position_metrics[f"mtp_{position}_loss"]
                )
                for position in range(1, self.block_size)
            }
        )
        metrics = {
            "ce_loss": global_stats[1] / global_denominator,
            "l1_loss": global_stats[2] / global_denominator,
            "kl_loss": global_stats[3] / global_denominator,
            "confidence_loss": global_stats[4] / global_denominator,
            "confidence_abs_error": global_stats[5] / global_denominator,
            "metric_loss_denoms": [ce_loss_den.detach()],
            "eval_metric_sums": eval_metric_sums,
            "eval_metric_denoms": eval_metric_denoms,
            **position_metrics,
        }
        eval_metric_sums["ce_loss"] = ce_loss_sum.detach()
        eval_metric_denoms["ce_loss"] = ce_loss_den.detach()
        if aligned_target_logits is not None and needs_target_distribution:
            eval_metric_sums["l1_loss"] = l1_loss_sum.detach()
            eval_metric_denoms["l1_loss"] = ce_loss_den.detach()
            eval_metric_sums["kl_loss"] = kl_loss_sum.detach()
            eval_metric_denoms["kl_loss"] = ce_loss_den.detach()
        if use_confidence_loss:
            eval_metric_sums["confidence_loss"] = confidence_loss_sum.detach()
            eval_metric_denoms["confidence_loss"] = ce_loss_den.detach()
            eval_metric_sums["confidence_abs_error"] = confidence_abs_error_sum.detach()
            eval_metric_denoms["confidence_abs_error"] = ce_loss_den.detach()
        return loss, metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        opd_anchor_positions: Optional[torch.Tensor] = None,
        opd_draft_token_ids: Optional[torch.Tensor] = None,
        opd_target_logprobs: Optional[torch.Tensor] = None,
        opd_accepted_lengths: Optional[torch.Tensor] = None,
        opd_candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel DSpark training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz = input_ids.shape[0]
        opd_values = (
            opd_anchor_positions,
            opd_draft_token_ids,
            opd_target_logprobs,
            opd_accepted_lengths,
            opd_candidate_mask,
        )
        has_opd_features = all(value is not None for value in opd_values)
        if any(value is not None for value in opd_values) and not has_opd_features:
            raise ValueError("DSpark OPD requires all opd_* tensors")
        if self.training and self.dspark_opd_loss_alpha > 0 and not has_opd_features:
            raise ValueError(
                "training.dspark_opd_loss_alpha > 0 requires OPD trace features"
            )
        use_opd = self.dspark_opd_loss_alpha > 0 and has_opd_features
        selected_opd = None
        if use_opd and opd_anchor_positions is not None and opd_anchor_positions.size(1):
            assert opd_anchor_positions is not None
            assert opd_draft_token_ids is not None
            assert opd_target_logprobs is not None
            assert opd_accepted_lengths is not None
            assert opd_candidate_mask is not None
            selected_opd = self._select_opd_blocks(
                input_ids=input_ids,
                anchor_positions=opd_anchor_positions,
                draft_token_ids=opd_draft_token_ids,
                target_logprobs=opd_target_logprobs,
                accepted_lengths=opd_accepted_lengths,
                candidate_mask=opd_candidate_mask,
            )
            anchor_positions, block_keep_mask = selected_opd[:2]
        else:
            anchor_positions = None
            block_keep_mask = None
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )

        logits = self.lm_head(output_hidden)
        num_blocks = anchor_positions.size(1)
        if use_opd and selected_opd is None:
            # Keep every FSDP rank on the same collective sequence when a
            # rank-local sample has no OPD blocks after truncation.
            candidate_shape = (bsz, num_blocks, self.block_size)
            selected_opd = (
                anchor_positions,
                block_keep_mask,
                torch.zeros(
                    candidate_shape, dtype=input_ids.dtype, device=input_ids.device
                ),
                torch.zeros(
                    candidate_shape, dtype=torch.float32, device=input_ids.device
                ),
                torch.zeros(
                    (bsz, num_blocks), dtype=torch.long, device=input_ids.device
                ),
                torch.zeros(
                    candidate_shape, dtype=torch.bool, device=input_ids.device
                ),
            )
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)
        (
            target_ids,
            eval_mask,
            safe_label_indices,
        ) = self._build_dspark_labels_and_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        base_logits = logits.reshape(bsz, num_blocks, self.block_size, -1)
        draft_logits = self.draft_model.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=output_hidden_4d,
        )
        confidence_pred = None
        if self.dspark_confidence_head_alpha > 0:
            confidence_pred = self.draft_model.predict_confidence(
                output_hidden_4d,
                prev_token_ids=prev_token_ids,
            )
        aligned_target_logits = self._aligned_target_logits(
            target_last_hidden_states,
            safe_label_indices,
        )
        loss, metrics = self._compute_dspark_loss(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )
        if selected_opd is not None:
            (
                _,
                _,
                selected_draft_token_ids,
                selected_target_logprobs,
                selected_accepted_lengths,
                selected_candidate_mask,
            ) = selected_opd
            opd_loss, opd_metrics = self._compute_opd_loss(
                input_ids=input_ids,
                base_logits=base_logits,
                draft_logits=draft_logits,
                target_ids=target_ids,
                eval_mask=eval_mask,
                aligned_target_logits=aligned_target_logits,
                output_hidden_4d=output_hidden_4d,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                draft_token_ids=selected_draft_token_ids,
                target_logprobs=selected_target_logprobs,
                accepted_lengths=selected_accepted_lengths,
                candidate_mask=selected_candidate_mask,
            )
            opd_loss_sum = opd_metrics.pop("_eval_opd_loss_sum")
            opd_loss_denom = opd_metrics.pop("_eval_opd_loss_denom")
            loss = loss + self.dspark_opd_loss_alpha * opd_loss
            metrics.update(opd_metrics)
            metrics["eval_metric_sums"]["opd_loss"] = opd_loss_sum
            metrics["eval_metric_denoms"]["opd_loss"] = opd_loss_denom
        if not self.training:
            if self.dspark_loss_mode == "kl":
                objective_weights = {
                    "kl_loss": self.dspark_kl_loss_alpha,
                    "confidence_loss": self.dspark_confidence_head_alpha,
                    "opd_loss": self.dspark_opd_loss_alpha,
                }
            else:
                objective_weights = {
                    "ce_loss": self.dspark_ce_loss_alpha,
                    "l1_loss": self.dspark_l1_loss_alpha,
                    "confidence_loss": self.dspark_confidence_head_alpha,
                    "opd_loss": self.dspark_opd_loss_alpha,
                }
            objective_weights["tau_loss"] = self.dspark_tau_loss_alpha
            metrics["eval_objective_weights"] = {
                name: weight
                for name, weight in objective_weights.items()
                if weight > 0 and name in metrics["eval_metric_sums"]
            }
        flat_logits = draft_logits.reshape(-1, draft_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        binary_eval_mask = eval_mask.reshape(-1)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & binary_eval_mask
            accuracy_denom = binary_eval_mask.to(torch.float32).sum()
            accuracy = correct.sum().float() / (accuracy_denom + 1e-6)
            metrics["accuracy_denom"] = accuracy_denom.detach()
            correct_3d = correct.view_as(eval_mask)
            metrics["acc_corrects"] = [
                correct_3d[..., position].sum().detach()
                for position in range(self.block_size)
            ]
            metrics["acc_denoms"] = [
                eval_mask[..., position].sum().detach()
                for position in range(self.block_size)
            ]
        return loss, accuracy, metrics
