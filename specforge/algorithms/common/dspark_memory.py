"""Two-pass DSpark training with detached, per-anchor draft-state memory.

Rejection is simulated against recorded tokens, not a fresh target verification.
Teacher-forced Markov argmax gives the same FIRST reference mismatch as greedy
autoregressive drafting. Predictions after that mismatch are deliberately unused.
"""

import torch

from .dflash_family_model import OnlineDSparkModel


def align_draft_memory(hidden, anchors, accepted, valid, keep, seq_len):
    """Drop accepted rows and the first rejected row; retain prediction positions.

    Hidden row j at query position a+j predicts token a+j+1. The next anchor
    is a+accepted+1 (including the bonus when all B candidates match).
    """
    batch, blocks, width, dim = hidden.shape
    counts = valid.long().sum(-1)
    transition_valid = (accepted < counts) | (counts == width)
    shifted = anchors + accepted + 1
    next_keep = keep & transition_valid & (shifted < seq_len - 1)
    next_anchors = torch.where(next_keep, shifted, torch.zeros_like(shifted))
    slots = torch.arange(max(width - 1, 0), device=hidden.device)
    source = accepted.unsqueeze(-1) + 1 + slots
    safe = source.clamp(max=width - 1)
    memory = hidden.detach().gather(
        2, safe.unsqueeze(-1).expand(batch, blocks, slots.numel(), dim)
    )
    memory_keep = (source < width) & valid.gather(2, safe) & next_keep.unsqueeze(-1)
    # Prediction positions, NOT the previous round's query RoPE positions.
    positions = anchors.unsqueeze(-1) + source + 1
    positions = torch.where(memory_keep, positions, torch.zeros_like(positions))
    return next_anchors, next_keep, memory, positions, memory_keep


class OnlineDSparkMemoryModel(OnlineDSparkModel):
    def __init__(
        self,
        *args,
        draft_memory_keep_prob=1.0,
        draft_memory_head_chunk_size=128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not getattr(self.draft_model, "draft_memory_enabled", False):
            raise ValueError("draft config must enable dflash_config.draft_memory")
        if self.dspark_opd_loss_alpha:
            raise ValueError("draft memory does not support OPD trace training")
        if not 0 <= draft_memory_keep_prob <= 1 or draft_memory_head_chunk_size < 1:
            raise ValueError("invalid draft memory dropout or head chunk size")
        if float(getattr(self.draft_model.config, "attention_dropout", 0)) != 0:
            raise ValueError(
                "draft memory first-pass greedy matching requires dropout=0"
            )
        self.draft_memory_keep_prob = draft_memory_keep_prob
        self.draft_memory_head_chunk_size = draft_memory_head_chunk_size
        self._memory_metrics = {}

    @torch.no_grad()
    def _reference_predictions(self, hidden, previous):
        """Bound vocabulary-logit memory without sequential token dependencies."""
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_prev = previous.reshape(-1)
        predictions = torch.empty_like(flat_prev)
        chunk = self.draft_memory_head_chunk_size
        for start in range(0, flat_hidden.size(0), chunk):
            h = flat_hidden[start : start + chunk]
            # Keep a block axis for all supported Markov head implementations.
            h = h.unsqueeze(1)
            logits = self.draft_model.apply_logits_head(
                self.lm_head(h),
                hidden_states=h,
                prev_token_ids=flat_prev[start : start + chunk].unsqueeze(1),
            )
            predictions[start : start + chunk] = logits.argmax(-1).squeeze(1)
        return predictions.reshape_as(previous)

    def _forward_draft_blocks(
        self,
        input_ids,
        hidden_states,
        loss_mask,
        anchor_positions=None,
        block_keep_mask=None,
    ):
        with torch.no_grad():
            anchors, keep, old_hidden = super()._forward_draft_blocks(
                input_ids, hidden_states, loss_mask, anchor_positions, block_keep_mask
            )
            targets, valid, _ = self._build_dspark_labels_and_mask(
                input_ids, loss_mask, anchors, keep
            )
            previous = torch.cat(
                [input_ids.gather(1, anchors).unsqueeze(-1), targets[..., :-1]], -1
            )
            old_hidden = old_hidden.reshape(
                input_ids.size(0), anchors.size(1), self.block_size, -1
            )
            predicted = self._reference_predictions(old_hidden, previous)
            accepted = ((predicted == targets) & valid).long().cumprod(-1).sum(-1)
            new_anchors, new_keep, memory, positions, memory_keep = align_draft_memory(
                old_hidden, anchors, accepted, valid, keep, input_ids.size(1)
            )
            # Do not cross an assistant-response boundary into a prompt/other turn.
            new_keep = new_keep & (loss_mask.gather(1, new_anchors) > 0.5)
            new_keep = new_keep & (
                loss_mask.gather(1, (new_anchors + 1).clamp(max=input_ids.size(1) - 1))
                > 0.5
            )
            memory_keep = memory_keep & new_keep.unsqueeze(-1)
            if self.training and self.draft_memory_keep_prob < 1:
                memory_keep = memory_keep & (
                    torch.rand_like(anchors, dtype=torch.float32)
                    < self.draft_memory_keep_prob
                ).unsqueeze(-1)
            denominator = new_keep.sum().clamp_min(1)
            self._memory_metrics = {
                "memory_reference_accepted": (accepted * keep).sum()
                / keep.sum().clamp_min(1),
                "memory_rows": memory_keep.sum() / denominator,
                "memory_active_fraction": memory_keep.any(-1).sum() / denominator,
                "memory_transition_fraction": new_keep.sum() / keep.sum().clamp_min(1),
            }
            # Zero padded content as well as masking it; masked slots never
            # expose states belonging to invalid anchors or response boundaries.
            memory = memory.masked_fill(~memory_keep.unsqueeze(-1), 0)
            new_anchors = torch.where(
                new_keep, new_anchors, torch.zeros_like(new_anchors)
            )
        return super()._forward_draft_blocks(
            input_ids,
            hidden_states,
            loss_mask,
            new_anchors,
            new_keep,
            draft_memory=memory,
            memory_positions=positions,
            memory_keep=memory_keep,
        )

    def forward(self, *args, **kwargs):
        loss, accuracy, metrics = super().forward(*args, **kwargs)
        metrics.update(self._memory_metrics)
        return loss, accuracy, metrics
