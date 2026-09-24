import unittest

import torch
import torch.nn as nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    create_dflash_sdpa_mask,
)


class _Draft(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.carry_embed = nn.Parameter(torch.zeros(feature_dim))


class TestCleanCarry(unittest.TestCase):
    def test_mask_reads_future_suffix_and_skips_anchor_state(self):
        anchors = torch.tensor([[2]])
        keep = torch.tensor([[True]])
        carry_keep = torch.tensor([[[True, True, False]]])
        mask = create_dflash_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=8,
            block_size=4,
            device=torch.device("cpu"),
            carry_len=3,
            carry_keep=carry_keep,
        )
        visible = mask[0, 0, 0].tolist()
        self.assertEqual(
            visible,
            [
                True, True, False, False, False, False, False, False,
                True, True, False,
                True, True, True, True,
            ],
        )

    def test_blocks_cannot_read_each_others_carry(self):
        anchors = torch.tensor([[1, 4]])
        keep = torch.tensor([[True, True]])
        carry_keep = torch.ones(1, 2, 2, dtype=torch.bool)
        mask = create_dflash_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=6,
            block_size=2,
            device=torch.device("cpu"),
            carry_len=2,
            carry_keep=carry_keep,
        )
        # KV: context 6 + carry 4 + draft 4. Block 0 queries start at 0.
        block0_carry = mask[0, 0, 0, 6:10].tolist()
        block1_carry = mask[0, 0, 2, 6:10].tolist()
        self.assertEqual(block0_carry, [True, True, False, False])
        self.assertEqual(block1_carry, [False, False, True, True])

    def test_sampled_carry_starts_after_anchor(self):
        feature_dim = 3
        model = OnlineDFlashModel(
            draft_model=_Draft(feature_dim),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(4, feature_dim),
            mask_token_id=0,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=1,
            carry_enabled=True,
            carry_keep_prob=1.0,
        )
        hidden = torch.arange(8 * feature_dim, dtype=torch.float32).view(1, 8, feature_dim)
        anchors = torch.tensor([[2]])
        keep = torch.tensor([[True]])
        torch.manual_seed(0)
        target, positions, carry_len, carry_keep = model._sample_clean_carry(
            hidden, anchors, keep
        )
        self.assertEqual(carry_len, 3)
        self.assertEqual(target.shape[1], 8 + 3)
        self.assertTrue(torch.all(positions >= 3))
        self.assertFalse(torch.any(positions == 2))
        self.assertTrue(torch.all(positions[carry_keep.view(1, -1)] > anchors))
        model.carry_embed = model.draft_model.carry_embed
        added = target[:, 8:] - hidden[:, positions.view(-1)]
        self.assertTrue(torch.equal(added, model.draft_model.carry_embed.expand_as(added)))


if __name__ == "__main__":
    unittest.main()
