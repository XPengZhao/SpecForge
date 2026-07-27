# coding=utf-8
"""Dense (non-flex) attention fallback for DeepSeek-V4 DSpark.

Exercises the block-diagonal-aware split-softmax path used on Ascend NPU.
"""

import unittest

import torch


def _import_attention():
    from specforge.modeling.draft.deepseek_v4_dspark import (
        DeepseekV4DSparkAttention,
    )

    return DeepseekV4DSparkAttention


class _FakeAttention:
    """Minimal host for ``_dense_attention``; only the attributes it reads."""

    def __init__(self, num_heads, head_dim, attn_sink, block_size):
        self.num_heads = num_heads
        self.scaling = head_dim**-0.5
        self.attn_sink = attn_sink
        self.block_size = block_size


def _build_dflash_mask(B, N, bs, S, device):
    """Context sliding window + block-diagonal draft, matching the sdpa builder."""
    Q = N * bs
    KV = S + Q
    q_idx = torch.arange(Q, device=device).view(1, 1, Q, 1)
    kv_idx = torch.arange(KV, device=device).view(1, 1, 1, KV)
    q_block = q_idx // bs
    # anchors: spread across context
    anchors = torch.linspace(bs, S - 1, N, device=device).long().view(1, 1, N, 1)
    anchor_exp = anchors.repeat_interleave(bs, dim=2)  # (1,1,Q,1)
    ctx = (kv_idx < S) & (kv_idx <= anchor_exp)
    is_draft = kv_idx >= S
    kv_block = (kv_idx - S) // bs
    draft = is_draft & (q_block == kv_block)
    valid = torch.ones(B, 1, N, 1, dtype=torch.bool, device=device).repeat_interleave(
        bs, dim=2
    )
    return (ctx | draft) & valid


def _naive_reference(fake, query, kv, mask):
    """Full dense softmax + sink, the ground truth the split path must match."""
    B, H, Q, d = query.shape
    KV = kv.shape[2]
    kv_f = kv.float()
    scores = torch.matmul(query.float(), kv_f.transpose(-1, -2)) * fake.scaling
    scores = scores.masked_fill(~mask.bool(), torch.finfo(torch.float32).min)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, kv_f)
    lse = torch.logsumexp(scores, dim=-1)
    sink = torch.sigmoid(lse - fake.attn_sink.float().view(1, -1, 1)).to(out.dtype)
    return out * sink.unsqueeze(-1)


class DeepseekV4DSparkDenseAttentionTest(unittest.TestCase):
    def setUp(self):
        try:
            self._cls = _import_attention()
        except Exception:  # pragma: no cover - env without transformers/DSv4
            self.skipTest("deepseek_v4 transformers module unavailable")

    def _run(self, fake, query, kv, mask):
        return self._cls._dense_attention(fake, query, kv, mask)

    def test_matches_naive_dense_softmax(self):
        torch.manual_seed(0)
        B, H, N, bs, d, S = 2, 3, 4, 5, 8, 7
        Q = N * bs
        KV = S + Q
        fake = _FakeAttention(H, d, torch.randn(H) * 0.1, bs)
        q = torch.randn(B, H, Q, d)
        kv = torch.randn(B, 1, KV, d)
        mask = _build_dflash_mask(B, N, bs, S, q.device)

        out = self._run(fake, q, kv, mask)
        ref = _naive_reference(fake, q, kv, mask)

        self.assertEqual(out.shape, (B, H, Q, d))
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-4, rtol=1e-4)

    def test_kv_longer_than_query_realistic_shape(self):
        torch.manual_seed(7)
        B, H, N, bs, d, S = 1, 2, 8, 5, 16, 4096
        Q = N * bs
        KV = S + Q
        fake = _FakeAttention(H, d, torch.zeros(H), bs)
        q = torch.randn(B, H, Q, d)
        kv = torch.randn(B, 1, KV, d)
        mask = _build_dflash_mask(B, N, bs, S, q.device)

        out = self._run(fake, q, kv, mask, chunk=2)

        self.assertEqual(out.shape, (B, H, Q, d))
        self.assertTrue(torch.isfinite(out).all())

    def test_invalid_block_is_zeroed(self):
        torch.manual_seed(2)
        B, H, N, bs, d, S = 1, 2, 3, 4, 8, 6
        Q = N * bs
        KV = S + Q
        fake = _FakeAttention(H, d, torch.zeros(H), bs)
        q = torch.randn(B, H, Q, d)
        kv = torch.randn(B, 1, KV, d)
        mask = _build_dflash_mask(B, N, bs, S, q.device)
        mask[:, :, bs : 2 * bs, :] = False  # block 1 fully masked

        out = self._run(fake, q, kv, mask)

        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out[:, :, bs : 2 * bs, :].abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
