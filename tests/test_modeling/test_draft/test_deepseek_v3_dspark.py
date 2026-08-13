# coding=utf-8
"""DeepSeek-V3 DSpark attention and draft-model tests."""

import unittest

import torch

from specforge.algorithms.common.dflash_family_model import (
    create_dflash_sdpa_mask,
)


def _model_classes():
    from specforge.modeling.draft.deepseek_v3_dspark import (
        DeepseekV3DSparkConfig,
        DeepseekV3DSparkDraftModel,
    )

    return DeepseekV3DSparkConfig, DeepseekV3DSparkDraftModel


def _attention_class():
    from specforge.modeling.draft.deepseek_v3_dspark_attention import (
        DeepseekV3DSparkAttention,
    )

    return DeepseekV3DSparkAttention


class _FakeAttention:
    def __init__(self, scaling: float, chunk_size: int = 2):
        self.scaling = scaling
        self.attention_chunk_size = chunk_size


def _naive_attention(fake, query, key, value, mask):
    scores = torch.matmul(
        query.float(),
        key.float().transpose(-1, -2),
    ) * fake.scaling
    scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def _tiny_config(*, sliding_window: int, mlp_type: str = "moe"):
    config_cls, _ = _model_classes()
    return config_cls(
        architectures=["DeepseekV3DSparkDraftModel"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        max_position_embeddings=32,
        dtype="float32",
        moe_train_group_size=2,
        attention_chunk_size=2,
        sliding_window=sliding_window,
        dflash_config={
            "block_size": 2,
            "markov_rank": 4,
            "mask_token_id": 0,
            "mlp_type": mlp_type,
            "num_layers": 2,
            "projector_type": "dspark",
            "target_layer_ids": [0, 1],
        },
    )


class DeepseekV3DSparkAttentionTest(unittest.TestCase):
    def setUp(self):
        self.attention_cls = _attention_class()

    def test_dense_attention_matches_naive_with_distinct_value_dim(self):
        torch.manual_seed(0)
        batch, heads, query_len, kv_len = 2, 3, 5, 9
        query = torch.randn(batch, heads, query_len, 6)
        key = torch.randn(batch, heads, kv_len, 6)
        value = torch.randn(batch, heads, kv_len, 4)
        mask = torch.rand(batch, 1, query_len, kv_len) > 0.25
        mask[..., 0] = True
        fake = _FakeAttention(scaling=6**-0.5)

        output = self.attention_cls._dense_attention(
            fake,
            query,
            key,
            value,
            mask,
        )
        expected = _naive_attention(fake, query, key, value, mask)

        self.assertEqual(output.shape, (batch, heads, query_len, 4))
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_fully_masked_rows_are_zero(self):
        query = torch.randn(1, 2, 3, 4)
        key = torch.randn(1, 2, 6, 4)
        value = torch.randn(1, 2, 6, 3)
        mask = torch.ones(1, 1, 3, 6, dtype=torch.bool)
        mask[..., 1, :] = False
        fake = _FakeAttention(scaling=0.5)

        output = self.attention_cls._dense_attention(
            fake,
            query,
            key,
            value,
            mask,
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[..., 1, :].abs().max().item(), 0.0)


class DeepseekV3DSparkModelTest(unittest.TestCase):
    def setUp(self):
        _, self.model_cls = _model_classes()

    def test_sliding_window_zero_disables_context_clipping(self):
        model = self.model_cls(_tiny_config(sliding_window=0))
        self.assertIsNone(model.context_window)

        anchor = torch.tensor([[4]])
        keep = torch.ones(1, 1, dtype=torch.bool)
        full = create_dflash_sdpa_mask(
            anchor,
            keep,
            S=6,
            block_size=2,
            device=anchor.device,
            context_window=model.context_window,
            include_anchor_context=model.include_anchor_context,
        )
        self.assertTrue(full[0, 0, 0, :5].all())

    def test_positive_sliding_window_clips_context(self):
        model = self.model_cls(_tiny_config(sliding_window=2))
        self.assertEqual(model.context_window, 2)

        anchor = torch.tensor([[4]])
        keep = torch.ones(1, 1, dtype=torch.bool)
        windowed = create_dflash_sdpa_mask(
            anchor,
            keep,
            S=6,
            block_size=2,
            device=anchor.device,
            context_window=model.context_window,
            include_anchor_context=model.include_anchor_context,
        )
        self.assertFalse(windowed[0, 0, 0, :3].any())
        self.assertTrue(windowed[0, 0, 0, 3:5].all())

    def test_dense_mlp_can_replace_moe_in_every_stage(self):
        model = self.model_cls(
            _tiny_config(sliding_window=0, mlp_type="dense")
        )
        self.assertTrue(
            all(
                stage.mlp.__class__.__name__ == "DeepseekV3DSparkMLP"
                for stage in model.mtp
            )
        )
        self.assertTrue(
            all(not hasattr(stage.mlp, "gate") for stage in model.mtp)
        )

    def test_invalid_mlp_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mlp_type"):
            self.model_cls(
                _tiny_config(sliding_window=0, mlp_type="unsupported")
            )

    def test_full_forward_backward_and_dspark_heads(self):
        torch.manual_seed(3)
        model = self.model_cls(_tiny_config(sliding_window=2))
        batch, context_len, query_len = 1, 4, 2
        target_hidden = torch.randn(
            batch,
            context_len,
            2 * model.config.hidden_size,
            requires_grad=True,
        )
        noise_embedding = torch.randn(
            batch,
            query_len,
            model.config.hidden_size,
            requires_grad=True,
        )
        position_ids = torch.tensor([[0, 1, 2, 3, 2, 3]])
        attention_mask = torch.ones(
            batch,
            1,
            query_len,
            context_len + query_len,
            dtype=torch.bool,
        )

        hidden_states = model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        self.assertEqual(
            hidden_states.shape,
            (batch, query_len, model.config.hidden_size),
        )
        self.assertTrue(torch.isfinite(hidden_states).all())

        prev_token_ids = torch.tensor([[1, 2]])
        base_logits = torch.randn(batch, query_len, model.config.vocab_size)
        corrected = model.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )
        confidence = model.predict_confidence(
            hidden_states,
            prev_token_ids=prev_token_ids,
        )
        self.assertEqual(corrected.shape, base_logits.shape)
        self.assertEqual(confidence.shape, (batch, query_len))

        (hidden_states.sum() + confidence.sum()).backward()
        self.assertIsNotNone(noise_embedding.grad)
        self.assertIsNotNone(target_hidden.grad)
        self.assertTrue(torch.isfinite(noise_embedding.grad).all())
        self.assertTrue(torch.isfinite(target_hidden.grad).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
