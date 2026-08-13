# coding=utf-8
"""GLM-5.2 MLA+SWA DSpark attention and draft-model tests."""

import tempfile
import unittest

import torch

from specforge.algorithms.common.dflash_family_model import (
    create_dflash_sdpa_mask,
)


def _model_classes():
    from specforge.modeling.draft.glm52_dspark import (
        Glm52DSparkConfig,
        Glm52DSparkDraftModel,
    )

    return Glm52DSparkConfig, Glm52DSparkDraftModel


def _attention_symbols():
    from specforge.modeling.draft.glm52_dspark_attention import (
        Glm52DSparkAttention,
        apply_interleaved_rope,
    )

    return Glm52DSparkAttention, apply_interleaved_rope


class _FakeAttention:
    def __init__(
        self,
        *,
        scaling: float,
        block_size: int,
        value_dim: int,
        chunk_size: int = 2,
    ):
        self.scaling = scaling
        self.block_size = block_size
        self.v_head_dim = value_dim
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


def _tiny_config(*, sliding_window: int = 2, mlp_type: str = "dense"):
    config_cls, _ = _model_classes()
    return config_cls(
        architectures=["Glm52DSparkDraftModel"],
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
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
        dtype="float32",
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


class Glm52DSparkAttentionTest(unittest.TestCase):
    def setUp(self):
        self.attention_cls, self.apply_rope = _attention_symbols()

    def test_interleaved_rope_rotates_adjacent_pairs(self):
        hidden = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        cos = torch.zeros(1, 1, 4)
        sin = torch.ones(1, 1, 4)

        rotated = self.apply_rope(hidden, cos, sin)

        torch.testing.assert_close(
            rotated,
            torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]),
        )

    def test_split_dense_attention_matches_naive_with_distinct_value_dim(self):
        torch.manual_seed(0)
        batch, heads, context_len, query_len = 2, 3, 4, 4
        key_dim, value_dim, block_size = 6, 4, 2
        query = torch.randn(batch, heads, query_len, key_dim)
        key = torch.randn(
            batch,
            heads,
            context_len + query_len,
            key_dim,
        )
        value = torch.randn(
            batch,
            heads,
            context_len + query_len,
            value_dim,
        )
        anchors = torch.tensor([[1, 3], [2, 3]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        mask = create_dflash_sdpa_mask(
            anchors,
            keep,
            S=context_len,
            block_size=block_size,
            device=anchors.device,
            context_window=2,
            include_anchor_context=True,
        )
        fake = _FakeAttention(
            scaling=key_dim**-0.5,
            block_size=block_size,
            value_dim=value_dim,
        )

        output = self.attention_cls._dense_attention(
            fake,
            query,
            key,
            value,
            mask,
        )
        expected = _naive_attention(fake, query, key, value, mask)

        self.assertEqual(output.shape, (batch, heads, query_len, value_dim))
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_fully_masked_rows_are_zero(self):
        query = torch.randn(1, 2, 4, 4)
        key = torch.randn(1, 2, 8, 4)
        value = torch.randn(1, 2, 8, 3)
        anchors = torch.tensor([[1, 3]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        mask = create_dflash_sdpa_mask(
            anchors,
            keep,
            S=4,
            block_size=2,
            device=anchors.device,
            context_window=2,
            include_anchor_context=True,
        )
        mask[..., 1, :] = False
        fake = _FakeAttention(
            scaling=0.5,
            block_size=2,
            value_dim=3,
        )

        output = self.attention_cls._dense_attention(
            fake,
            query,
            key,
            value,
            mask,
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[..., 1, :].abs().max().item(), 0.0)


class Glm52DSparkModelTest(unittest.TestCase):
    def setUp(self):
        _, self.model_cls = _model_classes()

    def test_fixed_sliding_window_clips_context(self):
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

    def test_every_stage_uses_dense_mlp(self):
        model = self.model_cls(_tiny_config())
        self.assertEqual(len(model.mtp), 2)
        self.assertTrue(
            all(
                stage.mlp.__class__.__name__ == "Glm52DSparkMLP"
                for stage in model.mtp
            )
        )
        self.assertEqual(model._no_split_modules, ["Glm52DSparkMLP"])

    def test_non_dense_mlp_and_zero_window_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mlp_type"):
            _tiny_config(mlp_type="moe")
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            _tiny_config(sliding_window=0)

    def test_full_forward_backward_and_dspark_heads(self):
        torch.manual_seed(3)
        model = self.model_cls(_tiny_config())
        batch, context_len, query_len = 1, 4, 4
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
        anchors = torch.tensor([[1, 3]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        attention_mask = create_dflash_sdpa_mask(
            anchors,
            keep,
            S=context_len,
            block_size=2,
            device=anchors.device,
            context_window=model.context_window,
            include_anchor_context=True,
        )
        position_ids = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4]])

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

        prev_token_ids = torch.tensor([[1, 2, 3, 4]])
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

    def test_scratch_optimizer_state_round_trips_for_resume(self):
        torch.manual_seed(11)
        model = self.model_cls(_tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step(current_model, current_optimizer):
            current_optimizer.zero_grad()
            loss = sum(
                parameter.float().square().mean()
                for parameter in current_model.parameters()
            )
            loss.backward()
            current_optimizer.step()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = f"{directory}/checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint,
            )
            resumed = self.model_cls(_tiny_config())
            resumed_optimizer = torch.optim.AdamW(
                resumed.parameters(),
                lr=1e-3,
            )
            state = torch.load(checkpoint, weights_only=True)
            resumed.load_state_dict(state["model"])
            resumed_optimizer.load_state_dict(state["optimizer"])

        step(model, optimizer)
        step(resumed, resumed_optimizer)
        for expected, actual in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
