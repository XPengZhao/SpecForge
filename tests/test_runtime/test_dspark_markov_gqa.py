"""Behavioral tests for independent backbone / context-only Markov GQA."""

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import DSparkDraftModel


def config(head_type="gqa", backend="sdpa"):
    c = Qwen3Config(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=23,
        attention_bias=False,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
    )
    c.block_size = 3
    c.num_target_layers = 4
    c._attn_implementation = backend
    c.dflash_config = {
        "target_layer_ids": [0, 2],
        "projector_type": "dspark",
        "mask_token_id": 22,
        "markov_rank": 4,
        "markov_head_type": head_type,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_gqa": {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    }
    return c


def inputs():
    torch.manual_seed(17)
    features = torch.randn(2, 8, 16, requires_grad=True)
    anchors = torch.tensor([[0, 3, 6], [2, 4, 5]])
    keep = torch.tensor([[True, True, True], [True, False, True]])
    positions = anchors[..., None] + torch.arange(3)
    tokens = torch.randint(23, (2, 3, 3))
    return features, anchors, keep, positions, tokens


@pytest.mark.parametrize("backend", ["sdpa", "flex_attention"])
def test_head_masks_future_invalid_and_padding_and_has_no_residual(backend):
    if backend == "flex_attention" and not torch.cuda.is_available():
        pytest.skip("Flex Attention backward requires CUDA")
    device = "cuda" if backend == "flex_attention" else "cpu"
    head = DSparkDraftModel(config()).markov_head
    head = head.to(device)
    features, anchors, keep, positions, tokens = (x.to(device) for x in inputs())
    features = features.detach().requires_grad_()
    valid = torch.ones(2, 8, dtype=torch.bool, device=device)
    valid[:, 1] = False

    def run(x):
        return head.compute_latents(
            tokens,
            context=head.prepare_context(x, context_valid_mask=valid),
            query_position_ids=positions,
            anchor_positions=anchors,
            block_keep_mask=keep,
            attention_backend=backend,
        )

    out = run(features)
    assert torch.equal(out[0, 0], torch.zeros_like(out[0, 0]))
    assert torch.equal(out[1, 1], torch.zeros_like(out[1, 1]))
    changed = features.detach().clone()
    changed[:, 6:] += 100
    changed[:, 1] -= 100
    torch.testing.assert_close(run(changed), out)
    out[0, 1].square().sum().backward()
    assert features.grad[0, :3].abs().sum() > 0
    assert features.grad[0, 3:].count_nonzero() == 0
    assert features.grad[0, 1].count_nonzero() == 0
    with torch.no_grad():
        head.o_proj.weight.zero_()
    assert run(features).count_nonzero() == 0  # no W1 residual bypass


@pytest.mark.parametrize("backend", ["sdpa", "eager", "flex_attention"])
@torch.no_grad()
def test_head_parallel_matches_sequential_and_sdpa(backend):
    head = DSparkDraftModel(config()).markov_head.eval()
    features, anchors, keep, positions, tokens = inputs()
    context = head.prepare_context(features)
    kwargs = {
        "context": context,
        "query_position_ids": positions,
        "anchor_positions": anchors,
        "block_keep_mask": keep,
    }
    reference = head.compute_latents(tokens, **kwargs, attention_backend="sdpa")
    actual = head.compute_latents(tokens, **kwargs, attention_backend=backend)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    base = torch.randn(2, 3, 3, 23)
    sampled, sequential = head.sample_block_tokens(
        base, first_prev_token_ids=tokens[..., 0], **kwargs, attention_backend=backend
    )
    prev = torch.cat((tokens[..., :1], sampled[..., :-1]), dim=-1)
    parallel = head.apply_block_logits(
        base, token_ids=prev, **kwargs, attention_backend=backend
    )
    torch.testing.assert_close(parallel, sequential, atol=2e-6, rtol=2e-5)


def test_empty_context_and_empty_proposal():
    head = DSparkDraftModel(config()).markov_head
    features, anchors, keep, positions, tokens = inputs()
    context = head.prepare_context(features[:, :0])
    out = head.compute_latents(
        tokens,
        context=context,
        query_position_ids=positions,
        anchor_positions=anchors,
        block_keep_mask=keep,
    )
    assert out.count_nonzero() == 0
    out.sum().backward()
    assert all(
        p.grad is not None for n, p in head.named_parameters() if "markov_w2" not in n
    )
    sampled, logits = head.sample_block_tokens(
        torch.empty(2, 3, 0, 23),
        first_prev_token_ids=tokens[..., 0],
        context=context,
        query_position_ids=positions[..., :0],
        anchor_positions=anchors,
    )
    assert sampled.shape == (2, 3, 0) and logits.shape == (2, 3, 0, 23)


def training_model(draft):
    c = draft.config
    return OnlineDSparkModel(
        draft,
        torch.nn.Linear(c.hidden_size, c.vocab_size, bias=False).requires_grad_(False),
        torch.nn.Embedding(c.vocab_size, c.hidden_size).requires_grad_(False),
        mask_token_id=22,
        block_size=3,
        attention_backend="sdpa",
        num_anchors=3,
    )


def test_online_forward_gradients_optimizer_and_checkpoint(tmp_path):
    torch.manual_seed(2)
    draft = DSparkDraftModel(config())
    model = training_model(draft)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    ids = torch.randint(22, (2, 10))
    features = torch.randn(2, 10, 32)
    final = torch.randn(2, 10, 16)
    fused = []
    hook = draft.fc.register_forward_hook(lambda m, a, y: fused.append(y))
    loss, accuracy, _metrics = model(ids, features, torch.ones_like(ids), final)
    hook.remove()
    assert len(fused) == 1
    assert torch.isfinite(loss) and torch.isfinite(accuracy)
    loss.backward()
    for name, parameter in draft.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert draft.fc.weight.grad.abs().sum() > 0
    assert draft.markov_head.k_norm.weight.grad.abs().sum() > 0
    assert draft.markov_head.q_proj.weight.grad.abs().sum() > 0
    assert (
        draft.markov_head.k_proj.weight is not draft.layers[0].self_attn.k_proj.weight
    )
    optimizer.step()
    draft.save_pretrained(tmp_path)
    restored = DSparkDraftModel.from_pretrained(tmp_path, attn_implementation="sdpa")
    assert restored.requires_markov_context
    for name, parameter in draft.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(
        torch.load(tmp_path / "optimizer.pt", weights_only=True)
    )
    restored_model = training_model(restored)
    restored_model.lm_head.load_state_dict(model.lm_head.state_dict())
    restored_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
    for current, opt in ((model, optimizer), (restored_model, resumed_optimizer)):
        torch.manual_seed(9)
        opt.zero_grad()
        current(ids, features, torch.ones_like(ids), final)[0].backward()
        opt.step()
    for name, parameter in draft.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])


def test_production_parameter_count():
    root = Path(__file__).resolve().parents[2]
    c = Qwen3Config(
        **json.loads((root / "configs/qwen3-4b-dspark-markov-gqa.json").read_text())
    )
    c._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = DSparkDraftModel(c)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 622_561_537
    head = model.markov_head
    assert (
        sum(
            p.numel()
            for n, p in head.named_parameters()
            if not n.startswith("markov_w")
        )
        == 7_340_288
    )


@torch.no_grad()
def test_reference_generation_uses_head_and_matches_target_greedy():
    from unittest.mock import patch

    from transformers import Qwen3ForCausalLM

    torch.manual_seed(12)
    draft = DSparkDraftModel(config()).eval()
    tc = Qwen3Config(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=23,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 4,
    )
    tc._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(tc).eval()
    # Keep the reserved draft mask token outside the target sampling space.
    target.lm_head.register_forward_hook(
        lambda m, a, logits: logits.index_fill(-1, torch.tensor([22]), -torch.inf)
    )
    prompt = torch.tensor([[1, 4, 7, 2]])
    expected = prompt
    for _ in range(7):
        nxt = target(expected).logits[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, nxt), dim=-1)
    with patch.object(
        draft.markov_head,
        "sample_block_tokens",
        wraps=draft.markov_head.sample_block_tokens,
    ) as sample_head:
        actual = draft.spec_generate(
            target, prompt, max_new_tokens=7, stop_token_ids=None, temperature=0.0
        )
    assert sample_head.call_count > 1
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("head", ["vanilla", "gqa"])
def test_head_switches_and_gradient_checkpointing(head):
    c = config(head_type=head)
    draft = DSparkDraftModel(c)
    draft.gradient_checkpointing_enable()
    model = training_model(draft).train()
    ids = torch.randint(22, (2, 9))
    loss = model(
        ids, torch.randn(2, 9, 32), torch.ones_like(ids), torch.randn(2, 9, 16)
    )[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Compiled Flex forward/backward requires CUDA"
)
def test_cuda_flex_online_forward_backward():
    torch.manual_seed(7)
    draft = DSparkDraftModel(config(backend="flex_attention")).cuda().bfloat16()
    model = training_model(draft).cuda().bfloat16()
    model.attention_backend = "flex_attention"
    ids = torch.randint(22, (2, 10), device="cuda")
    loss = model(
        ids,
        torch.randn(2, 10, 32, device="cuda", dtype=torch.bfloat16),
        torch.ones_like(ids),
        torch.randn(2, 10, 16, device="cuda", dtype=torch.bfloat16),
    )[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters()
    )


def test_resume_records_gqa_semantics_and_recipe_loads():
    from specforge.algorithms.dspark.providers import resume_contract
    from specforge.config import load_config

    draft = DSparkDraftModel(config())
    contract = resume_contract(None, draft, training_model(draft))
    assert contract["dspark_markov_gqa"]["residual"] is False
    assert contract["dspark_markov_gqa"]["num_kv_heads"] == 1
    root = Path(__file__).resolve().parents[2]
    recipe = load_config(
        str(root / "examples/configs/qwen3-4b-dspark-markov-gqa-online.yaml")
    )
    assert recipe.training.strategy == "dspark"
    assert recipe.model.draft_model_config.endswith("qwen3-4b-dspark-markov-gqa.json")


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@torch.no_grad()
def test_backbone_cannot_see_future_context_or_other_blocks(backend):
    from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask

    torch.manual_seed(11)
    draft = DSparkDraftModel(config(backend=backend)).eval()
    anchors = torch.tensor([[2, 5]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context = torch.randn(1, 8, 32)
    noise = torch.randn(1, 6, 16)
    positions = torch.cat(
        (torch.arange(8)[None], (anchors[..., None] + torch.arange(3)).reshape(1, 6)),
        dim=1,
    )
    mask = create_dflash_sdpa_mask(anchors, keep, 8, 3, "cpu")

    def forward(features, embeddings):
        return draft(
            position_ids=positions,
            target_hidden=features,
            noise_embedding=embeddings,
            attention_mask=mask,
        )

    before = forward(context, noise)
    context[:, 2:] += 200
    noise[:, 3:] -= 200
    after = forward(context, noise)
    torch.testing.assert_close(before[:, :3], after[:, :3])


@torch.no_grad()
def test_cached_backbone_and_head_logits_match_full_context_recomputation():
    from transformers import DynamicCache

    from specforge.modeling.draft.markov_gqa import MarkovGQAContext

    torch.manual_seed(47)
    draft = DSparkDraftModel(config()).eval()
    features = torch.randn(1, 12, 32)
    cache = DynamicCache()
    previous_end = 0
    head_cache = None
    # Unequal accepted lengths exercise cache cropping and absolute RoPE positions.
    for anchor in (3, 4, 7, 9):
        query_positions = torch.arange(anchor, anchor + 3)[None]
        noise = torch.randn(1, 3, 16)
        cached, new_context = draft(
            position_ids=torch.arange(previous_end, anchor + 3)[None],
            target_hidden=features[:, previous_end:anchor],
            noise_embedding=noise,
            past_key_values=cache,
            use_cache=True,
            return_context=True,
        )
        cache.crop(-3)
        full, full_context = draft(
            position_ids=torch.arange(anchor + 3)[None],
            target_hidden=features[:, :anchor],
            noise_embedding=noise,
            return_context=True,
        )
        torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-5)
        new_kv = draft.markov_head.prepare_context(
            new_context, torch.arange(previous_end, anchor)[None]
        )
        if head_cache is None:
            head_cache = new_kv
        else:
            head_cache = MarkovGQAContext(
                torch.cat((head_cache.key, new_kv.key), dim=-2),
                torch.cat((head_cache.value, new_kv.value), dim=-2),
                torch.cat((head_cache.positions, new_kv.positions), dim=-1),
                torch.cat((head_cache.valid, new_kv.valid), dim=-1),
            )
        ids = torch.randint(22, (1, 1, 3))
        base = torch.randn(1, 1, 3, 23)
        kwargs = {
            "token_ids": ids,
            "query_position_ids": query_positions[:, None],
            "anchor_positions": torch.tensor([[anchor]]),
        }
        incremental = draft.markov_head.apply_block_logits(
            base, context=head_cache, **kwargs
        )
        recomputed = draft.markov_head.apply_block_logits(
            base, context=draft.markov_head.prepare_context(full_context), **kwargs
        )
        torch.testing.assert_close(incremental, recomputed, atol=2e-6, rtol=2e-5)
        previous_end = anchor


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_reference_control_sampler_applies_markov_head_at_training_slots(head_type):
    from types import SimpleNamespace
    from unittest.mock import patch

    draft = DSparkDraftModel(config(head_type=head_type)).eval()
    target = SimpleNamespace(lm_head=torch.nn.Linear(16, 23, bias=False))
    hidden = torch.randn(1, 3, 16)
    block_ids = torch.tensor([[4, 22, 22]])
    expected_ids = torch.tensor([[7, 8]])
    with patch.object(
        draft.markov_head, "sample_block_tokens", return_value=(expected_ids, None)
    ) as sampler:
        actual = draft._sample_draft_tokens(target, hidden, block_ids)
    sampler.assert_called_once()
    torch.testing.assert_close(actual, expected_ids)
    torch.testing.assert_close(sampler.call_args.args[0], target.lm_head(hidden[:, :2]))
    torch.testing.assert_close(
        sampler.call_args.kwargs["first_prev_token_ids"], block_ids[:, 0]
    )
    torch.testing.assert_close(sampler.call_args.kwargs["hidden_states"], hidden[:, :2])


@torch.no_grad()
def test_gqa_matches_explicit_attention_with_offset_positions():
    """Independent matmul/softmax checks KV grouping, QK norm and RoPE."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half

    head = DSparkDraftModel(config()).markov_head.eval()
    features, anchors, keep, positions, tokens = inputs()
    context_positions = torch.arange(8)[None].expand(2, -1) + 13
    positions, anchors = positions + 13, anchors + 13
    context = head.prepare_context(features, context_positions)
    assert context.key.shape == (2, 1, 8, 8)
    q = head.q_proj(head.get_prev_embeddings(tokens)).reshape(2, 9, 2, 8)
    q = (
        q
        * torch.rsqrt(q.square().mean(-1, keepdim=True) + head.q_norm.variance_epsilon)
    ) * head.q_norm.weight
    q = q.transpose(1, 2)
    k = head.k_proj(features).reshape(2, 8, 1, 8)
    k = (
        k
        * torch.rsqrt(k.square().mean(-1, keepdim=True) + head.k_norm.variance_epsilon)
    ) * head.k_norm.weight
    k = k.transpose(1, 2)
    cq, sq = head.rotary_emb(features, positions.reshape(2, 9))
    ck, sk = head.rotary_emb(features, context_positions)
    q = q * cq[:, None] + rotate_half(q) * sq[:, None]
    k = k * ck[:, None] + rotate_half(k) * sk[:, None]
    torch.testing.assert_close(k, context.key)
    v = head.v_proj(features).reshape(2, 8, 1, 8).transpose(1, 2)
    k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    allowed = (
        context_positions[:, None, :] < anchors.repeat_interleave(3, 1)[:, :, None]
    )
    allowed &= keep.repeat_interleave(3, 1)[:, :, None]
    scores = (q @ k.transpose(-1, -2)) * (8**-0.5)
    weights = scores.masked_fill(~allowed[:, None], -torch.inf).softmax(-1).nan_to_num()
    out = (weights @ v).transpose(1, 2).reshape(2, 9, 16)
    expected = head.o_proj(out).reshape(2, 3, 3, 4)
    actual = head.compute_latents(
        tokens,
        context=context,
        query_position_ids=positions,
        anchor_positions=anchors,
        block_keep_mask=keep,
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_hf_export_preserves_head_and_rejects_missing_weights(tmp_path):
    from specforge.export.checkpoint_io import materialize_draft
    from specforge.export.to_hf import export_to_hf

    c = config()
    c.architectures = ["DSparkDraftModel"]
    draft = DSparkDraftModel(c).bfloat16()
    config_path = tmp_path / "draft.json"
    c.to_json_file(config_path)
    state = {"draft_state_dict": draft.state_dict()}
    checkpoint = tmp_path / "training_state.pt"
    torch.save(state, checkpoint)
    dest = tmp_path / "hf"
    export_to_hf(str(checkpoint), str(config_path), str(dest))
    restored = DSparkDraftModel.from_pretrained(dest, attn_implementation="sdpa")
    assert restored.markov_head.num_kv_heads == 1
    for name, weight in draft.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name].to(weight.dtype), weight)
    del state["draft_state_dict"]["markov_head.k_norm.weight"]
    with pytest.raises(ValueError, match="markov_head.k_norm.weight"):
        materialize_draft(state, str(config_path))


@torch.no_grad()
def test_gqa_head_keeps_baseline_backbone_and_confidence_unchanged():
    baseline = DSparkDraftModel(config(head_type="vanilla")).eval()
    candidate = DSparkDraftModel(config()).eval()
    missing, unexpected = candidate.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("markov_head.") for k in missing)
    assert type(candidate.layers[0].self_attn) is type(baseline.layers[0].self_attn)
    features = torch.randn(2, 5, 32)
    noise = torch.randn(2, 3, 16)
    kwargs = {
        "target_hidden": features,
        "noise_embedding": noise,
        "position_ids": torch.arange(8)[None].expand(2, -1),
    }
    reference = baseline(**kwargs)
    actual, context = candidate(**kwargs, return_context=True)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(context, candidate.hidden_norm(candidate.fc(features)))
    tokens = torch.randint(23, (2, 3))
    torch.testing.assert_close(
        candidate.predict_confidence(actual, prev_token_ids=tokens),
        baseline.predict_confidence(reference, prev_token_ids=tokens),
    )
