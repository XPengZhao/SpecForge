"""Reference transitions, memory isolation and real three-layer gradient flow."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    OnlineDSparkModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.algorithms.common.dspark_memory import (
    OnlineDSparkMemoryModel,
    align_draft_memory,
)
from specforge.modeling.draft.dspark import DSparkDraftModel


def config(memory=True, head="vanilla"):
    cfg = Qwen3Config(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
        layer_types=["full_attention"] * 3,
        architectures=["DSparkDraftModel"],
    )
    cfg.num_target_layers = 3
    cfg.block_size = 4
    cfg.dflash_config = dict(
        target_layer_ids=[0, 1, 2],
        mask_token_id=0,
        projector_type="dspark",
        markov_rank=4,
        markov_head_type=head,
        draft_memory=memory,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def model(head="vanilla", keep_prob=1.0):
    torch.manual_seed(42)
    draft = DSparkDraftModel(config(head=head))
    lm = nn.Linear(16, 19, bias=False).requires_grad_(False)
    embed = nn.Embedding(19, 16).requires_grad_(False)
    return OnlineDSparkMemoryModel(
        draft_model=draft,
        target_lm_head=lm,
        target_embed_tokens=embed,
        mask_token_id=0,
        block_size=4,
        num_anchors=2,
        attention_backend="sdpa",
        dspark_ce_loss_alpha=1.0,
        dspark_l1_loss_alpha=0.0,
        dspark_confidence_head_alpha=0.0,
        draft_memory_keep_prob=keep_prob,
        draft_memory_head_chunk_size=3,
    )


def test_alignment_first_middle_last_full_and_truncated():
    hidden = torch.arange(6 * 4.0).reshape(1, 6, 4, 1).requires_grad_()
    anchors = torch.tensor([[2, 2, 2, 2, 9, 2]])
    accepted = torch.tensor([[0, 1, 3, 4, 1, 2]])
    valid = torch.ones(1, 6, 4, dtype=torch.bool)
    valid[0, 4, 1:] = False
    valid[0, 5, 2:] = False
    out = align_draft_memory(
        hidden, anchors, accepted, valid, torch.ones(1, 6, dtype=torch.bool), 12
    )
    new, keep, mem, positions, mask = out
    assert new.tolist() == [[3, 4, 6, 7, 0, 0]]
    assert keep.tolist() == [[True, True, True, True, False, False]]
    assert mask.sum(-1).tolist() == [[3, 2, 0, 0, 0, 0]]
    assert positions[0, 0].tolist() == [4, 5, 6]
    assert positions[0, 1, :2].tolist() == [5, 6]
    assert mem[0, 0, :, 0].tolist() == [1, 2, 3]
    assert mem[0, 1, :2, 0].tolist() == [6, 7]
    assert not mem.requires_grad


def test_memory_mask_isolation_and_flex_equivalence():
    anchors = torch.tensor([[2, 5]])
    keep = torch.tensor([[True, True]])
    memory_keep = torch.tensor([[[True, False, False], [True, True, False]]])
    kwargs = dict(
        anchor_positions=anchors,
        block_keep_mask=keep,
        S=9,
        block_size=4,
        device=torch.device("cpu"),
        memory_keep=memory_keep,
    )
    dense = create_dflash_sdpa_mask(**kwargs)
    assert dense.shape == (1, 1, 8, 23)
    assert torch.where(dense[0, 0, 0])[0].tolist() == [0, 1, 9, 15, 16, 17, 18]
    assert torch.where(dense[0, 0, 4])[0].tolist() == [
        0,
        1,
        2,
        3,
        4,
        12,
        13,
        19,
        20,
        21,
        22,
    ]
    flex = create_dflash_block_mask(**kwargs)
    q = torch.arange(8)[:, None]
    kv = torch.arange(23)[None, :]
    actual = flex.mask_mod(torch.tensor(0), torch.tensor(0), q, kv)
    torch.testing.assert_close(actual, dense[0, 0])


@pytest.mark.parametrize("head", ["vanilla", "gated"])
def test_parallel_first_mismatch_equals_sequential_greedy(head):
    m = model(head)
    hidden = torch.randn(2, 3, 4, 16)
    targets = torch.randint(0, 19, (2, 3, 4))
    first = torch.randint(0, 19, (2, 3))
    # Construct references with exactly k matching tokens, including full acceptance.
    with torch.no_grad():
        base = m.lm_head(hidden)
        prev = first
        generated = []
        for j in range(4):
            logits = m.draft_model.apply_logits_head(
                base[:, :, j : j + 1],
                prev_token_ids=prev.unsqueeze(-1),
                hidden_states=hidden[:, :, j : j + 1],
            )
            prev = logits.argmax(-1).squeeze(-1)
            generated.append(prev)
        generated = torch.stack(generated, -1)
        for b in range(2):
            for n in range(3):
                k = (b * 3 + n) % 5
                targets[b, n, :k] = generated[b, n, :k]
                if k < 4:
                    targets[b, n, k] = (generated[b, n, k] + 1) % 19
        previous = torch.cat([first.unsqueeze(-1), targets[..., :-1]], -1)
        parallel = m._reference_predictions(hidden, previous)
    first_match = lambda x: (x == targets).long().cumprod(-1).sum(-1)
    torch.testing.assert_close(first_match(parallel), first_match(generated))


def test_two_pass_real_backbone_all_layers_and_detached_memory():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    aux = torch.randn(1, 12, 48)
    loss_mask = torch.ones_like(ids)
    calls, contexts = [], []

    def record(_module, _args, kwargs, output):
        calls.append((torch.is_grad_enabled(), kwargs, output))

    hooks = [m.draft_model.register_forward_hook(record, with_kwargs=True)]
    for layer in m.draft_model.layers:
        hooks.append(
            layer.self_attn.register_forward_pre_hook(
                lambda _m, _a, kw: contexts.append(kw["target_hidden"].shape[1]),
                with_kwargs=True,
            )
        )
    # Force first rejection: this test isolates memory alignment/gradient routing.
    anchors = torch.tensor([[1, 5]])
    with (
        patch.object(
            m,
            "_sample_anchor_positions",
            return_value=(anchors, torch.ones_like(anchors, dtype=torch.bool)),
        ),
        patch.object(
            m,
            "_reference_predictions",
            return_value=torch.zeros(1, 2, 4, dtype=torch.long),
        ),
    ):
        loss, _, metrics = m(ids, aux, loss_mask)
    loss.backward()
    for hook in hooks:
        hook.remove()
    assert torch.isfinite(loss)
    assert [c[0] for c in calls] == [False, True]
    assert not calls[0][2].requires_grad
    assert not calls[1][1]["draft_memory"].requires_grad
    assert contexts == [12, 12, 12, 18, 18, 18]
    assert metrics["memory_rows"].item() == 3
    grad = m.draft_model.draft_memory_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for layer in m.draft_model.layers:
        assert layer.self_attn.k_proj.weight.grad.abs().sum() > 0


def test_masked_memory_equals_baseline_at_same_anchor():
    m = model().eval()
    ids = torch.arange(12).unsqueeze(0)
    aux = torch.randn(1, 12, 48)
    anchors, keep = torch.tensor([[2, 5]]), torch.tensor([[True, True]])
    args = (ids, aux, torch.ones_like(ids), anchors, keep)
    plain = OnlineDSparkModel._forward_draft_blocks(m, *args)[2]
    masked = OnlineDSparkModel._forward_draft_blocks(
        m,
        *args,
        draft_memory=torch.randn(1, 2, 3, 16),
        memory_positions=torch.tensor([[[3, 4, 5], [6, 7, 8]]]),
        memory_keep=torch.zeros(1, 2, 3, dtype=torch.bool),
    )[2]
    torch.testing.assert_close(plain, masked, atol=1e-6, rtol=1e-5)


def test_no_future_target_or_other_anchor_memory_leak():
    m = model().eval()
    ids, aux = torch.arange(12).unsqueeze(0), torch.randn(1, 12, 48)
    anchors, keep = torch.tensor([[2, 6]]), torch.tensor([[True, True]])
    memory = torch.randn(1, 2, 3, 16)
    kwargs = dict(
        draft_memory=memory,
        memory_positions=torch.tensor([[[3, 4, 5], [7, 8, 9]]]),
        memory_keep=torch.ones(1, 2, 3, dtype=torch.bool),
    )
    run = lambda h: OnlineDSparkModel._forward_draft_blocks(
        m, ids, h, torch.ones_like(ids), anchors, keep, **kwargs
    )[2][:, :4]
    original = run(aux)
    changed = aux.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 10
    memory[:, 1] = torch.randn_like(memory[:, 1]) * 10
    torch.testing.assert_close(original, run(changed), atol=1e-6, rtol=1e-5)


def test_empty_response_has_finite_zero_loss_and_backward():
    m = model()
    ids = torch.arange(12).unsqueeze(0)
    loss, _, metrics = m(ids, torch.randn(1, 12, 48), torch.zeros_like(ids))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert metrics["memory_active_fraction"].item() == 0
    assert torch.isfinite(m.draft_model.draft_memory_proj.weight.grad).all()


def test_warm_start_baseline_and_strict_partial_adapter(tmp_path):
    from specforge.training.model_loading import warm_start_draft_model

    baseline = DSparkDraftModel(config(memory=False))
    assert not any(key.startswith("draft_memory_") for key in baseline.state_dict())
    state = dict(baseline.state_dict())
    checkpoint = tmp_path / "training_state.pt"
    torch.save(dict(strategy="dspark", draft_state_dict=state), checkpoint)
    memory_model = DSparkDraftModel(config())
    # Simulate uninitialized new tensors under a no_init_weights construction.
    with torch.no_grad():
        memory_model.draft_memory_proj.weight.fill_(float("nan"))
        memory_model.draft_memory_norm.weight.fill_(float("nan"))
    warm_start_draft_model(
        memory_model, str(checkpoint), draft_config=config(), strategy="dspark"
    )
    assert torch.isfinite(memory_model.draft_memory_proj.weight).all()
    torch.testing.assert_close(memory_model.draft_memory_norm.weight, torch.ones(16))
    for key, tensor in state.items():
        torch.testing.assert_close(memory_model.state_dict()[key], tensor)
    state["draft_memory_proj.weight"] = memory_model.draft_memory_proj.weight.detach()
    torch.save(dict(strategy="dspark", draft_state_dict=state), checkpoint)
    with pytest.raises(ValueError, match="missing draft weights"):
        warm_start_draft_model(
            memory_model, str(checkpoint), draft_config=config(), strategy="dspark"
        )


def test_memory_checkpoint_roundtrip_and_wrong_export_config(tmp_path):
    from specforge.export.checkpoint_io import materialize_draft

    draft = DSparkDraftModel(config())
    state = dict(draft_state_dict=draft.state_dict())
    path = tmp_path / "memory.json"
    config().to_json_file(path)
    restored = materialize_draft(state, str(path))
    assert restored.draft_memory_enabled
    for key, tensor in draft.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[key], tensor.to(torch.bfloat16)
        )
    config(memory=False).to_json_file(path)
    with pytest.raises(ValueError, match="architecture does not have"):
        materialize_draft(state, str(path))


def test_server_config_resolves_memory_architecture_and_resume_contract():
    import yaml

    from specforge.algorithms.dspark.providers import (
        apply_draft_overrides,
        resume_contract,
    )
    from specforge.config import Config

    root = Path(__file__).resolve().parents[2]
    path = root / "examples/configs/qwen3.8-flash-next-dspark-draft-memory-offline.yaml"
    run = Config.model_validate(yaml.safe_load(path.read_text()))
    cfg = Qwen3Config.from_json_file(root / run.model.draft_model_config)
    apply_draft_overrides(run, cfg)
    assert cfg.dflash_config["draft_memory"] is True
    m = model()
    contract = resume_contract(run, m.draft_model, m)
    assert (
        contract["dspark_draft_memory_version"]
        == "final_hidden_prediction_position_all_layers_v1"
    )
    run.training.dspark_draft_memory = False
    with pytest.raises(ValueError, match="requires training.dspark_draft_memory"):
        apply_draft_overrides(run, cfg)
