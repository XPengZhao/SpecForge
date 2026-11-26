"""DFlash mask alignment and the DSpark objective, through the public strategy."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.test_utils.test_dflash_losses import _dflash_module
from specforge.training.strategies.base import DFlashTrainStrategy
from specforge.training.metric_window import MetricWindow


@pytest.mark.parametrize("mode", ["original", "kl"])
@pytest.mark.parametrize("recompute", [False, True])
def test_distillation_matches_reference_and_has_no_anchor_gradient(mode, recompute):
    torch.manual_seed(8)
    batch, blocks, block_size, vocab, sequence = 2, 2, 4, 6, 7
    draft = nn.Module()
    draft.outputs = nn.Parameter(torch.randn(batch, blocks * block_size, vocab))
    model = _dflash_module.OnlineDFlashDistillationModel(
        draft,
        nn.Identity(),
        nn.Embedding(vocab, vocab),
        0,
        block_size=block_size,
        attention_backend="sdpa",
        loss_decay_gamma=2.0,
        dspark_loss_mode=mode,
        dspark_confidence_head_alpha=0.0,
        recompute_loss=recompute,
    )
    anchors = torch.tensor([[0, 4], [1, 3]])
    keep = torch.tensor([[True, True], [True, False]])

    def blocks_forward(self, **kwargs):
        return anchors, keep, self.draft_model.outputs

    model._forward_draft_blocks = MethodType(blocks_forward, model)
    ids = torch.randint(vocab, (batch, sequence))
    masks = torch.ones(batch, sequence)
    masks[1, 3] = 0  # Stop the valid prefix, even if subsequent mask entries are 1.
    teacher = torch.randn(batch, sequence, vocab, requires_grad=True)
    data = SimpleNamespace(
        tensors=dict(
            input_ids=ids,
            loss_mask=masks,
            hidden_states=torch.zeros(batch, sequence, 2),
            target_last_hidden_states=teacher,
        )
    )
    strategy = DFlashTrainStrategy(model)
    out = strategy.forward_loss(data)

    logits = draft.outputs.view(batch, blocks, block_size, vocab)[..., 1:, :]
    indices = anchors[..., None] + torch.arange(1, block_size)
    safe = indices.clamp(max=sequence - 1)
    labels = ids[:, None].expand(-1, blocks, -1).gather(2, safe)
    valid = (indices < sequence) & keep[..., None]
    valid &= masks[:, None].expand(-1, blocks, -1).gather(2, safe) > 0.5
    valid = valid.int().cumprod(-1).bool()
    weights = valid * torch.exp(-torch.arange(block_size - 1) / 2.0)
    target_probs = teacher.detach().softmax(-1)[
        torch.arange(batch)[:, None, None], safe - 1
    ]
    ce = F.cross_entropy(
        logits.reshape(-1, vocab), labels.reshape(-1), reduction="none"
    ).view_as(weights)
    l1 = (logits.softmax(-1) - target_probs).abs().sum(-1)
    kl = F.kl_div(logits.log_softmax(-1), target_probs, reduction="none").sum(-1)
    expected = (
        (0.1 * ce + 0.9 * l1 if mode == "original" else kl) * weights
    ).sum() / weights.sum()
    torch.testing.assert_close(out.loss, expected)
    (actual_grad,) = torch.autograd.grad(out.loss, draft.outputs, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, draft.outputs)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert (
        actual_grad.view(batch, blocks, block_size, vocab)[..., 0, :].count_nonzero()
        == 0
    )
    assert teacher.grad is None
    window = MetricWindow()
    window.update(out.metrics["log_window"])
    summary = window.summary()
    assert summary["loss_weighted"] == pytest.approx(expected.item(), abs=1e-6)
    assert "accept_rate@0" in summary and "accept_rate@2" in summary
    assert "accept_rate@3" not in summary
    assert "tau_probabilistic" in summary
    assert summary["accuracy_denom"] == valid.sum().item()
    model.eval()
    eval_out = strategy.forward_loss(data)
    torch.testing.assert_close(eval_out.loss, out.loss)
    assert len(eval_out.metrics["acc_corrects"]) == block_size - 1
    assert "eval_metric_sums" in eval_out.metrics
    assert "log_window" not in eval_out.metrics
    del data.tensors["target_last_hidden_states"]
    with pytest.raises(ValueError, match="target_last_hidden_states"):
        strategy.forward_loss(data)


def test_factory_and_resume_contract_preserve_distillation_settings(monkeypatch):
    from specforge.algorithms import model_providers
    from specforge.algorithms.dflash.providers import resume_contract
    from specforge.config import TrainingConfig

    draft = nn.Module()
    draft.block_size = 4
    draft.config = SimpleNamespace(num_hidden_layers=2)
    draft.target_layer_ids = [1, 5]
    cfg = SimpleNamespace(
        training=TrainingConfig(
            strategy="dflash",
            loss_type="dspark",
            dspark_ce_loss_alpha=0.2,
            dspark_l1_loss_alpha=0.8,
            recompute_loss=True,
        )
    )

    def build(_cfg, model, tokenizer, factory, **kwargs):
        return factory(
            dict(
                draft_model=model,
                target_lm_head=nn.Identity(),
                target_embed_tokens=nn.Embedding(6, 6),
                mask_token_id=0,
                block_size=4,
                attention_backend="sdpa",
            )
        )

    monkeypatch.setattr(model_providers, "_build_dflash_family_model", build)
    model = model_providers.build_dflash_model(cfg, draft, None, None, None)
    assert model.loss_type == "dspark"
    assert model.dspark_ce_loss_alpha == 0.2
    assert model.dspark_l1_loss_alpha == 0.8
    assert model.dspark_confidence_head_alpha == 0.0
    assert model.recompute_loss
    contract = resume_contract(cfg, draft, model)
    assert contract["dflash_loss_type"] == "dspark"
    assert contract["dflash_l1_loss_alpha"] == 0.8
    assert contract["dflash_distillation_mode"] == "original"


@pytest.mark.parametrize("distillation", [False, True])
def test_real_ddp_forwards_teacher_features(tmp_path, distillation):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tests.test_optimizer.test_dflash_family_muon import online_dflash

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/init", rank=0, world_size=1,
    )
    try:
        model = online_dflash(distillation)
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        assert not hasattr(wrapped, "loss_type")
        strategy = DFlashTrainStrategy(wrapped)
        assert ("target_last_hidden_states" in strategy.required_features) == distillation
        assert strategy.trainable_module() is wrapped
        tensors = {
            "input_ids": torch.randint(0, 31, (1, 12)),
            "hidden_states": torch.randn(1, 12, 32, dtype=torch.bfloat16),
            "loss_mask": torch.ones(1, 12),
        }
        if distillation:
            tensors["target_last_hidden_states"] = torch.randn(1, 12, 16, dtype=torch.bfloat16)
        calls = []
        handle = wrapped.register_forward_pre_hook(lambda *args: calls.append(True))
        out = strategy.forward_loss(SimpleNamespace(tensors=tensors))
        handle.remove()
        assert calls == [True]
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert model.draft_model.fc.weight.grad.abs().sum() > 0
        if distillation:
            assert "l1_loss" in out.metrics["log_window"]["sums"]
            del tensors["target_last_hidden_states"]
            with pytest.raises(ValueError, match="batch missing required features"):
                strategy.forward_loss(SimpleNamespace(tensors=tensors))
    finally:
        dist.destroy_process_group()
