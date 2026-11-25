"""Muon grouping for shared/per-layer DSpark, with baseline scheduling semantics."""

from types import SimpleNamespace

import pytest
import torch

from specforge.config.schema import TrainingConfig
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.training.assembly import _ConfiguredOptimizerFactory
from tests.test_modeling.test_draft.test_layerwise_context_projection import config


def draft(mode):
    cfg = config(mode)
    cfg.num_hidden_layers = 5
    cfg.layer_types = ["full_attention"] * 5
    return DSparkDraftModel(cfg)


@pytest.mark.parametrize("mode", ["shared", "per_layer"])
def test_factory_groups_all_backbone_matrices_only(mode):
    model = draft(mode)
    cfg = SimpleNamespace(
        training=TrainingConfig(
            strategy="dspark",
            fsdp_sharding="NO_SHARD",
            dspark_backbone_muon=True,
            total_steps=20,
        )
    )
    opt = _ConfiguredOptimizerFactory(cfg)(model)
    assert isinstance(opt.muon, torch.optim.Muon)
    assert opt.muon.param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    names = {id(p): name for name, p in model.named_parameters()}
    master_names = {
        id(master): names[id(p)] for p, master in zip(opt.model_params, opt.fp32_params)
    }
    muon_names = {
        master_names[id(p)] for group in opt.muon.param_groups for p in group["params"]
    }
    adam_names = {
        master_names[id(p)]
        for group in opt.optimizer.param_groups
        for p in group["params"]
    }
    projections = (
        {f"context_projections.{i}.weight" for i in range(5)}
        if mode == "per_layer"
        else {"fc.weight"}
    )
    expected = projections | {
        f"layers.{i}.{suffix}.weight"
        for i in range(5)
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    assert muon_names == expected
    assert not muon_names & adam_names
    assert muon_names | adam_names == set(names.values())
    assert "markov_head.markov_w2.weight" in adam_names
    assert "hidden_norm.weight" in adam_names


@pytest.mark.parametrize("rebase", [False, True])
def test_muon_uses_same_scheduled_lr_as_adamw(monkeypatch, rebase):
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    model = draft("per_layer")
    opt = BF16Optimizer(
        model, lr=0.01, warmup_ratio=0.3, total_steps=10, backbone_muon=True
    )
    for i in range(6):
        if rebase and i == 2:
            opt.rebase_learning_rate(0.02)
        expected_lr = opt.get_learning_rate()
        for p in model.parameters():
            p.grad = torch.full_like(p, 0.01)
        opt.step()
        # Muon's LR is synchronized before its update; AdamW's scheduler then advances.
        assert opt.muon.param_groups[0]["lr"] == expected_lr
        assert all(p.grad is None for p in opt.fp32_params)


@pytest.mark.parametrize("source_muon", [False, True])
def test_resume_rejects_optimizer_switch(source_muon):
    model = draft("per_layer")
    source = BF16Optimizer(model, lr=0.01, backbone_muon=source_muon)
    dest = BF16Optimizer(model, lr=0.01, backbone_muon=not source_muon)
    with pytest.raises(ValueError, match="disagree on backbone Muon"):
        dest.load_state_dict(source.state_dict())


def test_default_remains_adamw():
    model = draft("per_layer")
    opt = BF16Optimizer(model, lr=0.01)
    assert opt.muon is None
    assert "muon_state_dict" not in opt.state_dict()
    assert not TrainingConfig().dspark_backbone_muon


@pytest.mark.parametrize(
    "strategy,sharding",
    [
        ("dspark", "FULL_SHARD"),
        ("dspark", "SHARD_GRAD_OP"),
        ("eagle3", "NO_SHARD"),
    ],
)
def test_config_rejects_incompatible_muon(strategy, sharding):
    with pytest.raises(ValueError, match="requires DSpark training with NO_SHARD"):
        TrainingConfig(
            strategy=strategy, fsdp_sharding=sharding, dspark_backbone_muon=True
        )


def test_rejects_non_dspark_model():
    with pytest.raises(ValueError, match="requires a DSparkDraftModel"):
        BF16Optimizer(torch.nn.Linear(4, 4), lr=0.01, backbone_muon=True)
