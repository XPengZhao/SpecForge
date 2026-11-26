"""Shared Muon policy for pure DFlash and DSpark, including training/export."""

import pytest
import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.config import Config, apply_overrides
from specforge.config.schema import TrainingConfig
from specforge.export.to_hf import export_to_hf
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.training.assembly import _ConfiguredOptimizerFactory


def draft(strategy):
    cls = DFlashDraftModel if strategy == "dflash" else DSparkDraftModel
    cfg = Qwen3Config(
        architectures=[cls.__name__],
        hidden_size=16,
        intermediate_size=32,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=2,
        num_target_layers=8,
        layer_types=["full_attention"] * 2,
        block_size=4,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        dflash_config={
            "target_layer_ids": [1, 5],
            "mask_token_id": 31,
            **(
                {"projector_type": "dspark", "markov_rank": 4}
                if strategy == "dspark"
                else {}
            ),
        },
    )
    cfg._attn_implementation = "sdpa"
    return cls(cfg)


@pytest.mark.parametrize("strategy", ["dflash", "dspark"])
def test_cli_factory_and_parameter_partition(strategy):
    cfg = Config.model_validate(
        {
            "model": {"target_model_path": "unused"},
            "data": {"hidden_states_path": "unused"},
            "training": {
                "strategy": strategy,
                "fsdp_sharding": "NO_SHARD",
                "total_steps": 10,
            },
        }
    )
    assert not cfg.training.dspark_backbone_muon
    cfg = apply_overrides(cfg, ["training.dspark_backbone_muon=true"])
    model = draft(strategy)
    opt = _ConfiguredOptimizerFactory(cfg)(model)
    assert isinstance(opt.muon, torch.optim.Muon)
    assert opt.muon.param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    names = {id(p): name for name, p in model.named_parameters()}
    masters = {id(m): names[id(p)] for p, m in zip(opt.model_params, opt.fp32_params)}
    muon = {masters[id(p)] for g in opt.muon.param_groups for p in g["params"]}
    adam = {masters[id(p)] for g in opt.optimizer.param_groups for p in g["params"]}
    expected = {"fc.weight"} | {
        f"layers.{i}.{suffix}.weight"
        for i in range(2)
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
    assert muon == expected
    assert not muon & adam
    assert muon | adam == set(names.values())
    assert "hidden_norm.weight" in adam
    if strategy == "dspark":
        assert "markov_head.markov_w1.weight" in adam
        assert "markov_head.markov_w2.weight" in adam
    else:
        assert not hasattr(model, "markov_head")
        assert not hasattr(model, "confidence_head")


def online_dflash():
    model = draft("dflash").to(torch.bfloat16)
    wrapper = OnlineDFlashModel(
        model,
        torch.nn.Linear(16, 32, bias=False).to(torch.bfloat16).requires_grad_(False),
        torch.nn.Embedding(32, 16).to(torch.bfloat16).requires_grad_(False),
        mask_token_id=31,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=2,
    )
    return wrapper


def test_pure_dflash_loss_muon_resume_and_hf_export(tmp_path, monkeypatch):
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    torch.manual_seed(12)
    model = online_dflash()
    optimizer = BF16Optimizer(
        model, lr=0.01, warmup_ratio=0.2, total_steps=10, backbone_muon=True
    )
    batch = {
        "input_ids": torch.randint(0, 31, (1, 12)),
        "hidden_states": torch.randn(1, 12, 32, dtype=torch.bfloat16),
        "loss_mask": torch.ones(1, 12),
    }

    def step(wrapper, opt):
        loss, _, _ = wrapper(**batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert wrapper.draft_model.fc.weight.grad.count_nonzero() > 0
        opt.step()
        return loss.detach()

    before = model.draft_model.fc.weight.detach().clone()
    target_before = model.lm_head.weight.detach().clone()
    step(model, optimizer)
    assert not torch.equal(before, model.draft_model.fc.weight)
    torch.testing.assert_close(target_before, model.lm_head.weight, atol=0, rtol=0)
    state_path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.random.get_rng_state(),
        },
        state_path,
    )
    restored = online_dflash()
    saved = torch.load(state_path, weights_only=True)
    restored.load_state_dict(saved["model"], strict=True)
    restored_opt = BF16Optimizer(
        restored, lr=0.01, warmup_ratio=0.2, total_steps=10, backbone_muon=True
    )
    restored_opt.load_state_dict(saved["optimizer"])
    torch.random.set_rng_state(saved["rng"])
    continuous_loss = step(model, optimizer)
    torch.random.set_rng_state(saved["rng"])
    resumed_loss = step(restored, restored_opt)
    torch.testing.assert_close(continuous_loss, resumed_loss, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    for original, resumed in zip(optimizer.fp32_params, restored_opt.fp32_params):
        torch.testing.assert_close(original, resumed, atol=0, rtol=0)

    cfg_path = tmp_path / "draft.json"
    model.draft_model.config.to_json_file(cfg_path)
    checkpoint = tmp_path / "training_state.pt"
    torch.save({"draft_state_dict": model.draft_model.state_dict()}, checkpoint)
    out = tmp_path / "hf"
    export_to_hf(str(checkpoint), str(cfg_path), str(out))
    exported, info = AutoDraftModel.from_pretrained(str(out), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    for key, value in model.draft_model.state_dict().items():
        torch.testing.assert_close(
            value.float(), exported.state_dict()[key].float(), atol=0, rtol=0
        )


@pytest.mark.parametrize("strategy", ["dflash", "dspark"])
def test_lr_schedule_and_rebase(strategy, monkeypatch):
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    model = draft(strategy)
    opt = BF16Optimizer(
        model, lr=0.01, warmup_ratio=0.3, total_steps=10, backbone_muon=True
    )
    for i in range(6):
        if i == 2:
            opt.rebase_learning_rate(0.02)
        expected = opt.get_learning_rate()
        for p in model.parameters():
            p.grad = torch.full_like(p, 0.01)
        opt.step()
        assert opt.muon.param_groups[0]["lr"] == expected
        assert all(p.grad is None for p in opt.fp32_params)


@pytest.mark.parametrize("strategy", ["dflash", "dspark"])
@pytest.mark.parametrize("source_muon", [False, True])
def test_optimizer_switch_on_resume_rejected(strategy, source_muon):
    model = draft(strategy)
    source = BF16Optimizer(model, lr=0.01, backbone_muon=source_muon)
    dest = BF16Optimizer(model, lr=0.01, backbone_muon=not source_muon)
    with pytest.raises(ValueError, match="disagree on backbone Muon"):
        dest.load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    "strategy,sharding",
    [
        ("dflash", "FULL_SHARD"),
        ("dflash", "SHARD_GRAD_OP"),
        ("dspark", "FULL_SHARD"),
        ("eagle3", "NO_SHARD"),
    ],
)
def test_unsupported_config_rejected(strategy, sharding):
    with pytest.raises(
        ValueError, match="requires DFlash/DSpark training with NO_SHARD"
    ):
        TrainingConfig(
            strategy=strategy, fsdp_sharding=sharding, dspark_backbone_muon=True
        )


def test_default_adamw_and_unsupported_model():
    for strategy in ("dflash", "dspark"):
        opt = BF16Optimizer(draft(strategy), lr=0.01)
        assert opt.muon is None
        assert "muon_state_dict" not in opt.state_dict()
    with pytest.raises(ValueError, match="requires a Qwen DFlash/DSpark backbone"):
        BF16Optimizer(torch.nn.Linear(4, 4), lr=0.01, backbone_muon=True)
