"""Controlled context-projection ablation, warm starts and strict HF export."""

import json

import pytest
import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
from specforge.export.to_hf import export_to_hf
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.training.model_loading import warm_start_draft_model


def config(mode="shared"):
    cfg = Qwen3Config(
        architectures=["DSparkDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=3,
        num_target_layers=8,
        layer_types=["full_attention"] * 3,
        vocab_size=32,
        block_size=3,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        dflash_config={
            "target_layer_ids": [1, 5],
            "mask_token_id": 31,
            "projector_type": "dspark",
            "markov_rank": 4,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "context_projection_mode": mode,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def models():
    torch.manual_seed(42)
    shared = DSparkDraftModel(config())
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    per_layer = DSparkDraftModel(config("per_layer"))
    assert torch.equal(rng, torch.random.get_rng_state())
    return shared, per_layer


def inputs(dtype=torch.float32):
    torch.manual_seed(7)
    return {
        "target_hidden": torch.randn(1, 8, 32, dtype=dtype),
        "noise_embedding": torch.randn(1, 6, 16, dtype=dtype),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7]]),
        "attention_mask": create_dflash_sdpa_mask(
            torch.tensor([[2, 5]]), torch.tensor([[True, True]]), 8, 3, "cpu"
        ),
    }


def write_checkpoint(path, state):
    torch.save(
        {"draft_state_dict": state, "strategy": "dspark", "global_step": 999}, path
    )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_initialization_forward_and_backward_equivalence(backend):
    shared, per_layer = models()
    shared.config._attn_implementation = backend
    per_layer.config._attn_implementation = backend
    state = per_layer.state_dict()
    for key, value in shared.state_dict().items():
        if key != "fc.weight":
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    weights = [p.weight for p in per_layer.context_projections]
    assert len({w.data_ptr() for w in weights}) == 3
    assert (
        sum(p.numel() for p in per_layer.parameters())
        - sum(p.numel() for p in shared.parameters())
        == 2 * 32 * 16
    )
    for weight in weights:
        torch.testing.assert_close(weight, shared.fc.weight, atol=0, rtol=0)
    args = inputs()
    left, right = shared(**args), per_layer(**args)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    # Include the actual Markov and confidence paths in the comparison.
    logits = torch.randn(1, 6, 32)
    tokens = torch.arange(6).view(1, -1)
    torch.testing.assert_close(
        shared.apply_markov_logits(logits, prev_token_ids=tokens, hidden_states=left),
        per_layer.apply_markov_logits(
            logits, prev_token_ids=tokens, hidden_states=right
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        shared.predict_confidence(left, prev_token_ids=tokens),
        per_layer.predict_confidence(right, prev_token_ids=tokens),
        atol=0,
        rtol=0,
    )
    probe = torch.randn_like(left)
    (left * probe).sum().backward()
    (right * probe).sum().backward()
    for weight in weights:
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.count_nonzero() > 0
    torch.testing.assert_close(shared.fc.weight.grad, sum(w.grad for w in weights))
    torch.testing.assert_close(
        shared.hidden_norm.weight.grad, per_layer.hidden_norm.weight.grad
    )
    snapshots = [w.detach().clone() for w in weights]
    with torch.no_grad():
        weights[1].add_(torch.randn_like(weights[1]))
    assert not torch.equal(per_layer(**args), right)
    for index in (0, 2):
        torch.testing.assert_close(weights[index], snapshots[index], atol=0, rtol=0)


def test_each_layer_receives_its_own_projection_and_mask_is_unchanged():
    _, model = models()
    args = inputs()
    with torch.no_grad():
        for index, projection in enumerate(model.context_projections):
            projection.weight.add_(torch.randn_like(projection.weight) * index)
    seen = []
    handles = [
        layer.register_forward_pre_hook(
            lambda module, positional, kwargs: seen.append(
                kwargs["target_hidden"].detach().clone()
            ),
            with_kwargs=True,
        )
        for layer in model.layers
    ]
    before = model(**args)
    for handle in handles:
        handle.remove()
    for actual, projection in zip(seen, model.context_projections):
        torch.testing.assert_close(
            actual, model.hidden_norm(projection(args["target_hidden"]))
        )
    args["target_hidden"][:, 2:] += 100
    args["noise_embedding"][:, 3:] -= 100
    after = model(**args)
    torch.testing.assert_close(before[:, :3], after[:, :3], atol=0, rtol=0)


def test_gradient_checkpointing():
    _, model = models()
    reference = DSparkDraftModel(config("per_layer"))
    reference.load_state_dict(model.state_dict())
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    args = inputs()
    probe = torch.randn(1, 6, 16)
    for candidate in (reference, model):
        (candidate(**args) * probe).sum().backward()
    for left, right in zip(reference.parameters(), model.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize("source_mode", ["shared", "per_layer"])
@pytest.mark.parametrize("source_format", ["specforge", "hf"])
def test_warm_start(tmp_path, source_mode, source_format):
    from transformers import initialization as hf_init

    source = DSparkDraftModel(config(source_mode))
    # The training provider skips random initialization when a source is given.
    with hf_init.no_init_weights():
        destination = DSparkDraftModel(config("per_layer"))
    if source_format == "specforge":
        path = tmp_path / "training_state.pt"
        write_checkpoint(path, source.state_dict())
    else:
        path = tmp_path / "hf"
        source.save_pretrained(path)
    report = warm_start_draft_model(
        destination, str(path), draft_config=destination.config, strategy="dspark"
    )
    assert not report.missing_keys
    source.eval()
    destination.eval()
    torch.testing.assert_close(
        source(**inputs()), destination(**inputs()), atol=0, rtol=0
    )


def test_hf_export_roundtrip_and_strict_projection_validation(tmp_path):
    _, model = models()
    model = model.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in model.context_projections:
            p.weight.add_(torch.randn_like(p.weight))
    cfg_path = tmp_path / "draft.json"
    model.config.to_json_file(cfg_path)
    checkpoint = tmp_path / "training_state.pt"
    write_checkpoint(checkpoint, model.state_dict())
    out = tmp_path / "export"
    export_to_hf(str(checkpoint), str(cfg_path), str(out))
    exported_config = json.loads((out / "config.json").read_text())
    assert exported_config["dflash_config"]["context_projection_mode"] == "per_layer"
    restored, info = AutoDraftModel.from_pretrained(
        str(out), output_loading_info=True, attn_implementation="eager"
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    restored = restored.to(torch.bfloat16).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(
        model(**inputs(torch.bfloat16)),
        restored(**inputs(torch.bfloat16)),
        atol=0,
        rtol=0,
    )
    # A wrong JSON must not turn shared weights into an untrained ablation.
    shared = DSparkDraftModel(config())
    write_checkpoint(checkpoint, shared.state_dict())
    with pytest.raises(ValueError, match="does not have"):
        export_to_hf(str(checkpoint), str(cfg_path), str(tmp_path / "wrong"))
    missing = dict(model.state_dict())
    missing.pop("context_projections.1.weight")
    write_checkpoint(checkpoint, missing)
    with pytest.raises(ValueError, match="missing non-embedding"):
        export_to_hf(str(checkpoint), str(cfg_path), str(tmp_path / "missing"))
    config().to_json_file(cfg_path)
    write_checkpoint(checkpoint, model.state_dict())
    with pytest.raises(ValueError, match="does not have"):
        export_to_hf(str(checkpoint), str(cfg_path), str(tmp_path / "wrong-shared"))


def test_normal_load_does_not_expand_baseline_and_mixed_warm_start_fails(tmp_path):
    shared, per_layer = models()
    with pytest.raises(RuntimeError, match="Missing key"):
        per_layer.load_state_dict(shared.state_dict())
    mixed = dict(shared.state_dict())
    mixed["context_projections.0.weight"] = shared.fc.weight.detach()
    with pytest.raises(ValueError, match="mixes shared and per-layer"):
        per_layer.adapt_warm_start_state_dict(mixed)
    partial = dict(per_layer.state_dict())
    partial.pop("context_projections.1.weight")
    path = tmp_path / "training_state.pt"
    write_checkpoint(path, partial)
    with pytest.raises(ValueError, match="missing draft weights"):
        warm_start_draft_model(
            per_layer, str(path), draft_config=per_layer.config, strategy="dspark"
        )


def test_default_and_invalid_mode():
    cfg = config()
    cfg.dflash_config.pop("context_projection_mode")
    model = DSparkDraftModel(cfg)
    assert model.context_projection_mode == "shared"
    assert "fc.weight" in model.state_dict()
    with pytest.raises(ValueError, match="context_projection_mode"):
        DSparkDraftModel(config("typo"))


@pytest.mark.parametrize("source_mode", ["shared", "per_layer"])
def test_hf_warm_start_rejects_mixed_projection_layouts(tmp_path, source_mode):
    from safetensors.torch import load_file, save_file

    source = DSparkDraftModel(config(source_mode))
    source.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if source_mode == "shared":
        state["context_projections.0.weight"] = state["fc.weight"].clone()
    else:
        state["fc.weight"] = state["context_projections.0.weight"].clone()
    save_file(state, path)
    destination = DSparkDraftModel(config("per_layer"))
    with pytest.raises(ValueError, match="unexpected.*weights"):
        warm_start_draft_model(
            destination,
            str(tmp_path),
            draft_config=destination.config,
            strategy="dspark",
        )


@pytest.mark.parametrize("backbone_muon", [False, True])
@pytest.mark.parametrize("projection_mode", ["shared", "per_layer"])
def test_bf16_optimizer_update_and_resume(
    tmp_path, monkeypatch, backbone_muon, projection_mode
):
    from specforge.optimizer import BF16Optimizer

    # This CPU unit test has no process group; suppress only rank-aware logging.
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    model = DSparkDraftModel(config(projection_mode)).to(torch.bfloat16)
    optimizer = BF16Optimizer(
        model, lr=0.01, warmup_ratio=0, total_steps=10, backbone_muon=backbone_muon
    )
    args = inputs(torch.bfloat16)
    probe = torch.randn(1, 6, 16)

    def step(candidate, opt):
        (candidate(**args).float() * probe).sum().backward()
        opt.step()

    projections = (
        list(model.context_projections)
        if projection_mode == "per_layer"
        else [model.fc]
    )
    before = [p.weight.detach().clone() for p in projections]
    step(model, optimizer)
    for old, projection in zip(before, projections):
        assert not torch.equal(old, projection.weight)
    if projection_mode == "per_layer":
        assert not torch.equal(projections[0].weight, projections[1].weight)
    path = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    state = torch.load(path, weights_only=True)
    restored = DSparkDraftModel(config(projection_mode)).to(torch.bfloat16)
    restored.load_state_dict(state["model"], strict=True)
    restored_opt = BF16Optimizer(
        restored, lr=0.01, warmup_ratio=0, total_steps=10, backbone_muon=backbone_muon
    )
    restored_opt.load_state_dict(state["optimizer"])
    step(model, optimizer)
    step(restored, restored_opt)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    for original, resumed in zip(optimizer.fp32_params, restored_opt.fp32_params):
        torch.testing.assert_close(original, resumed, atol=0, rtol=0)
