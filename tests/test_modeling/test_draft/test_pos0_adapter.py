"""Position isolation, zero-init, shared gradients and HF persistence."""

import copy

import pytest
import torch

from tests.test_optimizer.test_dflash_family_muon import draft
from specforge.export.to_hf import export_to_hf
from specforge.modeling.auto import AutoDraftModel
from specforge.optimizer import BF16Optimizer


def models(kind):
    baseline = draft(kind)
    config = copy.deepcopy(baseline.config)
    config.dflash_config["pos0_adapter_intermediate_size"] = 8
    model = type(baseline)(config)
    result = model.load_state_dict(baseline.state_dict(), strict=False)
    assert all(key.startswith("pos0_adapter.") for key in result.missing_keys)
    assert not result.unexpected_keys
    return baseline, model


@pytest.mark.parametrize("kind,slot", [("dspark", 0), ("dflash", 1)])
def test_isolation_and_learning(kind, slot):
    _, model = models(kind)
    adapter = model.pos0_adapter
    assert model.pos0_adapter_slot == slot
    assert adapter.down_proj.weight.count_nonzero() == 0
    hidden = torch.randn(2, 12, 16, requires_grad=True)
    torch.testing.assert_close(
        model._apply_pos0_adapter(hidden), hidden, rtol=0, atol=0
    )
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    probe = torch.randn(2, 3, 16)
    for step in range(2):
        optimizer.zero_grad()
        output = model.norm(model._apply_pos0_adapter(hidden)).reshape(2, 3, 4, 16)
        (output[:, :, slot] * probe).sum().backward()
        assert hidden.grad is not None and hidden.grad.abs().sum() > 0
        assert adapter.down_proj.weight.grad.abs().sum() > 0
        if step:
            for name, parameter in adapter.named_parameters():
                assert parameter.grad.abs().sum() > 0, name
        optimizer.step()
    modified = model._apply_pos0_adapter(hidden).reshape(2, 3, 4, 16)
    original = hidden.reshape(2, 3, 4, 16)
    for position in range(4):
        if position == slot:
            assert not torch.equal(modified[:, :, position], original[:, :, position])
        else:
            torch.testing.assert_close(
                modified[:, :, position], original[:, :, position], atol=0, rtol=0
            )
    optimizer.zero_grad()
    output = model.norm(model._apply_pos0_adapter(hidden)).reshape(2, 3, 4, 16)
    (output[:, :, (slot + 1) % 4] * probe).sum().backward()
    for parameter in adapter.parameters():
        assert parameter.grad is None or parameter.grad.count_nonzero() == 0


@pytest.mark.parametrize("kind", ["dspark", "dflash"])
def test_real_forward_and_hf_export(kind, tmp_path):
    baseline, model = models(kind)
    baseline = baseline.to(torch.bfloat16)
    model = model.to(torch.bfloat16)
    inputs = dict(
        noise_embedding=torch.randn(1, 8, 16, dtype=torch.bfloat16),
        target_hidden=torch.randn(1, 6, 32, dtype=torch.bfloat16),
        position_ids=torch.arange(14)[None],
        attention_mask=torch.ones(1, 1, 8, 14, dtype=torch.bool),
    )
    baseline.eval()
    model.eval()
    torch.testing.assert_close(model(**inputs), baseline(**inputs), atol=0, rtol=0)
    with torch.no_grad():
        model.pos0_adapter.down_proj.weight.normal_(std=0.1)
    before_norm = []
    handle = model.norm.register_forward_pre_hook(
        lambda module, args: before_norm.append(args[0])
    )
    output = model(**inputs)
    handle.remove()
    torch.testing.assert_close(output, model.norm(before_norm[0]), atol=0, rtol=0)
    config = tmp_path / "draft.json"
    model.config.to_json_file(config)
    ckpt = tmp_path / "training_state.pt"
    torch.save({"draft_state_dict": model.state_dict()}, ckpt)
    export_to_hf(str(ckpt), str(config), str(tmp_path / "hf"))
    loaded, info = AutoDraftModel.from_pretrained(
        str(tmp_path / "hf"), output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    # Match training assembly's dtype conversion, including nonpersistent RoPE buffers.
    loaded.to(torch.bfloat16).eval()
    torch.testing.assert_close(loaded(**inputs), output, atol=0, rtol=0)


def test_muon_partition_and_resume(monkeypatch):
    monkeypatch.setattr("specforge.optimizer.print_on_rank0", lambda *args: None)
    _, model = models("dspark")
    optimizer = BF16Optimizer(model, lr=0.01, total_steps=10, backbone_muon=True)
    masters = {id(p): m for p, m in zip(optimizer.model_params, optimizer.fp32_params)}
    muon_ids = {id(p) for group in optimizer.muon.param_groups for p in group["params"]}
    for name, p in model.pos0_adapter.named_parameters():
        assert (id(masters[id(p)]) in muon_ids) == ("proj" in name)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    _, restored = models("dspark")
    restored.load_state_dict(model.state_dict(), strict=True)
    resumed = BF16Optimizer(restored, lr=0.01, total_steps=10, backbone_muon=True)
    resumed.load_state_dict(optimizer.state_dict())
    for a, b in zip(optimizer.fp32_params, resumed.fp32_params):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
