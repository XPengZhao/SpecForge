"""Standalone memory checks, without target-model/runtime dependencies."""
import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[2] / 'specforge/modeling/draft/markov_engram.py'
_spec = importlib.util.spec_from_file_location('engram_vector_gate_memory', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
EngramMemory = _module.EngramMemory


def options(enabled=False, value_init='random'):
    return dict(hash_version='xor_v1', heads_per_order=2,
                bucket_counts=[31, 37, 41, 43], hash_multipliers=[11, 13, 17],
                head_dim=2, vector_gate=enabled, value_init=value_init)


def test_identity_initialization_and_shared_rng():
    torch.manual_seed(42)
    base = EngramMemory(options(), 16, 4, 1e-6)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    gated = EngramMemory(options(True), 16, 4, 1e-6)
    assert torch.equal(rng, torch.get_rng_state())
    for key, weight in base.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[key], weight, atol=0, rtol=0)
    ids = gated.bucket_ids(torch.tensor([[1, 2, 3, 4]]))
    h = torch.randn(1, 4, 16)
    torch.testing.assert_close(gated(ids, h), base(ids, h), atol=0, rtol=0)
    assert sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in base.parameters()) == 16*4+4+16


def test_formula_gradients_and_checkpoint(tmp_path):
    m = EngramMemory(options(True), 16, 4, 1e-6)
    with torch.no_grad():
        m.vector_gate_proj.weight.normal_(std=.2)
    h = torch.randn(2, 4, 16, requires_grad=True)
    ids = m.bucket_ids(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
    base = EngramMemory(options(), 16, 4, 1e-6)
    base.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('vector_gate_')})
    r = 2 * m.vector_gate_proj(m.vector_gate_norm(h)).float().sigmoid()
    torch.testing.assert_close(m(ids,h), base(ids,h) * r)
    m(ids,h).square().sum().backward()
    for p in (m.vector_gate_proj.weight, m.vector_gate_proj.bias, m.vector_gate_norm.weight):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    torch.save(m.state_dict(), tmp_path/'state.pt')
    restored = EngramMemory(options(True),16,4,1e-6)
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
    torch.testing.assert_close(restored(ids,h),m(ids,h))


def test_zero_value_training_starts():
    m = EngramMemory(options(True, 'zero'),16,4,1e-6)
    ids=m.bucket_ids(torch.tensor([[1,2,3,4]]));h=torch.randn(1,4,16)
    opt=torch.optim.SGD(m.parameters(),lr=.1)
    for step in range(3):
        opt.zero_grad();m(ids,h).sum().backward()
        if step == 0:
            assert m.value.weight.grad.abs().sum()>0
            assert m.vector_gate_proj.weight.grad.count_nonzero()==0
        if step == 1:
            assert m.vector_gate_proj.weight.grad.abs().sum()>0
        if step == 2:
            assert m.vector_gate_norm.weight.grad.abs().sum()>0
        opt.step()


def test_layout_and_invalid_switch():
    a=options();b=dict(a);b.pop('vector_gate')
    assert _module.engram_weight_layout(a)==_module.engram_weight_layout(b)
    assert _module.engram_weight_layout(a)!=_module.engram_weight_layout(options(True))
    with pytest.raises(ValueError,match='boolean'):
        EngramMemory(dict(a,vector_gate='true'),16,4,1e-6)
