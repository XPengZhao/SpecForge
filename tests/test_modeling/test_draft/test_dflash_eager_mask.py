"""Eager attention must honor DSpark boolean masks, including empty rows."""

import pytest
import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
from specforge.modeling.draft.dflash import Qwen3DFlashAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS


def attention(backend):
    config = Qwen3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = backend
    return Qwen3DFlashAttention(config, 0, DEFAULT_DFLASH_KERNELS)


def inputs():
    torch.manual_seed(32)
    context = torch.randn(1, 8, 16)
    noise = torch.randn(1, 6, 16)
    mask = create_dflash_sdpa_mask(
        torch.tensor([[2, 5]]), torch.tensor([[True, True]]), 8, 3, "cpu"
    )
    rotary = (torch.ones(1, 14, 8), torch.zeros(1, 14, 8))
    return context, noise, mask, rotary


def test_eager_cannot_read_future_context_or_other_blocks():
    module = attention("eager")
    context, noise, mask, rotary = inputs()
    before = module(noise, context, rotary, mask)[0]
    context[:, 2:] += 100
    noise[:, 3:] -= 100
    after = module(noise, context, rotary, mask)[0]
    torch.testing.assert_close(before[:, :3], after[:, :3], atol=0, rtol=0)


@pytest.mark.parametrize("empty_rows", [False, True])
def test_eager_matches_sdpa_outputs_and_gradients(empty_rows):
    eager, sdpa = attention("eager"), attention("sdpa")
    sdpa.load_state_dict(eager.state_dict())
    context, noise, mask, rotary = inputs()
    if empty_rows:
        mask[:, :, 3:, :] = False
    outputs, grads = [], []
    for module in (eager, sdpa):
        c = context.clone().requires_grad_()
        n = noise.clone().requires_grad_()
        output = module(n, c, rotary, mask)[0]
        assert torch.isfinite(output).all()
        if empty_rows:
            assert output[:, 3:].count_nonzero() == 0
        output.square().sum().backward()
        outputs.append(output)
        grads.append([c.grad, n.grad] + [p.grad for p in module.parameters()])
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
    for left, right in zip(*grads):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_eager_boolean_and_additive_masks_agree():
    module = attention("eager")
    context, noise, mask, rotary = inputs()
    additive = torch.zeros_like(mask, dtype=noise.dtype).masked_fill(~mask, -torch.inf)
    boolean_result = module(noise, context, rotary, mask)[0]
    additive_result = module(noise, context, rotary, additive)[0]
    torch.testing.assert_close(boolean_result, additive_result)
