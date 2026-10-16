"""Sparse routing math, causal inputs, checkpointing and DDP accumulation."""

from contextlib import nullcontext
from copy import deepcopy
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.dspark_moe import SharedSparseMLP


def config(experts=4):
    c = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
    )
    c.block_size = 2
    c.num_target_layers = 4
    c.dflash_config = dict(
        projector_type="dspark",
        target_layer_ids=[0, 1],
        markov_rank=2,
        confidence_head_alpha=1.0,
        moe=dict(enabled=True, num_experts=experts, balance_loss_alpha=0.01),
    )
    c._attn_implementation = "eager"
    return c


def inputs():
    return dict(
        position_ids=torch.tensor([[0, 1, 2, 3, 2, 3, 0, 1]]),
        noise_embedding=torch.randn(1, 4, 8),
        target_hidden=torch.randn(1, 4, 16),
        routing_mask=torch.tensor([[True, False]]),
    )


def test_top1_gate_reference_and_router_gradient():
    torch.manual_seed(42)
    m = SharedSparseMLP(config(), Qwen3MLP)
    x, context = torch.randn(2, 6, 8), torch.randn(2, 3, 8)
    keep = torch.tensor([[True, True, False], [True, False, True]])
    actual, stats = m(x, context, keep)
    expected = m.shared(x).reshape(2, 3, 2, 8).clone()
    p = m.router(context).softmax(-1)
    for b in range(2):
        for a in range(3):
            if keep[b, a]:
                e = int(p[b, a].argmax())
                expected[b, a] += p[b, a, e] * m.experts[e](x[b, 2 * a : 2 * a + 2])
    torch.testing.assert_close(actual, expected.reshape_as(actual))
    counts = F.one_hot(p.argmax(-1), 4)[keep].float().sum(0)
    expected_aux = 4 * (counts / keep.sum() * p[keep].mean(0)).sum()
    torch.testing.assert_close(stats["balance_loss"], expected_aux)
    actual.square().sum().backward()
    assert m.router.weight.grad.abs().sum() > 0  # task loss alone trains router
    assert all(p.grad is not None for p in m.parameters())


def test_auxiliary_gradient_and_invalid_blocks():
    torch.manual_seed(4)
    m = SharedSparseMLP(config(), Qwen3MLP)
    x, c = torch.randn(1, 6, 8), torch.randn(1, 3, 8)
    keep = torch.tensor([[True, False, False]])
    _, stats = m(x, c, keep)
    stats["balance_loss"].backward()
    assert m.router.weight.grad.abs().sum() > 0
    changed = c.clone()
    changed[:, 1:] = 1000
    _, other = m(x, changed, keep)
    torch.testing.assert_close(stats["balance_loss"], other["balance_loss"])
    empty, stats = m(x, c, torch.zeros_like(keep))
    torch.testing.assert_close(empty, m.shared(x))
    assert stats["balance_loss"].item() == 0
    assert stats["count"].item() == 0
    assert torch.isfinite(stats["balance_loss"])


def test_e1_split_dense_output_and_gradients():
    dense = Qwen3MLP(config(1)).double()
    sparse = SharedSparseMLP(config(1), Qwen3MLP).double()
    with torch.no_grad():
        for part, section in (
            (sparse.shared, slice(0, 8)),
            (sparse.experts[0], slice(8, 16)),
        ):
            for name in ("gate_proj", "up_proj"):
                getattr(part, name).weight.copy_(getattr(dense, name).weight[section])
            part.down_proj.weight.copy_(dense.down_proj.weight[:, section])
    x = torch.randn(1, 4, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, stats = sparse(x, torch.randn(1, 2, 8), torch.ones(1, 2, dtype=torch.bool))
    expected = dense(y)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for part, section in (
        (sparse.shared, slice(0, 8)),
        (sparse.experts[0], slice(8, 16)),
    ):
        for name in ("gate_proj", "up_proj"):
            torch.testing.assert_close(
                getattr(part, name).weight.grad,
                getattr(dense, name).weight.grad[section],
            )
        torch.testing.assert_close(
            part.down_proj.weight.grad, dense.down_proj.weight.grad[:, section]
        )


def test_backbone_router_context_is_causal_and_handles_cached_positions():
    m = DSparkDraftModel(config()).eval()
    data = inputs()
    seen = []
    hook = m.layers[0].mlp.register_forward_pre_hook(
        lambda _, args: seen.append(args[1].detach().clone())
    )
    _, stats = m(**data, return_router_stats=True)
    projected = m.hidden_norm(m.fc(data["target_hidden"]))
    torch.testing.assert_close(seen[-1][:, 0], projected[:, 1])
    assert seen[-1][:, 1].count_nonzero() == 0
    changed = deepcopy(data)
    changed["target_hidden"][:, 2:] += 100
    m(**changed)
    torch.testing.assert_close(seen[-1], seen[-2])
    # Same block, but context chunk starts at absolute position 10.
    cached = dict(
        position_ids=torch.tensor([[10, 11, 12, 13]]),
        target_hidden=data["target_hidden"][:, :2],
        noise_embedding=data["noise_embedding"][:, :2],
    )
    m(**cached)
    torch.testing.assert_close(seen[-1][:, 0], projected[:, 1])
    assert stats[0]["count"].item() == 1
    hook.remove()


def test_checkpoint_roundtrip_and_inference_gate(tmp_path):
    m = DSparkDraftModel(config()).eval()
    data = inputs()
    expected = m(**data)
    m.save_pretrained(tmp_path)
    loaded = DSparkDraftModel.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(loaded(**data), expected)
    assert loaded.moe_enabled
    assert len(loaded.layers[0].mlp.experts) == 4


def test_wrapper_adds_auxiliary_without_changing_task_metrics():
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

    torch.manual_seed(42)
    draft = DSparkDraftModel(config())
    model = OnlineDSparkModel(
        draft,
        torch.nn.Linear(8, 32, bias=False),
        torch.nn.Embedding(32, 8),
        mask_token_id=31,
        block_size=2,
        num_anchors=3,
        attention_backend="sdpa",
    )
    data = dict(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        hidden_states=torch.randn(1, 6, 16),
        loss_mask=torch.ones(1, 6),
        target_last_hidden_states=torch.randn(1, 6, 8),
    )
    torch.manual_seed(10)
    loss, _, metrics = model(**data)
    draft.moe_balance_alpha = 0
    torch.manual_seed(10)
    base_loss, _, base_metrics = model(**data)
    aux = metrics["log_window"]["sums"]["moe_balance_loss"]
    torch.testing.assert_close(loss, base_loss + 0.01 * aux)
    for name in ("ce_loss", "l1_loss", "confidence_loss"):
        torch.testing.assert_close(
            metrics["log_window"]["sums"][name],
            base_metrics["log_window"]["sums"][name],
        )
    loss.backward()
    assert all(layer.mlp.router.weight.grad is not None for layer in draft.layers)
    assert "moe_balance_loss" not in metrics["log_window"]["weights"]


def test_layer_checkpoint_gradients():
    from torch.utils.checkpoint import checkpoint

    m = DSparkDraftModel(config())
    data = inputs()
    plain, stats = m(**data, return_router_stats=True)
    (plain.square().sum() + sum(s["balance_loss"] for s in stats)).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    # Exercise the same nested layer return (hidden, statistics) under non-reentrant checkpointing.
    originals = [layer.forward for layer in m.layers]
    for layer, original in zip(m.layers, originals):
        layer.forward = lambda *a, _forward=original, **kw: checkpoint(
            _forward, *a, use_reentrant=False, **kw
        )
    actual, stats = m(**data, return_router_stats=True)
    (actual.square().sum() + sum(s["balance_loss"] for s in stats)).backward()
    torch.testing.assert_close(actual, plain)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n])


def ddp_worker(rank, root):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(99)
        model = SharedSparseMLP(config(), Qwen3MLP)
        # Force distinct experts on different ranks and across microbatches.
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.weight[:, :4].copy_(torch.eye(4))
        reference = deepcopy(model)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
        ref_opt = torch.optim.SGD(reference.parameters(), lr=0.01)
        for step in range(2):
            opt.zero_grad()
            ref_opt.zero_grad()
            for micro in range(3):
                for r in range(2):
                    x = torch.full((1, 4, 8), float(1 + micro + r))
                    c = torch.zeros(1, 2, 8)
                    c[..., (micro + r) % 4] = 3
                    keep = torch.ones(1, 2, dtype=torch.bool)
                    out, stats = reference(x, c, keep)
                    (
                        (out.square().mean() + 0.01 * stats["balance_loss"]) / 6
                    ).backward()
                    if r == rank:
                        with ddp.no_sync() if micro < 2 else nullcontext():
                            out, stats = ddp(x, c, keep)
                            (
                                (out.square().mean() + 0.01 * stats["balance_loss"]) / 3
                            ).backward()
            for a, b in zip(model.parameters(), reference.parameters()):
                assert a.grad is not None
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
            opt.step()
            ref_opt.step()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_ddp_sparse_experts_with_accumulation():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(ddp_worker, args=(root,), nprocs=2, join=True)


def test_disabled_moe_preserves_dense_initialization_and_output():
    enabled_config = config()
    enabled_config.dflash_config["moe"]["enabled"] = False
    dense_config = deepcopy(enabled_config)
    del dense_config.dflash_config["moe"]
    torch.manual_seed(17)
    disabled = DSparkDraftModel(enabled_config).eval()
    torch.manual_seed(17)
    dense = DSparkDraftModel(dense_config).eval()
    assert disabled.state_dict().keys() == dense.state_dict().keys()
    for key, value in disabled.state_dict().items():
        torch.testing.assert_close(value, dense.state_dict()[key], rtol=0, atol=0)
    data = inputs()
    torch.testing.assert_close(disabled(**data), dense(**data), rtol=0, atol=0)


def test_bfloat16_empty_experts_have_finite_gradients():
    m = SharedSparseMLP(config(), Qwen3MLP).bfloat16()
    with torch.no_grad():
        m.router.weight.zero_()  # all blocks select expert 0
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    c = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    out, stats = m(x, c, torch.ones(1, 2, dtype=torch.bool))
    (out.float().square().mean() + 0.01 * stats["balance_loss"]).backward()
    assert out.dtype == torch.bfloat16
    assert stats["balance_loss"].dtype == torch.float32
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for expert in m.experts[1:]:
        assert all(p.grad.count_nonzero() == 0 for p in expert.parameters())


def legacy_dispatch(model, x, context, keep):
    """Original per-expert where implementation, retained only as a reference."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        probs = F.linear(context.float(), model.router.weight.float()).softmax(-1)
    ids = probs.argmax(-1)
    gates = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    flat = x.reshape(-1, model.block_size, x.shape[-1])
    routed = torch.zeros_like(flat)
    for e, expert in enumerate(model.experts):
        indices = torch.where(keep.reshape(-1) & (ids.reshape(-1) == e))[0]
        values = expert(flat.index_select(0, indices))
        weight = gates.reshape(-1).index_select(0, indices).to(values.dtype)
        routed = routed.index_add(0, indices, values * weight[:, None, None])
    valid = keep.float()
    count = valid.sum()
    counts = (F.one_hot(ids, model.num_experts) * valid.unsqueeze(-1)).sum((0, 1))
    probability_sums = (probs * valid.unsqueeze(-1)).sum((0, 1))
    balance = (
        model.num_experts
        * (
            (counts / count.clamp_min(1)).detach()
            * probability_sums
            / count.clamp_min(1)
        ).sum()
    )
    return model.shared(x) + routed.reshape_as(x), balance


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_mode", ["mixed", "empty", "all"])
def test_grouped_dispatch_matches_legacy_outputs_and_all_gradients(dtype, mask_mode):
    torch.manual_seed(123)
    model = SharedSparseMLP(config(), Qwen3MLP).to(dtype)
    reference = deepcopy(model)
    x = torch.randn(2, 8, 8, dtype=dtype).requires_grad_()
    c = torch.randn(2, 4, 8, dtype=dtype).requires_grad_()
    rx = x.detach().clone().requires_grad_()
    rc = c.detach().clone().requires_grad_()
    keep = torch.tensor([[True, False, True, True], [False, True, False, True]])
    if mask_mode != "mixed":
        keep.fill_(mask_mode == "all")
    actual, stats = model(x, c, keep)
    expected, aux = legacy_dispatch(reference, rx, rc, keep)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(stats["balance_loss"], aux, rtol=0, atol=0)
    (actual.float().square().sum() + 0.01 * stats["balance_loss"]).backward()
    (expected.float().square().sum() + 0.01 * aux).backward()
    for a, b in [(x, rx), (c, rc), *zip(model.parameters(), reference.parameters())]:
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_inference_skips_auxiliary_statistics_without_changing_output():
    model = DSparkDraftModel(config()).eval()
    data = inputs()
    from unittest.mock import patch

    with (
        torch.no_grad(),
        patch.object(
            model.layers[0].mlp, "forward", wraps=model.layers[0].mlp.forward
        ) as call,
    ):
        actual = model(**data)
        assert call.call_args.kwargs["return_router_stats"] is False
        expected, stats = model(**data, return_router_stats=True)
        assert call.call_args.kwargs["return_router_stats"] is True
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert len(stats) == 2
    _, empty_stats = model.layers[0].mlp(
        data["noise_embedding"],
        torch.zeros(1, 2, 8),
        data["routing_mask"],
        return_router_stats=False,
    )
    assert empty_stats is None
