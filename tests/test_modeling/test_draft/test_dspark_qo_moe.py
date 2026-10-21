"""Q/O sparse attention math, shared KV, causal routing and accumulation."""
from contextlib import nullcontext
from copy import deepcopy
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache

from specforge.modeling.draft.dflash import Qwen3DFlashAttention, apply_rotary_pos_emb
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.dspark import DSparkDraftModel
from tests.test_modeling.test_draft.test_dspark_moe import config as ffn_config, inputs


def config(experts=4, ffn=True):
    c = ffn_config()
    c.dflash_config['moe']['enabled'] = ffn
    c.dflash_config['attention_moe'] = dict(enabled=True, num_experts=experts,
                                           balance_loss_alpha=0.02)
    # Q width need not equal residual width (4096 vs 2560 in the real recipe).
    c.num_attention_heads = 4
    return c


def attention_inputs(dtype=torch.float32):
    torch.manual_seed(10)
    x, ctx, route = (torch.randn(2, n, 8, dtype=dtype) for n in (6, 3, 3))
    keep = torch.tensor([[True, True, False], [True, False, True]])
    angles = torch.randn(2, 9, 4, dtype=dtype)
    visible = torch.arange(9)[None, :] <= (3 + torch.arange(6)[:, None])
    mask = torch.zeros(2, 1, 6, 9, dtype=dtype).masked_fill(~visible, torch.finfo(dtype).min)
    return dict(hidden_states=x, target_hidden=ctx, routing_context=route,
                routing_mask=keep, position_embeddings=(angles.cos(), angles.sin()),
                attention_mask=mask)


def dense_reference(m, data):
    x, ctx = data['hidden_states'], data['target_hidden']
    block = m.qo_moe.block_size
    p = F.linear(data['routing_context'].float(), m.qo_moe.router.weight.float()).softmax(-1)
    ids = p.argmax(-1)
    masks = [((ids == e) & data['routing_mask']).repeat_interleave(block, 1).unsqueeze(-1)
             for e in range(m.qo_moe.num_experts)]
    # Deliberately compute every expert densely; this reference does not use
    # sparse dispatch or either Q/O projection method from the implementation.
    q = sum(expert(x) * mask for expert, mask in zip(m.qo_moe.q_experts, masks))
    q = m.q_norm(q.view(2, 6, -1, m.head_dim)).transpose(1, 2)
    kv_input = torch.cat([ctx, x], dim=1)
    k = m.k_norm(m.k_proj(kv_input).view(2, 9, -1, m.head_dim)).transpose(1, 2)
    v = m.v_proj(kv_input).view(2, 9, -1, m.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *data['position_embeddings'])
    k = k.repeat_interleave(m.num_key_value_groups, dim=1)
    v = v.repeat_interleave(m.num_key_value_groups, dim=1)
    scores = q @ k.transpose(-1, -2) * m.scaling + data['attention_mask']
    weights = scores.float().softmax(-1).to(q.dtype)
    z = (weights @ v).transpose(1, 2).reshape(2, 6, -1)
    gate = p.gather(-1, ids.unsqueeze(-1)).squeeze(-1).repeat_interleave(block, 1).unsqueeze(-1)
    out = sum(expert(z) * mask * gate.to(z.dtype)
              for expert, mask in zip(m.qo_moe.o_experts, masks))
    valid_p = p[data['routing_mask']]
    counts = F.one_hot(ids[data['routing_mask']], m.qo_moe.num_experts).float().sum(0)
    balance = m.qo_moe.num_experts * (counts / counts.sum() * valid_p.mean(0)).sum()
    return out, balance


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_attention_matches_dense_reference_outputs_aux_and_gradients(dtype):
    torch.manual_seed(8)
    m = Qwen3DFlashAttention(config(), 0, DEFAULT_DFLASH_KERNELS).to(dtype)
    reference = deepcopy(m)
    data = attention_inputs(dtype)
    actual, _, stats = m(**data, collect_router_stats=True)
    expected, aux = dense_reference(reference, data)
    tolerance = dict(atol=2e-3, rtol=0.06) if dtype == torch.bfloat16 else dict(atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(stats['balance_loss'], aux)
    (actual.float().square().mean() + .02 * stats['balance_loss']).backward()
    (expected.float().square().mean() + .02 * aux).backward()
    for (name, p), (_, ref) in zip(m.named_parameters(), reference.named_parameters()):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, ref.grad, **tolerance)
    assert m.qo_moe.router.weight.grad.abs().sum() > 0
    assert not hasattr(m, 'q_proj') and not hasattr(m, 'o_proj')


def test_task_loss_alone_trains_router_and_empty_experts_get_zero_gradients():
    m = Qwen3DFlashAttention(config(), 0, DEFAULT_DFLASH_KERNELS).bfloat16()
    with torch.no_grad():
        m.qo_moe.router.weight.zero_()
    data = attention_inputs(torch.bfloat16)
    out, _, stats = m(**data, collect_router_stats=True)
    out.float().square().sum().backward()
    assert m.qo_moe.router.weight.grad.abs().sum() > 0
    for experts in (m.qo_moe.q_experts, m.qo_moe.o_experts):
        for expert in experts[1:]:
            for p in expert.parameters():
                assert p.grad is not None and p.grad.count_nonzero() == 0
    m.zero_grad()
    data['routing_mask'].zero_()
    out, _, stats = m(**data, collect_router_stats=True)
    assert out.count_nonzero() == 0
    assert stats['balance_loss'] == 0 and stats['count'] == 0
    (out.float().sum() + stats['balance_loss']).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_e1_equals_dense_attention_and_cache_is_not_expert_replicated():
    c = config(experts=1)
    m = Qwen3DFlashAttention(c, 0, DEFAULT_DFLASH_KERNELS)
    dense_c = deepcopy(c)
    dense_c.dflash_config.pop('attention_moe')
    dense = Qwen3DFlashAttention(dense_c, 0, DEFAULT_DFLASH_KERNELS)
    for name in ('k_proj', 'v_proj', 'q_norm', 'k_norm'):
        getattr(dense, name).load_state_dict(getattr(m, name).state_dict())
    dense.q_proj.load_state_dict(m.qo_moe.q_experts[0].state_dict())
    dense.o_proj.load_state_dict(m.qo_moe.o_experts[0].state_dict())
    data = attention_inputs()
    data['routing_mask'].fill_(True)
    expected = dense(**data)[0]
    actual = m(**data)[0]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(m.qo_moe.q_experts[0].weight.grad, dense.q_proj.weight.grad)
    torch.testing.assert_close(m.qo_moe.o_experts[0].weight.grad, dense.o_proj.weight.grad)
    # An E=4 attention still appends one K/V tensor with num_key_value_heads.
    m4 = Qwen3DFlashAttention(config(), 0, DEFAULT_DFLASH_KERNELS).eval()
    cache = DynamicCache()
    from unittest.mock import patch
    data['attention_mask'] = None
    with patch.object(cache, 'update', wraps=cache.update) as update:
        m4(**data, past_key_values=cache)
        key, value = update.call_args.args[:2]
        assert update.call_count == 1
        assert key.shape == value.shape == (2, 1, 9, 4)
        assert cache.get_seq_length(0) == 9
        m4(**data, past_key_values=cache)
        assert cache.get_seq_length(0) == 18


@pytest.mark.parametrize('ffn', [True, False])
def test_roundtrip_causal_routing_inference_and_checkpoint_gradients(tmp_path, ffn):
    from torch.utils.checkpoint import checkpoint
    m = DSparkDraftModel(config(ffn=ffn))
    data = inputs()
    seen = []
    hook = m.layers[0].self_attn.qo_moe.register_forward_pre_hook(
        lambda _, args: seen.append(args[1].detach().clone()))
    plain, stats = m(**data, return_router_stats=True)
    projected = m.hidden_norm(m.fc(data['target_hidden']))
    torch.testing.assert_close(seen[-1][:, 0], projected[:, 1])
    assert seen[-1][:, 1].count_nonzero() == 0
    changed = deepcopy(data)
    changed['target_hidden'][:, 2:] += 100
    m(**changed)
    torch.testing.assert_close(seen[-1], seen[-2])
    cached = dict(position_ids=torch.tensor([[10, 11, 12, 13]]),
                  target_hidden=data['target_hidden'][:, :2],
                  noise_embedding=data['noise_embedding'][:, :2])
    m(**cached)
    torch.testing.assert_close(seen[-1][:, 0], projected[:, 1])
    hook.remove()
    def loss(y, rows):
        return y.square().sum() + sum(r['attention']['balance_loss']
               + r.get('balance_loss', 0) for r in rows)
    loss(plain, stats).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    originals = [layer.forward for layer in m.layers]
    for layer, forward in zip(m.layers, originals):
        layer.forward = lambda *a, _forward=forward, **kw: checkpoint(
            _forward, *a, use_reentrant=False, **kw)
    actual, stats = m(**data, return_router_stats=True)
    loss(actual, stats).backward()
    torch.testing.assert_close(actual, plain)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n])
    m.eval()
    with torch.no_grad():
        y = m(**data)
        torch.testing.assert_close(y, m(**data, return_router_stats=True)[0])
    m.save_pretrained(tmp_path)
    loaded = DSparkDraftModel.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(loaded(**data), y)
    assert len(loaded.layers[0].self_attn.qo_moe.q_experts) == 4
    assert loaded.layers[0].self_attn.qo_moe(data['noise_embedding'],
        torch.zeros(1, 2, 8), data['routing_mask'], return_router_stats=False)[1][2] is None


def test_attention_disabled_preserves_ffn_checkpoint_and_outputs():
    c = ffn_config()
    torch.manual_seed(4)
    baseline = DSparkDraftModel(c)
    disabled = deepcopy(c)
    disabled.dflash_config['attention_moe'] = dict(enabled=False)
    torch.manual_seed(4)
    model = DSparkDraftModel(disabled)
    assert baseline.state_dict().keys() == model.state_dict().keys()
    for key, p in baseline.state_dict().items():
        torch.testing.assert_close(p, model.state_dict()[key], atol=0, rtol=0)
    data = inputs()
    torch.testing.assert_close(model(**data), baseline(**data), atol=0, rtol=0)


def test_wrapper_auxiliary_terms_preserve_ffn_weight_and_task_metrics():
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
    draft = DSparkDraftModel(config())
    model = OnlineDSparkModel(draft, nn.Linear(8, 32, bias=False), nn.Embedding(32, 8),
                             mask_token_id=31, block_size=2, num_anchors=3,
                             attention_backend='sdpa')
    data = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
                hidden_states=torch.randn(1, 6, 16), loss_mask=torch.ones(1, 6),
                target_last_hidden_states=torch.randn(1, 6, 8))
    torch.manual_seed(10)
    loss, _, metrics = model(**data)
    draft.moe_balance_alpha = draft.attention_moe_balance_alpha = 0
    torch.manual_seed(10)
    base, _, base_metrics = model(**data)
    sums = metrics['log_window']['sums']
    torch.testing.assert_close(loss, base + .01 * sums['moe_balance_loss']
                               + .02 * sums['attention_moe_balance_loss'])
    for key in ('ce_loss', 'l1_loss', 'confidence_loss'):
        torch.testing.assert_close(sums[key], base_metrics['log_window']['sums'][key])
    loss.backward()
    assert all(layer.self_attn.qo_moe.router.weight.grad is not None for layer in draft.layers)
    assert all(layer.mlp.router.weight.grad is not None for layer in draft.layers)


def ddp_worker(rank, root):
    dist.init_process_group('gloo', init_method=f'file://{root}/init', rank=rank, world_size=2)
    try:
        torch.manual_seed(44)
        layer = DSparkDraftModel(config()).layers[0]
        with torch.no_grad():
            for router in (layer.self_attn.qo_moe.router, layer.mlp.router):
                router.weight.zero_()
                router.weight[:, :4].copy_(torch.eye(4))
        ref = deepcopy(layer)
        ddp = nn.parallel.DistributedDataParallel(layer)
        opt, ref_opt = torch.optim.AdamW(layer.parameters(), lr=.001), torch.optim.AdamW(ref.parameters(), lr=.001)
        for step in range(2):
            opt.zero_grad()
            ref_opt.zero_grad()
            for micro in range(3):
                for r in range(2):
                    context = torch.zeros(1, 2, 8)
                    context[..., (micro+r) % 4] = 3
                    args = dict(hidden_states=torch.full((1, 4, 8), float(1+micro+r)),
                                target_hidden=torch.arange(24).reshape(1, 3, 8).float()/24,
                                position_embeddings=(torch.ones(1, 7, 4), torch.zeros(1, 7, 4)),
                                routing_context=context, routing_mask=torch.ones(1, 2, dtype=torch.bool),
                                collect_router_stats=True)
                    def objective(result):
                        y, stats = result
                        return y.square().mean() + .01 * stats['balance_loss'] + .02 * stats['attention']['balance_loss']
                    (objective(ref(**args)) / 6).backward()
                    if r == rank:
                        with ddp.no_sync() if micro < 2 else nullcontext():
                            (objective(ddp(**args)) / 3).backward()
            for (name, p), (_, q) in zip(layer.named_parameters(), ref.named_parameters()):
                assert p.grad is not None, name
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
            opt.step()
            ref_opt.step()
            for p, q in zip(layer.parameters(), ref.parameters()):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='requires Gloo')
def test_combined_attention_and_ffn_ddp_accumulation():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(ddp_worker, args=(root,), nprocs=2, join=True)
