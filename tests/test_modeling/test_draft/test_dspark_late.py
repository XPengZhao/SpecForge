"""4+1 visibility, slot alignment, training gradients and exact greedy verification."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.dspark_late import prepare_bank, finish_block, training_forward
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel


def config(enabled=True):
    c = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                   num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=1,
                   head_dim=4, max_position_embeddings=128)
    c.block_size = 7
    c.architectures = ['DSparkDraftModel']
    c.num_target_layers = 2
    c.dflash_config = dict(projector_type='dspark', target_layer_ids=[0, 1],
                          markov_rank=4, confidence_head_alpha=1.,
                          confidence_head_with_markov=True,
                          late_conditioning=dict(enabled=enabled))
    c._attn_implementation = 'eager'
    return c


def test_boundary_selection_and_old_new_context_visibility():
    torch.manual_seed(7)
    model = DSparkDraftModel(config()).eval()
    target = torch.randn(1, 20, 16)
    old = torch.tensor([[3]])
    starts = old + 2
    keep = torch.ones_like(old, dtype=torch.bool)
    bank = prepare_bank(model, target, old, starts, keep)
    changed = target.clone()
    changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 10
    torch.testing.assert_close(bank, prepare_bank(model, changed, old, starts, keep), atol=0, rtol=0)
    assert bank.shape == (1, 1, 12, 8)
    # All seven acceptance boundaries select exactly six positions.
    for r in range(7):
        anchor = old + r + 1
        embedding = torch.randn(1, 1, 8)
        seen = []
        hook = model.layers[-1].register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs['hidden_states'].detach()), with_kwargs=True)
        output = finish_block(model, bank, target, embedding, anchor, starts, keep)
        hook.remove()
        torch.testing.assert_close(seen[0][:, 1:], bank[:, 0, r:r+6])
        assert output.shape == (1, 7, 8)
        future = target.clone()
        future[:, int(anchor.item()):] += 100
        torch.testing.assert_close(output, finish_block(model, bank, future, embedding, anchor, starts, keep))
        changed_bonus = finish_block(model, bank, target, embedding+1, anchor, starts, keep)
        assert not torch.allclose(output[:, 1:], changed_bonus[:, 1:])
    # Late-only target features affect the final layer, not the bank.
    anchor = old + 7
    assert not torch.allclose(
        finish_block(model, bank, target, embedding, anchor, starts, keep),
        finish_block(model, bank, changed, embedding, anchor, starts, keep))


def test_block_isolation_checkpoint_gradients_and_save_load(tmp_path):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(3)
    model = DSparkDraftModel(config())
    data = dict(target_hidden=torch.randn(1, 18, 16),
                anchor_embeddings=torch.randn(1, 2, 8), anchors=torch.tensor([[8, 13]]),
                keep=torch.tensor([[True, True]]), old_cutoffs=torch.tensor([[3, 8]]))
    out = model(late_inputs=data)
    changed = deepcopy(data)
    changed['anchor_embeddings'][:, 1] += 9
    torch.testing.assert_close(out[:, :7], model(late_inputs=changed)[:, :7])
    out.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads['future_mask'].abs().sum() > 0
    assert grads['anchor_adapter.weight'].abs().sum() > 0
    for i in range(5):
        assert grads[f'layers.{i}.self_attn.q_proj.weight'].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    for layer in model.layers:
        original = layer.forward
        layer.forward = lambda *a, _f=original, **kw: checkpoint(_f, *a, use_reentrant=False, **kw)
    actual = model(late_inputs=data)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, out)
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n])
    model.save_pretrained(tmp_path)
    loaded = DSparkDraftModel.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(loaded(late_inputs=data), out)
    with pytest.raises(ValueError, match='boundaries'):
        loaded(target_hidden=data['target_hidden'])


def test_training_loss_backwards_and_next_token_labels():
    model = DSparkDraftModel(config())
    wrapper = OnlineDSparkModel(model, nn.Linear(8, 32, bias=False), nn.Embedding(32, 8),
                               mask_token_id=31, block_size=7, num_anchors=3,
                               attention_backend='eager')
    ids = torch.arange(18).unsqueeze(0)
    data = dict(input_ids=ids, hidden_states=torch.randn(1, 18, 16),
                loss_mask=torch.ones(1, 18), target_last_hidden_states=torch.randn(1, 18, 8))
    loss, _, _ = wrapper(**data)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    labels, mask, _ = wrapper._build_dspark_labels_and_mask(
        ids, data['loss_mask'], torch.tensor([[5]]), torch.tensor([[True]]))
    assert labels.tolist() == [[[6, 7, 8, 9, 10, 11, 12]]]
    assert mask.all()


class ToyTarget(nn.Module):
    """Exact greedy sequence increments by one, with a real cache length."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.calls = []
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, input_ids, past_key_values, **kwargs):
        self.calls.append((past_key_values.get_seq_length(), input_ids.clone()))
        x = self.embedding(input_ids)
        kv = x[:, None]
        past_key_values.update(kv, kv, 0)
        logits = x.new_full((*input_ids.shape, 32), -100.)
        logits.scatter_(-1, ((input_ids+1) % 32).unsqueeze(-1), 100.)
        return SimpleNamespace(logits=logits, hidden_states=(x, x+1, x+2))


@pytest.mark.parametrize('accepted', range(7))
def test_sync_generation_all_boundaries_matches_target_greedy(accepted):
    model = DSparkDraftModel(config())
    target = ToyTarget()
    def proposals(logits, *, first_prev_token_ids, **kwargs):
        p = (first_prev_token_ids[:, None] + torch.arange(1, 7)) % 32
        if accepted < 6:
            p[:, accepted] = (p[:, accepted]+5) % 32
        return p, logits
    model.markov_head.sample_block_tokens = proposals
    result = model.spec_generate(target, torch.tensor([[1, 2]]), 19, [], 0., return_stats=True)
    assert result['output_ids'].tolist() == [list(range(1, 22))]
    assert result['accepted'] == accepted * result['rounds']
    for cached, block in target.calls[1:]:
        # Rejected suffixes have been removed from the cache before next verify.
        assert cached == int(block[0, 0]) - 1
    stopped = model.spec_generate(target, torch.tensor([[1, 2]]), 19, [8], 0.)
    assert stopped.tolist() == [list(range(1, 9))]
    torch.testing.assert_close(model.spec_generate(target, torch.tensor([[1, 2]]), 0, [], 0.), torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match='greedy'):
        model.spec_generate(target, torch.tensor([[1, 2]]), 3, [], .7)


def test_disabled_feature_preserves_dense_model():
    c = config(False)
    torch.manual_seed(42)
    disabled = DSparkDraftModel(c)
    del c.dflash_config['late_conditioning']
    torch.manual_seed(42)
    baseline = DSparkDraftModel(c)
    assert disabled.state_dict().keys() == baseline.state_dict().keys()
    for n, p in disabled.state_dict().items():
        torch.testing.assert_close(p, baseline.state_dict()[n], atol=0, rtol=0)


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_bfloat16_masked_blocks_and_export(tmp_path, backend):
    from specforge.export.to_hf import export_to_hf
    c = config()
    c._attn_implementation = backend
    model = DSparkDraftModel(c).bfloat16()
    wrapper = OnlineDSparkModel(model, nn.Linear(8, 32, bias=False).bfloat16(),
                               nn.Embedding(32, 8).bfloat16(), mask_token_id=31,
                               block_size=7, num_anchors=3, attention_backend=backend)
    data = dict(input_ids=torch.arange(10).unsqueeze(0),
                hidden_states=torch.randn(1, 10, 16).bfloat16(),
                loss_mask=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]]),
                target_last_hidden_states=torch.randn(1, 10, 8).bfloat16())
    loss, _, _ = wrapper(**data)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    checkpoint = tmp_path / 'checkpoint.pt'
    cfgpath = tmp_path / 'draft.json'
    c.to_json_file(cfgpath)
    torch.save({'draft_state_dict': model.state_dict()}, checkpoint)
    export_to_hf(str(checkpoint), str(cfgpath), str(tmp_path / 'export'))
    loaded = DSparkDraftModel.from_pretrained(tmp_path / 'export')
    assert loaded.late_conditioning
    torch.testing.assert_close(loaded.future_mask.float(), model.future_mask.float())


def accumulation_worker(rank, root):
    import torch.distributed as dist
    from contextlib import nullcontext
    dist.init_process_group('gloo', init_method=f'file://{root}/init', rank=rank, world_size=2)
    try:
        torch.manual_seed(13)
        model = DSparkDraftModel(config())
        reference = deepcopy(model)
        ddp = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=.001)
        def objective(m, data):
            y = m(late_inputs=data)
            base = m.module if hasattr(m, 'module') else m
            ids = torch.arange(7).unsqueeze(0)
            logits = base.apply_logits_head(torch.zeros(1, 7, 32), prev_token_ids=ids, hidden_states=y)
            confidence = base.predict_confidence(y, prev_token_ids=ids)
            return y.square().mean()+logits.square().mean()+confidence.square().mean()
        for step in range(2):
            opt.zero_grad()
            ref_opt.zero_grad()
            for micro in range(2):
                for r in range(2):
                    torch.manual_seed(100+step*20+micro*2+r)
                    data = dict(target_hidden=torch.randn(1, 15, 16),
                                anchor_embeddings=torch.randn(1, 1, 8), anchors=torch.tensor([[8]]),
                                old_cutoffs=torch.tensor([[1+micro+r]]), keep=torch.tensor([[True]]))
                    (objective(reference, data)/4).backward()
                    if r == rank:
                        with ddp.no_sync() if micro == 0 else nullcontext():
                            (objective(ddp, data)/2).backward()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
            opt.step()
            ref_opt.step()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason='requires Gloo')
def test_ddp_accumulation_and_optimizer_updates():
    import tempfile
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(accumulation_worker, args=(root,), nprocs=2, join=True)


def test_reference_with_real_tiny_qwen_target():
    from transformers import Qwen3ForCausalLM
    torch.manual_seed(19)
    c = config()
    c.num_hidden_layers = 2
    target = Qwen3ForCausalLM(c).eval()
    draft = DSparkDraftModel(config()).eval()
    prompt = torch.tensor([[1, 2, 3]])
    actual = draft.spec_generate(target, prompt, 8, [], 0.)
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(8):
            expected = torch.cat((expected, target(expected).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA FlexAttention')
def test_cuda_flex_training_smoke():
    c = config()
    c._attn_implementation = 'flex_attention'
    model = DSparkDraftModel(c).cuda().bfloat16()
    wrapper = OnlineDSparkModel(model, nn.Linear(8, 32, bias=False).cuda().bfloat16(),
                               nn.Embedding(32, 8).cuda().bfloat16(), mask_token_id=31,
                               block_size=7, num_anchors=3, attention_backend='flex_attention')
    loss, _, _ = wrapper(input_ids=torch.arange(18, device='cuda').unsqueeze(0),
                         hidden_states=torch.randn(1, 18, 16, device='cuda', dtype=torch.bfloat16),
                         loss_mask=torch.ones(1, 18, device='cuda'),
                         target_last_hidden_states=torch.randn(1, 18, 8, device='cuda', dtype=torch.bfloat16))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
