"""Compare deduplicated target distributions against the original logits path."""
import pytest
import torch
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel


def aligned_target_logits(model, target_hidden, safe_label_indices):
    """Dense teacher projection retained only as a test reference."""
    indices = (safe_label_indices - 1).clamp(min=0)
    batch, anchors, block_size = indices.shape
    width = target_hidden.size(-1)
    hidden = torch.gather(
        target_hidden, 1,
        indices.reshape(batch, anchors * block_size).unsqueeze(-1).expand(-1, -1, width),
    ).view(batch, anchors, block_size, width)
    return model.lm_head(hidden)


def model(mode='original', confidence=True, recompute=False):
    m = OnlineDSparkModel.__new__(OnlineDSparkModel)
    torch.nn.Module.__init__(m)
    m.lm_head = torch.nn.Linear(8, 19, bias=False).requires_grad_(False)
    m.block_size = 7
    m.loss_decay_gamma = 4.0
    m.dspark_loss_mode = mode
    m.dspark_ce_loss_alpha = 0.1
    m.dspark_l1_loss_alpha = 0.0 if mode == 'ce' else 0.9
    if mode == 'ce':
        m.dspark_loss_mode = 'original'
    m.dspark_kl_loss_alpha = 1.0
    m.dspark_confidence_head_alpha = float(confidence)
    m.recompute_loss = recompute
    return m


def assert_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree(a[k], b[k])
    elif isinstance(a, list):
        for x, y in zip(a, b):
            assert_tree(x, y)
    else:
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('mode', ['original', 'ce', 'kl'])
@pytest.mark.parametrize('confidence', [False, True])
@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_loss_metrics_and_gradients(mode, confidence, recompute, training):
    torch.manual_seed(23)
    m = model(mode, confidence, recompute).train(training)
    hidden = torch.randn(2, 12, 8)
    # Overlap, duplicate anchors, clamped sequence tails and position zero.
    labels = (torch.tensor([[0, 1, 2, 9], [1, 1, 3, 10]])[..., None]
              + torch.arange(7)).clamp(max=11)
    target_ids = torch.randint(19, (2, 4, 7))
    mask = torch.ones(2, 4, 7, dtype=torch.bool)
    mask[0, 3, 2:] = False
    mask[1, 1] = False
    logits = torch.randn(2, 4, 7, 19, requires_grad=True)
    conf = torch.randn(2, 4, 7, requires_grad=True) if confidence else None
    baseline = aligned_target_logits(m, hidden, labels)
    unique = m._unique_target_probs(hidden, labels)
    assert not unique[0].requires_grad
    assert unique[0].shape[0] < labels.numel()
    torch.testing.assert_close(unique[0][unique[1]], baseline.float().softmax(-1))
    kwargs = dict(draft_logits=logits, target_ids=target_ids, eval_mask=mask, confidence_pred=conf)
    dense_probs = baseline.float().softmax(-1).reshape(-1, baseline.size(-1))
    dense_inverse = torch.arange(labels.numel()).reshape_as(labels)
    dense_loss, dense_metrics = m._compute_dspark_loss(
        **kwargs, unique_target_probs=(dense_probs, dense_inverse),
    )
    new_loss, new_metrics = m._compute_dspark_loss(**kwargs, unique_target_probs=unique)
    assert_tree(dense_loss, new_loss)
    assert_tree(dense_metrics, new_metrics)
    inputs = (logits, conf) if conf is not None else (logits,)
    old_grad = torch.autograd.grad(dense_loss, inputs, retain_graph=True)
    new_grad = torch.autograd.grad(new_loss, inputs)
    for a, b in zip(old_grad, new_grad):
        assert_tree(a, b)


def test_projection_count_chunking_and_batch_identity():
    torch.manual_seed(5)
    m = model()
    hidden = torch.randn(2, 1030, 8)
    labels = torch.arange(1, 1031).view(1, 1030, 1).expand(2, -1, 7)
    rows = []
    handle = m.lm_head.register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
    probs, inverse = m._unique_target_probs(hidden, labels)
    handle.remove()
    assert sum(rows) == 2060  # not 2 * 1030 * 7
    assert max(rows) <= 1024
    torch.testing.assert_close(probs[inverse[:, :, 0]], m.lm_head(hidden).float().softmax(-1))
    assert not torch.equal(probs[inverse[0, :, 0]], probs[inverse[1, :, 0]])
    assert m._unique_target_probs(None, labels) is None


def test_bfloat16_distribution():
    m = model().bfloat16()
    hidden = torch.randn(2, 16, 8).bfloat16()
    labels = torch.tensor([[[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]]]*2)
    probs, inverse = m._unique_target_probs(hidden, labels)
    expected = aligned_target_logits(m, hidden, labels).float().softmax(-1)
    assert probs.dtype == torch.float32
    torch.testing.assert_close(probs[inverse], expected, atol=1e-4, rtol=1e-3)


def test_forward_routes_through_unique_projection():
    from unittest.mock import patch
    from tests.test_utils.test_dflash_losses import _make_dspark_model, _sample_tensors
    logits, ids, mask, hidden, anchors, keep = _sample_tensors()
    head = torch.nn.Linear(4, logits.size(-1), bias=False).double().requires_grad_(False)
    m = _make_dspark_model(logits, anchors, keep, lm_head=head,
                          dspark_l1_loss_alpha=.9, dspark_confidence_head_alpha=0)
    target = torch.randn_like(hidden)
    with patch.object(m, '_unique_target_probs', wraps=m._unique_target_probs) as project:
        loss, acc, metrics = m(ids, hidden, mask, target)
    project.assert_called_once()
    assert torch.isfinite(loss) and torch.isfinite(acc)
    assert 'tau_probabilistic' in metrics['log_window']['sums']
