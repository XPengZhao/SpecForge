"""Concat Engram initialization, causal lookup and HF serialization."""
import torch
from transformers import Qwen3Config
from specforge.modeling.draft.dspark import DSparkDraftModel
from tests.test_runtime.test_dspark_markov_engram import config


def test_concat_baseline_initialization_and_roundtrip(tmp_path):
    c = config()
    c.dflash_config['engram']['fusion'] = 'concat'
    torch.manual_seed(42)
    draft = DSparkDraftModel(c)
    after = torch.get_rng_state()
    torch.manual_seed(42)
    baseline = DSparkDraftModel(config('vanilla'))
    assert torch.equal(after, torch.get_rng_state())
    head = draft.markov_head
    assert head.markov_w2.in_features == 4 + 16 * 2
    assert not hasattr(head.memory, 'query')
    ids = torch.tensor([[1, 2, 3, 4]])
    buckets = head.memory.bucket_ids(ids)
    h = torch.randn(1, 4, 16)
    actual = head.compute_step_bias(ids, h, bucket_ids=buckets)
    torch.testing.assert_close(actual, baseline.markov_head.compute_step_bias(ids, h))
    actual.square().sum().backward()
    assert head.markov_w2.weight.grad[:, 4:].abs().sum() > 0
    with torch.no_grad():
        head.markov_w2.weight[:, 4:].normal_()
    reference = head.project_bias(torch.cat((head.get_prev_embeddings(ids), head.memory(buckets, h)), -1))
    torch.testing.assert_close(head.compute_step_bias(ids, h, bucket_ids=buckets), reference)
    changed = ids.clone(); changed[:, -1] = 7
    assert torch.equal(buckets[:, :-1], head.memory.bucket_ids(changed)[:, :-1])
    draft.save_pretrained(tmp_path)
    restored = DSparkDraftModel.from_pretrained(tmp_path)
    torch.testing.assert_close(restored.markov_head.compute_step_bias(ids, h, bucket_ids=buckets), reference)
