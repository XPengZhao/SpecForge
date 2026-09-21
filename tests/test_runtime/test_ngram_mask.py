"""Raw sidecar alignment and strictly pre-anchor MASK conditioning."""
import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from specforge.modeling.draft.dspark import DSparkDraftModel, NgramMaskEmbedding
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.common.dflash_family_data import normalize_offline_dspark_sample, build_dspark_collator
from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, read_deepspec_features
from tests.test_runtime.test_deepspec_cache import write_fixture


def sidecar(root):
    write_fixture(root)
    base = root / 'features/ngram_embedding'
    base.mkdir(parents=True)
    values = [torch.arange(n * 4).reshape(n, 4).bfloat16() + 100 * i for i, n in enumerate((5,3))]
    payload = b''.join(v.view(torch.uint8).numpy().tobytes() for v in values)
    (base / 'data.bin').write_bytes(payload)
    (base / 'samples.idx').write_bytes(struct.pack('<QIIQ',0,0,5,0) + struct.pack('<QIIQ',1,0,3,40))
    path = root / 'manifest.json'
    meta = json.loads(path.read_text())
    meta['extra_features'] = {'ngram_embedding': dict(version=1, path='features/ngram_embedding',
        dtype='bfloat16', hidden_size=4, num_samples=2, index_file='samples.idx',
        index_record_format='<QIIQ', index_record_size=24, token_alignment='same_position_as_input_ids',
        feature_stage='raw_ngram_lookup_concat_before_key_value_projection',
        shards=[dict(file_name='data.bin', num_bytes=len(payload))])}
    path.write_text(json.dumps(meta))
    return values


def test_sidecar_read_truncate_pad(tmp_path):
    expected = sidecar(tmp_path)
    baseline = DeepSpecCacheReader(tmp_path)
    assert 'ngram_embedding' not in next(iter(baseline)).feature_specs
    reader = DeepSpecCacheReader(tmp_path, max_len=4)
    reader.enable_ngram()
    samples = []
    for i, ref in enumerate(reader):
        raw = read_deepspec_features(ref, reader.feature_keys)
        torch.testing.assert_close(raw['ngram_embedding'], expected[i][:4])
        samples.append(normalize_offline_dspark_sample(raw, 4))
    batch = build_dspark_collator()(samples)
    assert batch['ngram_embedding'].shape == (2,4,4)
    assert not batch['ngram_embedding'][1,3].any()
    assert not batch['loss_mask'][1,3]
    del samples[1]['ngram_embedding']
    with pytest.raises(ValueError, match='Mixed'):
        build_dspark_collator()(samples)


@pytest.mark.parametrize('error', ['id', 'length', 'offset', 'stage', 'count'])
def test_sidecar_rejects_misalignment(tmp_path, error):
    sidecar(tmp_path)
    if error in ('id','length','offset'):
        p=tmp_path/'features/ngram_embedding/samples.idx'
        data=p.read_bytes()
        values=[0,0,5,0]
        values[{'id':0,'length':2,'offset':3}[error]] = 999
        p.write_bytes(struct.pack('<QIIQ',*values)+data[24:])
    else:
        p=tmp_path/'manifest.json';meta=json.loads(p.read_text())
        meta['extra_features']['ngram_embedding']['feature_stage' if error=='stage' else 'num_samples']='wrong'
        p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        r=DeepSpecCacheReader(tmp_path);r.enable_ngram();list(r)


def tiny_config(enabled=True):
    return Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,vocab_size=32,
        block_size=7,num_target_layers=48,dflash_config=dict(target_layer_ids=[45,46,47],
            projector_type='dspark',mask_token_id=0,markov_rank=4,ngram_mask=enabled))


def training_model(enabled=True):
    config=tiny_config(enabled);config._attn_implementation='sdpa'
    draft=DSparkDraftModel(config)
    return OnlineDSparkModel(draft_model=draft, target_lm_head=torch.nn.Linear(16,32,bias=False).requires_grad_(False),
        target_embed_tokens=torch.nn.Embedding(32,16).requires_grad_(False),mask_token_id=0,
        block_size=7,attention_backend='sdpa',num_anchors=3,dspark_confidence_head_alpha=0)


def test_zero_gate_baseline_and_training_gradients():
    torch.manual_seed(10)
    m=training_model();base=training_model(False)
    base.load_state_dict({k:v for k,v in m.state_dict().items() if 'ngram_mask.' not in k})
    ids=torch.randint(1,32,(1,12));aux=torch.randn(1,12,48);mask=torch.ones(1,12)
    ngram=torch.randn(1,12,16);target=torch.randn(1,12,16)
    anchors=torch.tensor([[0,2,4]]);keep=torch.ones_like(anchors,dtype=torch.bool)
    with patch.object(m,'_sample_anchor_positions',return_value=(anchors,keep)), patch.object(base,'_sample_anchor_positions',return_value=(anchors,keep)):
        old=base(ids,aux,mask,target)
        new=m(ids,aux,mask,target,ngram_embedding=ngram)
    torch.testing.assert_close(old[0],new[0],rtol=0,atol=0)
    new[0].backward()
    assert m.draft_model.ngram_mask.gate.grad.abs().sum()>0
    assert not m.draft_model.ngram_mask.proj.weight.grad.any()
    assert all(p.grad is None for p in m.lm_head.parameters())
    m.zero_grad()
    m.draft_model.ngram_mask.gate.data.fill_(.1)
    with patch.object(m,'_sample_anchor_positions',return_value=(anchors,keep)):
        m(ids,aux,mask,target,ngram_embedding=ngram)[0].backward()
    assert m.draft_model.ngram_mask.proj.weight.grad.abs().sum()>0
    clone=training_model();clone.load_state_dict(m.state_dict())
    torch.testing.assert_close(clone.draft_model.ngram_mask.gate,m.draft_model.ngram_mask.gate)


def test_anchor_minus_one_and_anchor_embedding_untouched():
    m=training_model();m.draft_model.ngram_mask.gate.data.fill_(1)
    ids=torch.randint(1,32,(1,8));aux=torch.randn(1,8,48);mask=torch.ones(1,8)
    ngram=torch.randn(1,8,16);anchors=torch.tensor([[0,3,6]]);keep=torch.tensor([[True,True,False]])
    contexts=[]
    handle=m.draft_model.ngram_mask.register_forward_pre_hook(lambda module,args: contexts.append(args[1].detach().clone()))
    _,_,before=m._forward_draft_blocks(ids,aux,mask,anchors,keep,ngram)
    changed=ngram.clone();changed[:,3:]+=100
    _,_,after=m._forward_draft_blocks(ids,aux,mask,anchors,keep,changed)
    handle.remove()
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    torch.testing.assert_close(contexts[0][0,1],ngram[0,2])
    assert not contexts[0][0,0].any() and not contexts[0][0,2].any()
    noise=m._create_noise_embed(ids,anchors,keep)
    out=m.draft_model.ngram_mask(noise,contexts[0]).reshape(1,3,7,16)
    torch.testing.assert_close(out[:,:,0],noise.reshape_as(out)[:,:,0],rtol=0,atol=0)
    with pytest.raises(ValueError,match='token-aligned'):
        m._forward_draft_blocks(ids,aux,mask,anchors,keep)


def test_runtime_reads_optional_sidecar_only_when_enabled(tmp_path):
    from specforge.launch import _read_offline_refs
    from specforge.algorithms.common.dflash_family_data import build_offline_dspark_reader
    from specforge.runtime.data_plane.feature_store import LocalFeatureStore
    from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
    from specforge.runtime.data_plane.ref_serialization import ref_from_dict, ref_to_dict
    sidecar(tmp_path)
    provider=SimpleNamespace(build_reader=build_offline_dspark_reader)
    kwargs=dict(run_id='test',ttt_length=7,max_len=4)
    base=_read_offline_refs(provider,str(tmp_path),**kwargs)
    assert 'ngram_embedding' not in base[0].feature_keys
    m=SimpleNamespace(draft_model=SimpleNamespace(ngram_mask_enabled=True))
    refs=_read_offline_refs(provider,str(tmp_path),model=m,**kwargs)
    refs=[ref_from_dict(ref_to_dict(ref)) for ref in refs]
    loader=FeatureDataLoader(LocalFeatureStore(),refs=refs,batch_size=2,
        collate_fn=build_dspark_collator(),per_sample_transform=lambda raw:normalize_offline_dspark_sample(raw,4),
        strategy='dspark',num_workers=2)
    batch=next(iter(loader))
    assert batch.tensors['ngram_embedding'].shape==(2,4,4)
    loader.close()


def test_strategy_forwards_ngram_feature():
    from specforge.training.strategies.base import DSparkTrainStrategy
    from specforge.runtime.contracts import TrainBatch
    m=training_model()
    tensors=dict(input_ids=torch.randint(1,32,(1,12)),hidden_states=torch.randn(1,12,48),
        loss_mask=torch.ones(1,12),target_last_hidden_states=torch.randn(1,12,16),
        ngram_embedding=torch.randn(1,12,16))
    out=DSparkTrainStrategy(m).forward_loss(TrainBatch(sample_ids=['a'],strategy='dspark',tensors=tensors))
    out.loss.backward()
    assert m.draft_model.ngram_mask.gate.grad is not None
