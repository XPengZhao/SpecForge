"""Qwen3.8 cache contract and metadata-only frozen target loading."""
import copy
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.prepare_qwen38_dspark import prepare
from specforge.modeling.target.offline_config import load_offline_target_config
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, read_deepspec_features


@pytest.fixture
def checkpoint_cache(tmp_path):
    target, cache = tmp_path / 'target', tmp_path / 'cache'
    target.mkdir()
    cache.mkdir()
    config = dict(model_type='qwen4_exp', tie_word_embeddings=False,
                  text_config=dict(hidden_size=2560, vocab_size=8, num_hidden_layers=48,
                                   hc_count=4, pad_token_id=0, eos_token_id=2,
                                   num_attention_heads=24, num_key_value_heads=2, head_dim=256))
    (target / 'config.json').write_text(json.dumps(config))
    (target / 'tokenizer_config.json').write_text(json.dumps({'added_tokens_decoder': {'3': {'content': '<|fim_pad|>'}}}))
    weights = {'model.language_model.embed_tokens.weight': torch.randn(8, 2560),
               'lm_head.weight': torch.randn(8, 2560)}
    save_file(weights, str(target / 'model.safetensors'))
    fields = [torch.tensor([1, 2], dtype=torch.int32), torch.ones(2, dtype=torch.uint8),
              torch.tensor([0, 1], dtype=torch.uint8), torch.randn(2, 7680).bfloat16(),
              torch.randn(2, 2560).bfloat16()]
    data, offsets = bytearray(), []
    for t in fields:
        offsets.append(len(data))
        data.extend(t.view(torch.uint8).numpy().tobytes())
    (cache / 'shard-00000.bin').write_bytes(data)
    (cache / 'samples.idx').write_bytes(struct.pack('<QIIQQQQQ', 0, 0, 2, *offsets))
    manifest = dict(version=2, index_record_size=56, hidden_dtype='bfloat16', token_dtype='int32',
                    mask_dtype='uint8', hidden_size=2560, target_layer_ids=[45,46,47],
                    target_model_name_or_path=str(target), num_samples=1, num_shards=1,
                    shards=[dict(shard_id=0, file_name='shard-00000.bin')],
                    aux_reduction='post_layer_hc_mean_fp32_to_bfloat16',
                    target_final_feature='last_hidden_state_after_hyper_connection_mixer',
                    hc_count=4, capture_scope='full_sequence', loss_mask_scope='assistant_response',
                    enable_thinking=False, target_config=config,
                    extra_features={'ngram': {'file_name': 'not-read.bin'}})
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    return target, cache, config, fields, weights


def test_setup_and_frozen_weights(checkpoint_cache, tmp_path):
    target, cache, _, fields, weights = checkpoint_cache
    draft, train = prepare(target, cache, tmp_path / 'setup', world_size=1, global_batch=1)
    assert draft['dflash_config']['target_layer_ids'] == [45,46,47]
    assert draft['dflash_config']['mask_token_id'] == 3
    assert draft['vocab_size'] == 8
    assert (draft['num_attention_heads'], draft['num_key_value_heads'], draft['head_dim']) == (24, 2, 256)
    assert draft['num_hidden_layers'] == len(draft['layer_types']) == 3
    assert train['training']['total_steps'] == 10
    assert train['training']['max_steps'] == 1
    config = load_offline_target_config(target)
    model = TargetEmbeddingsAndHead.from_pretrained(str(target), config=config,
                embed_key=train['model']['embedding_key'], device='cpu', dtype=torch.float32)
    assert all(not p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model.embed_tokens.weight, weights['model.language_model.embed_tokens.weight'])
    # The stored final mixer output goes directly through the LM head: no extra norm.
    torch.testing.assert_close(model.lm_head(fields[-1].float()), fields[-1].float() @ weights['lm_head.weight'].T)
    reader = DeepSpecCacheReader(cache)
    features = read_deepspec_features(next(iter(reader)), reader.feature_keys)
    torch.testing.assert_close(features['aux_hidden_state'], fields[-2])
    torch.testing.assert_close(features['hidden_state'], fields[-1])
    assert 'ngram' not in features


@pytest.mark.parametrize('key,value', [('aux_reduction','mean_bf16'), ('target_final_feature','pre_mixer'),
                                       ('capture_scope','response'), ('enable_thinking',True)])
def test_wrong_cache_semantics_rejected(checkpoint_cache, key, value):
    _, cache, config, _, _ = checkpoint_cache
    reader = DeepSpecCacheReader(cache)
    reader.manifest[key] = value
    with pytest.raises(ValueError, match=key):
        reader.validate_qwen38(config)


def test_wrong_layers_and_vocab_rejected(checkpoint_cache):
    _, cache, config, _, _ = checkpoint_cache
    reader = DeepSpecCacheReader(cache)
    reader.layers = [46,47]
    with pytest.raises(ValueError, match='layers'):
        reader.validate_qwen38(config)
    reader.layers = [45,46,47]
    config = copy.deepcopy(config)
    config['text_config']['vocab_size'] += 1
    with pytest.raises(ValueError, match='vocab_size'):
        reader.validate_qwen38(config)


def test_three_aux_draft_backward():
    from transformers import Qwen3Config
    from specforge.modeling.draft.dspark import DSparkDraftModel
    from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask

    config = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                         vocab_size=32, block_size=7, num_target_layers=48,
                         dflash_config=dict(target_layer_ids=[45,46,47], mask_token_id=0,
                                            projector_type='dspark', markov_rank=4,
                                            enable_confidence_head=True))
    config._attn_implementation = 'sdpa'
    model = DSparkDraftModel(config)
    aux = torch.randn(1, 10, 48, requires_grad=True)
    anchors = torch.tensor([[2]])
    mask = create_dflash_sdpa_mask(anchors, torch.ones_like(anchors, dtype=torch.bool),
                                   S=10, block_size=7, device=anchors.device)
    hidden = model(position_ids=torch.cat([torch.arange(10), torch.arange(2,9)])[None],
                   attention_mask=mask, noise_embedding=torch.randn(1,7,16), target_hidden=aux)
    logits = model.apply_logits_head(torch.randn(1,7,32), prev_token_ids=torch.ones(1,7,dtype=torch.long),
                                     hidden_states=hidden)
    (hidden.square().mean() + logits.square().mean()).backward()
    assert torch.isfinite(aux.grad).all()
    assert (aux.grad.reshape(1,10,3,16).abs().sum(dim=(0,1,3)) > 0).all()
    assert any(p.grad is not None for p in model.parameters())


def test_metadata_loader_rejects_quantization_and_preserves_native_config(tmp_path):
    from transformers import Qwen3Config
    (tmp_path / 'config.json').write_text(json.dumps(dict(model_type='qwen3', hidden_size=16)))
    assert isinstance(load_offline_target_config(tmp_path), Qwen3Config)
    raw = dict(model_type='qwen4_exp', text_config=dict(hidden_size=2560, vocab_size=8,
                    num_hidden_layers=48, hc_count=4), quantization_config={'quant_method': 'fp8'})
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='unquantized'):
        load_offline_target_config(tmp_path)
