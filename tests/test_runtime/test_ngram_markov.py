"""Raw Engram sidecar alignment and causal Markov conditioning."""

import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_data import (
    build_dspark_collator,
    normalize_offline_dspark_sample,
)
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import (
    DSparkDraftModel,
    NgramMarkovHead,
)
from specforge.runtime.data_plane.deepspec_cache import (
    DeepSpecCacheReader,
    read_deepspec_features,
)
from tests.test_runtime.test_deepspec_cache import write_fixture


def sidecar(root):
    write_fixture(root)
    base = root / "features/ngram_embedding"
    base.mkdir(parents=True)
    values = [
        torch.arange(n * 4).reshape(n, 4).bfloat16() + 100 * i
        for i, n in enumerate((5, 3))
    ]
    payload = b"".join(v.view(torch.uint8).numpy().tobytes() for v in values)
    (base / "data.bin").write_bytes(payload)
    (base / "samples.idx").write_bytes(
        struct.pack("<QIIQ", 0, 0, 5, 0) + struct.pack("<QIIQ", 1, 0, 3, 40)
    )
    path = root / "manifest.json"
    meta = json.loads(path.read_text())
    meta["extra_features"] = {
        "ngram_embedding": dict(
            version=1,
            path="features/ngram_embedding",
            dtype="bfloat16",
            hidden_size=4,
            num_samples=2,
            index_file="samples.idx",
            index_record_format="<QIIQ",
            index_record_size=24,
            token_alignment="same_position_as_input_ids",
            feature_stage="raw_ngram_lookup_concat_before_key_value_projection",
            shards=[dict(file_name="data.bin", num_bytes=len(payload))],
        )
    }
    path.write_text(json.dumps(meta))
    return values


def tiny_config(enabled=True):
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        block_size=7,
        num_target_layers=48,
        dflash_config=dict(
            target_layer_ids=[45, 46, 47],
            projector_type="dspark",
            mask_token_id=0,
            markov_rank=4,
            markov_head_type="ngram" if enabled else "vanilla",
        ),
    )


def training_model(enabled=True):
    config = tiny_config(enabled)
    config._attn_implementation = "sdpa"
    draft = DSparkDraftModel(config)
    return OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(16, 32, bias=False).requires_grad_(False),
        target_embed_tokens=torch.nn.Embedding(32, 16).requires_grad_(False),
        mask_token_id=0,
        block_size=7,
        attention_backend="sdpa",
        num_anchors=3,
        dspark_confidence_head_alpha=0,
    )


def test_sidecar_read_truncate_pad(tmp_path):
    expected = sidecar(tmp_path)
    baseline = DeepSpecCacheReader(tmp_path)
    assert "ngram_embedding" not in next(iter(baseline)).feature_specs
    reader = DeepSpecCacheReader(tmp_path, max_len=4)
    reader.enable_ngram()
    samples = []
    for i, ref in enumerate(reader):
        raw = read_deepspec_features(ref, reader.feature_keys)
        torch.testing.assert_close(raw["ngram_embedding"], expected[i][:4])
        samples.append(normalize_offline_dspark_sample(raw, 4))
    batch = build_dspark_collator()(samples)
    assert batch["ngram_embedding"].shape == (2, 4, 4)
    assert not batch["ngram_embedding"][1, 3].any()
    del samples[1]["ngram_embedding"]
    with pytest.raises(ValueError, match="Mixed"):
        build_dspark_collator()(samples)


@pytest.mark.parametrize("error", ["id", "length", "offset", "stage", "count"])
def test_sidecar_rejects_misalignment(tmp_path, error):
    sidecar(tmp_path)
    if error in ("id", "length", "offset"):
        path = tmp_path / "features/ngram_embedding/samples.idx"
        data = path.read_bytes()
        values = [0, 0, 5, 0]
        values[{"id": 0, "length": 2, "offset": 3}[error]] = 999
        path.write_bytes(struct.pack("<QIIQ", *values) + data[24:])
    else:
        path = tmp_path / "manifest.json"
        meta = json.loads(path.read_text())
        key = "feature_stage" if error == "stage" else "num_samples"
        meta["extra_features"]["ngram_embedding"][key] = "wrong"
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        reader = DeepSpecCacheReader(tmp_path)
        reader.enable_ngram()
        list(reader)


def test_zero_projection_matches_vanilla_and_receives_gradient():
    torch.manual_seed(10)
    model = training_model()
    baseline = training_model(False)
    state = {
        key: value
        for key, value in model.state_dict().items()
        if "markov_head.ngram_" not in key
    }
    baseline.load_state_dict(state)
    ids = torch.randint(1, 32, (1, 12))
    aux = torch.randn(1, 12, 48)
    mask = torch.ones(1, 12)
    ngram = torch.randn(1, 12, 16)
    target = torch.randn(1, 12, 16)
    anchors = torch.tensor([[0, 2, 4]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    with (
        patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)),
        patch.object(
            baseline, "_sample_anchor_positions", return_value=(anchors, keep)
        ),
    ):
        original = baseline(ids, aux, mask, target)
        enhanced = model(ids, aux, mask, target, ngram_embedding=ngram)
    torch.testing.assert_close(original[0], enhanced[0], rtol=0, atol=0)
    enhanced[0].backward()
    grad = model.draft_model.markov_head.ngram_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_training_uses_previous_token_same_position_ngram():
    model = training_model()
    with torch.no_grad():
        model.draft_model.markov_head.ngram_proj.weight.fill_(0.01)
    ids = torch.randint(1, 32, (1, 12))
    aux = torch.randn(1, 12, 48)
    mask = torch.ones(1, 12)
    ngram = torch.arange(12 * 16).reshape(1, 12, 16).float()
    target = torch.randn(1, 12, 16)
    anchors = torch.tensor([[0, 2, 4]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    captured = []
    handle = model.draft_model.markov_head.ngram_norm.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    with patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
        model(ids, aux, mask, target, ngram_embedding=ngram)
    handle.remove()
    expected_indices = anchors.unsqueeze(-1) + torch.arange(7)
    expected = (
        ngram[:, None]
        .expand(-1, 3, -1, -1)
        .gather(2, expected_indices[..., None].expand(-1, -1, -1, 16))
    )
    torch.testing.assert_close(captured[0], expected)


def test_sampler_recomputes_ngram_after_each_sample():
    head = NgramMarkovHead(
        vocab_size=8,
        markov_rank=4,
        hidden_size=4,
        rms_norm_eps=1e-6,
    )
    with torch.no_grad():
        head.markov_w1.weight.zero_()
        head.markov_w2.weight.zero_()
    base_logits = torch.zeros(1, 2, 8)
    base_logits[0, 0, 2] = 10
    base_logits[0, 1, 3] = 10
    prefixes = []

    def lookup(prefix):
        prefixes.append(prefix.clone())
        return torch.ones(prefix.size(0), 4)

    tokens, _ = head.sample_block_tokens(
        base_logits,
        prefix_token_ids=torch.tensor([[5, 6]]),
        hidden_states=None,
        ngram_lookup=lookup,
    )
    assert tokens.tolist() == [[2, 3]]
    assert [value.tolist() for value in prefixes] == [
        [[5, 6]],
        [[5, 6, 2]],
    ]


def test_runtime_reads_sidecar_only_when_enabled(tmp_path):
    from specforge.algorithms.common.dflash_family_data import (
        build_offline_dspark_reader,
    )
    from specforge.launch import _read_offline_refs

    sidecar(tmp_path)
    provider = SimpleNamespace(build_reader=build_offline_dspark_reader)
    kwargs = dict(run_id="test", ttt_length=7, max_len=4)
    baseline = _read_offline_refs(provider, str(tmp_path), **kwargs)
    assert "ngram_embedding" not in baseline[0].feature_keys
    model = SimpleNamespace(draft_model=SimpleNamespace(ngram_markov_enabled=True))
    refs = _read_offline_refs(provider, str(tmp_path), model=model, **kwargs)
    assert "ngram_embedding" in refs[0].feature_keys
