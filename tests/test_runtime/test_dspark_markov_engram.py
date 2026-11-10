import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.dspark.providers import resume_contract
from specforge.export.checkpoint_io import materialize_draft
from specforge.cli import main as specforge_cli
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.training.model_loading import warm_start_draft_model
from specforge.training.strategies.base import DSparkTrainStrategy


def config(head="engram"):
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/qwen3-4b-dspark-markov-engram.json"
    )
    data = json.loads(path.read_text())
    data.update(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        vocab_size=23,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        num_target_layers=4,
    )
    options = data["dflash_config"]
    options.update(
        markov_head_type=head, markov_rank=4, mask_token_id=22, target_layer_ids=[1, 2]
    )
    options["engram"]["bucket_counts"] = [31, 37, 41, 43]
    options["engram"]["head_dim"] = 2
    if head == "vanilla":
        del options["engram"]
    c = Qwen3Config.from_dict(data)
    c._attn_implementation = "sdpa"
    return c


def bucket_reference(prefix, options):
    out = []
    for i, count in enumerate(options["bucket_counts"]):
        n = 2 if i < 2 else 3
        if len(prefix) < n:
            out.append(-1)
            continue
        mixed = 0
        for token, multiplier in zip(
            reversed(prefix[-n:]), options["hash_multipliers"]
        ):
            mixed ^= token * multiplier
        out.append(mixed % count)
    return out


def training(draft):
    return OnlineDSparkModel(
        draft,
        torch.nn.Linear(16, 23, bias=False).requires_grad_(False),
        torch.nn.Embedding(23, 16).requires_grad_(False),
        mask_token_id=22,
        block_size=7,
        num_anchors=2,
        attention_backend="sdpa",
    )


def test_engram_training_alignment_gradient_and_no_future_leakage():
    torch.manual_seed(11)
    c = config()
    draft = DSparkDraftModel(c)
    model = training(draft)
    memory = draft.markov_head.memory
    assert memory.value.weight.count_nonzero() == 0
    anchors = torch.tensor([[0, 3], [1, 8]])
    keep = torch.tensor([[True, True], [True, False]])
    model._sample_anchor_positions = lambda *args: (anchors, keep)
    ids = torch.randint(22, (2, 12))
    mask = torch.ones_like(ids)
    mask[1, 9:] = 0
    features = torch.randn(2, 12, 32)
    final = torch.randn(2, 12, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    loss, _, _ = model(ids, features, mask, final)
    loss.backward()
    assert memory.value.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()

    with patch.object(
        draft, "apply_logits_head", wraps=draft.apply_logits_head
    ) as head:
        loss, _, stats = model(ids, features, mask, final)
        loss.backward()
    for name, param in memory.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name
    assert stats["log_window"]["denoms"]["mtp_7_accuracy"] == 3

    base = head.call_args.args[0]
    kw = head.call_args.kwargs
    parallel = draft.apply_logits_head(base, **kw)
    for b, n in keep.nonzero().tolist():
        a = anchors[b, n].item()
        for i in range(7):
            p = a + i
            prefix = ids[b, : p + 1].tolist()
            expected = torch.tensor(bucket_reference(prefix, c.dflash_config["engram"]))
            torch.testing.assert_close(kw["engram_bucket_ids"][b, n, i], expected)
            hidden = kw["hidden_states"][b, n, i]
            sequential = base[b, n, i] + draft.markov_head.compute_step_bias(
                ids[b, p],
                hidden,
                bucket_ids=expected,
            )
            torch.testing.assert_close(parallel[b, n, i], sequential)
    # Mutate the first anchor's next token and all later tokens. Its logits
    # should be unchanged even though the second packed block sees other input.
    changed = ids.clone()
    changed[0, 1:] = (ids[0, 1:] + 1) % 22
    with patch.object(
        draft, "apply_logits_head", wraps=draft.apply_logits_head
    ) as head:
        model(changed, features, mask, final)
    after = draft.apply_logits_head(*head.call_args.args, **head.call_args.kwargs)
    torch.testing.assert_close(after[0, 0, 0], parallel[0, 0, 0], rtol=0, atol=0)


@pytest.mark.parametrize("hf_source", [False, True])
def test_vanilla_warm_start_and_engram_hf_roundtrip(tmp_path, hf_source):
    vanilla = DSparkDraftModel(config("vanilla")).bfloat16()
    c = config()
    draft = DSparkDraftModel(c).bfloat16()
    src = tmp_path / "vanilla"
    src.mkdir()
    if hf_source:
        vanilla.save_pretrained(src)
    else:
        torch.save(
            {"draft_state_dict": vanilla.state_dict()}, src / "training_state.pt"
        )
    warm_start_draft_model(draft, str(src), draft_config=c, strategy="dspark")
    tokens = torch.tensor([[1, 5, 7]])
    hidden = torch.randn(1, 3, 16).bfloat16()
    base = torch.randn(1, 3, 23).bfloat16()
    buckets = draft.markov_head.memory.bucket_ids(tokens)
    kwargs = dict(prev_token_ids=tokens, hidden_states=hidden)
    torch.testing.assert_close(
        draft.apply_logits_head(base, **kwargs, engram_bucket_ids=buckets),
        vanilla.apply_logits_head(base, **kwargs),
        atol=0,
        rtol=0,
    )
    with torch.no_grad():
        draft.markov_head.memory.value.weight.normal_()
    expected = draft.apply_logits_head(base, **kwargs, engram_bucket_ids=buckets)
    wrapped = training(draft)
    contract = resume_contract(None, draft, wrapped)
    weights = DSparkTrainStrategy(wrapped).checkpoint_state_filter(wrapped.state_dict())
    state = {"draft_state_dict": weights, **contract}
    ckpt = tmp_path / "training_state.pt"
    torch.save(state, ckpt)
    cfg = tmp_path / "draft.json"
    c.to_json_file(cfg)
    dest = tmp_path / "hf"
    assert (
        specforge_cli(
            [
                "export",
                "--to",
                "hf",
                "--checkpoint",
                str(ckpt),
                "--draft-config",
                str(cfg),
                "--output-dir",
                str(dest),
            ]
        )
        == 0
    )
    restored = DSparkDraftModel.from_pretrained(dest, dtype=torch.bfloat16)
    restored_ids = restored.markov_head.memory.bucket_ids(tokens)
    torch.testing.assert_close(restored_ids, buckets)
    torch.testing.assert_close(
        restored.apply_logits_head(base, **kwargs, engram_bucket_ids=restored_ids),
        expected,
        atol=0,
        rtol=0,
    )
    c.dflash_config["engram"]["hash_multipliers"][0] += 2
    c.to_json_file(cfg)
    # Reload from disk: the checkpoint contract must not follow the config mutation.
    with pytest.raises(ValueError, match="hash"):
        materialize_draft(torch.load(ckpt), str(cfg))


def test_engram_checkpoint_requires_all_memory_weights_and_same_hash(tmp_path):
    cfg = config()
    draft = DSparkDraftModel(cfg)
    record = {
        "draft_state_dict": dict(draft.state_dict()),
        "dspark_engram": dict(cfg.dflash_config["engram"]),
    }
    path = tmp_path / "training_state.pt"
    cfg_path = tmp_path / "draft.json"
    cfg.to_json_file(cfg_path)
    del record["draft_state_dict"]["markov_head.memory.tables.3.weight"]
    torch.save(record, path)
    with pytest.raises(ValueError, match="tables.3.weight"):
        warm_start_draft_model(draft, str(path), draft_config=cfg, strategy="dspark")
    with pytest.raises(ValueError, match="tables.3.weight"):
        materialize_draft(record, str(cfg_path))
    record["dspark_engram"]["hash_multipliers"] = [11, 13, 17]
    torch.save(record, path)
    with pytest.raises(ValueError, match="hash"):
        warm_start_draft_model(draft, str(path), draft_config=cfg, strategy="dspark")


def test_scratch_build_matches_seeded_vanilla_and_anchor_sampling(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from specforge.algorithms import model_providers

    monkeypatch.setattr(model_providers, "_device", lambda: torch.device("cpu"))
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            draft_checkpoint_path=None,
            torch_dtype="bfloat16",
            cache_dir=None,
            trust_remote_code=False,
        ),
        training=SimpleNamespace(attention_backend="sdpa", strategy="dspark"),
    )
    torch.manual_seed(42)
    vanilla = model_providers.build_registered_draft(cfg, config("vanilla"))
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    draft = model_providers.build_registered_draft(cfg, config())
    assert torch.equal(torch.get_rng_state(), rng)
    for name, param in vanilla.named_parameters():
        torch.testing.assert_close(draft.get_parameter(name), param, atol=0, rtol=0)
    tokens = torch.tensor([[1, 3, 5, 7]])
    h = torch.randn(1, 4, 16).bfloat16()
    logits = torch.randn(1, 4, 23).bfloat16()
    args = dict(prev_token_ids=tokens, hidden_states=h)
    baseline = vanilla.apply_logits_head(logits, **args)

    def check(model):
        memory = model.markov_head.memory
        assert memory.value.weight.count_nonzero() == 0
        for table in memory.tables:
            assert 0.01 < table.weight.float().std() < 0.03
        for norm in (memory.query_norm, memory.key_norm):
            assert (norm.weight == 1).all()
        actual = model.apply_logits_head(
            logits, engram_bucket_ids=memory.bucket_ids(tokens), **args
        )
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)

    check(draft)
    vanilla.save_pretrained(tmp_path)
    cfg.model.draft_checkpoint_path = str(tmp_path)
    check(model_providers.build_registered_draft(cfg, config()))
