"""CPU regressions for causal training, local KV decoding and baseline warm start."""

from types import MethodType
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, initialization as hf_init

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.common.prefix_reranker_loss import prefix_reranker_loss
from specforge.modeling.draft.dspark import DSparkDraftModel, VanillaMarkovHead
from specforge.modeling.draft.prefix_reranker import PrefixReranker
from specforge.training.model_loading import warm_start_draft_model


def config(enabled=True):
    cfg = Qwen3Config(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
    )
    cfg.architectures = ["DSparkDraftModel"]
    cfg.block_size, cfg.num_target_layers = 3, 3
    cfg._attn_implementation = "sdpa"
    cfg.dflash_config = dict(
        projector_type="dspark",
        target_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=4,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    if enabled:
        cfg.dflash_config["prefix_reranker"] = dict(width=8, num_heads=2, top_k=19)
    return cfg


def head(top_k=5):
    return PrefixReranker(8, 4, 7, width=8, num_heads=2, top_k=top_k)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parallel_encoding_matches_incremental_kv(dtype):
    torch.manual_seed(1)
    m = head().to(dtype)
    tokens = torch.randn(2, 3, 7, 4).to(dtype)
    parallel, _ = m.encode(tokens)
    cache, outputs = None, []
    for i in range(7):
        out, cache = m.encode(tokens[..., i : i + 1, :], cache)
        outputs.append(out)
        assert cache[0].shape == (6, 2, i + 1, 4)
    tol = 3e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(torch.cat(outputs, -2), parallel, atol=tol, rtol=tol)
    with pytest.raises(ValueError, match="reset cache"):
        m.encode(tokens[..., :1, :], cache)
    torch.testing.assert_close(m.encode(tokens)[0], parallel)


def test_causal_and_block_isolation():
    torch.manual_seed(2)
    m = head()
    tokens = torch.randn(2, 3, 7, 4)
    before = m.encode(tokens)[0]
    changed = tokens.clone()
    changed[..., 4:, :] += torch.randn_like(changed[..., 4:, :]) * 10
    torch.testing.assert_close(before[..., :4, :], m.encode(changed)[0][..., :4, :])
    changed = tokens.clone()
    changed[:, 1] *= 20
    torch.testing.assert_close(before[:, 0], m.encode(changed)[0][:, 0])


@pytest.mark.parametrize("top_k", [1, 5, 30])
def test_zero_residual_preserves_baseline_including_ties(top_k):
    m, markov = head(top_k), VanillaMarkovHead(vocab_size=19, markov_rank=4)
    logits = torch.randn(2, 3, 7, 19)
    logits[0].zero_()
    hidden = torch.randn(2, 3, 7, 8)
    ids = torch.randint(19, (2, 3, 7))
    scores, candidates = m(logits, hidden, ids, markov)
    selected = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
    assert torch.equal(selected, logits.argmax(-1))
    assert (candidates.sort(-1).values.diff(dim=-1) > 0).all()
    torch.testing.assert_close(scores, logits.gather(-1, candidates))


def test_sampler_matches_sequential_baseline_and_resets_cache():
    torch.manual_seed(3)
    m, markov = head(), VanillaMarkovHead(vocab_size=19, markov_rank=4)
    logits, hidden = torch.randn(2, 7, 19), torch.randn(2, 7, 8)
    anchor = torch.tensor([2, 3])
    expected, _ = markov.sample_block_tokens(
        logits, first_prev_token_ids=anchor, hidden_states=hidden
    )
    for _ in range(2):
        assert torch.equal(m.sample(logits, hidden, anchor, markov), expected)
    with pytest.raises(ValueError, match="greedy"):
        m.sample(logits, hidden, anchor, markov, temperature=1)


def test_sampling_uses_reranked_previous_token():
    m, markov = head(), VanillaMarkovHead(vocab_size=19, markov_rank=4)
    previous, cache_lengths = [], []
    original_bias, original_encode = markov.compute_step_bias, m.encode

    def bias(token_ids, hidden_states):
        previous.append(token_ids.clone())
        return original_bias(token_ids, hidden_states)

    def encode(tokens, cache=None):
        result = original_encode(tokens, cache)
        cache_lengths.append(result[1][0].shape[-2])
        return result

    def score(logits, hidden, prefix, table):
        # Force a choice different from baseline, then inspect the next Markov input.
        ids = torch.full((*logits.shape[:-1], 1), 11, dtype=torch.long)
        return torch.zeros_like(ids, dtype=torch.float), ids

    with (
        patch.object(markov, "compute_step_bias", side_effect=bias),
        patch.object(m, "encode", side_effect=encode),
        patch.object(m, "score", side_effect=score),
    ):
        result = m.sample(
            torch.randn(2, 3, 19), torch.randn(2, 3, 8), torch.tensor([1, 2]), markov
        )
    assert torch.equal(previous[0], torch.tensor([1, 2]))
    assert all((p == 11).all() for p in previous[1:])
    assert (result == 11).all()
    assert cache_lengths == [1, 2, 3]


def test_missing_candidates_and_padding_have_zero_gradient():
    scores = torch.randn(1, 1, 3, 2, requires_grad=True)
    candidates = torch.tensor([[[[0, 1], [0, 1], [0, 1]]]])
    teacher = torch.tensor([[[1, 3, 1]]])
    valid = torch.tensor([[[True, True, False]]])
    loss, _, metrics = prefix_reranker_loss(
        scores,
        candidates,
        candidates[..., 0],
        teacher,
        teacher,
        valid,
        torch.ones_like(teacher, dtype=torch.float),
        training=False,
    )
    loss.backward()
    assert scores.grad[..., :1, :].abs().sum() > 0
    assert scores.grad[..., 1:, :].abs().sum() == 0
    assert metrics["eval_metric_sums"]["reranker/candidate_recall"] == 1
    assert metrics["eval_metric_denoms"]["reranker/candidate_recall"] == 2
    assert "tau_probabilistic" not in metrics["eval_metric_sums"]
    scores = scores.detach().requires_grad_()
    loss, _, _ = prefix_reranker_loss(
        scores,
        candidates,
        candidates[..., 0],
        teacher + 10,
        teacher,
        valid,
        torch.ones_like(teacher, dtype=torch.float),
        training=True,
    )
    loss.backward()
    assert (
        loss == 0 and torch.isfinite(scores.grad).all() and scores.grad.abs().sum() == 0
    )


def test_baseline_warm_start_initializes_only_wholly_missing_head(tmp_path):
    torch.manual_seed(4)
    baseline = DSparkDraftModel(config(False))
    checkpoint = tmp_path / "training_state.pt"
    torch.save(
        dict(strategy="dspark", draft_state_dict=baseline.state_dict()), checkpoint
    )
    with hf_init.no_init_weights():
        model = DSparkDraftModel(config())
    report = warm_start_draft_model(
        model, str(checkpoint), draft_config=model.config, strategy="dspark"
    )
    assert report.missing_keys and all(
        k.startswith("prefix_reranker.") for k in report.missing_keys
    )
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
    for key, value in model.named_parameters():
        assert value.requires_grad
        assert torch.isfinite(value).all()
    assert model.prefix_reranker.residual_out.weight.count_nonzero() == 0
    assert model.prefix_reranker.qkv.weight.abs().sum() > 0
    state = model.state_dict()
    state.pop("prefix_reranker.qkv.weight")
    torch.save(dict(strategy="dspark", draft_state_dict=state), checkpoint)
    with pytest.raises(ValueError, match="missing draft weights"):
        warm_start_draft_model(
            model, str(checkpoint), draft_config=model.config, strategy="dspark"
        )


def test_full_head_checkpoint_is_not_reinitialized(tmp_path):
    model = DSparkDraftModel(config())
    with torch.no_grad():
        model.prefix_reranker.residual_out.weight.fill_(0.123)
    checkpoint = tmp_path / "training_state.pt"
    torch.save(dict(strategy="dspark", draft_state_dict=model.state_dict()), checkpoint)
    copy = DSparkDraftModel(config())
    report = warm_start_draft_model(
        copy, str(checkpoint), draft_config=copy.config, strategy="dspark"
    )
    assert not report.missing_keys
    for key, value in model.state_dict().items():
        torch.testing.assert_close(copy.state_dict()[key], value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reranker_only_warm_start_preserves_baseline_after_optimizer_steps(tmp_path, dtype):
    from specforge.algorithms.dspark.providers import build_draft, resume_contract
    from specforge.config import load_config
    from specforge.optimizer import BF16Optimizer

    torch.manual_seed(53)
    baseline = DSparkDraftModel(config(False)).to(dtype=dtype)
    checkpoint = tmp_path / "training_state.pt"
    torch.save(
        dict(strategy="dspark", draft_state_dict=baseline.state_dict(), global_step=2604),
        checkpoint,
    )
    with hf_init.no_init_weights():
        draft = DSparkDraftModel(config()).to(dtype=dtype)
    report = warm_start_draft_model(
        draft, str(checkpoint), draft_config=draft.config, strategy="dspark"
    )
    assert report.missing_keys and all(
        key.startswith("prefix_reranker.") for key in report.missing_keys
    )
    cfg = load_config("examples/configs/qwen3.8-flash-next-dspark-prefix-reranker.yaml")
    cfg.model.draft_checkpoint_path = str(checkpoint)
    cfg.training.dspark_reranker_only = True
    # Exercise the provider's freeze policy after a real CPU weights-only load.
    with patch(
        "specforge.algorithms.model_providers.build_registered_draft", return_value=draft
    ):
        assert build_draft(cfg, draft.config) is draft
    for name, parameter in draft.named_parameters():
        assert parameter.requires_grad == name.startswith("prefix_reranker.")
        assert torch.isfinite(parameter).all()
    assert draft.prefix_reranker.residual_out.weight.count_nonzero() == 0
    assert draft.prefix_reranker.qkv.weight.abs().sum() > 0
    for key, value in baseline.state_dict().items():
        assert torch.equal(draft.state_dict()[key], value)

    model = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=nn.Linear(8, 19, bias=False).requires_grad_(False),
        target_embed_tokens=nn.Embedding(19, 8).requires_grad_(False),
        mask_token_id=0,
        block_size=3,
        attention_backend="sdpa",
        num_anchors=2,
        dspark_ce_loss_alpha=0.1,
        dspark_l1_loss_alpha=0.9,
        dspark_confidence_head_alpha=1.0,
    ).to(dtype=dtype)

    def anchors(self, seq_len, loss_mask, device):
        return torch.tensor([[0, 2], [0, 2]], device=device), torch.ones(
            2, 2, dtype=torch.bool, device=device
        )

    model._sample_anchor_positions = MethodType(anchors, model)
    inputs = dict(
        input_ids=torch.randint(1, 19, (2, 8)),
        hidden_states=torch.randn(2, 8, 16).to(dtype),
        loss_mask=torch.ones(2, 8),
        target_last_hidden_states=torch.randn(2, 8, 8).to(dtype),
    )
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = BF16Optimizer(draft, lr=0.01, total_steps=10, warmup_ratio=0)
    assert {id(p) for p in optimizer.model_params} == {
        id(p) for p in draft.prefix_reranker.parameters()
    }
    assert len(optimizer.fp32_params) == len(list(draft.prefix_reranker.parameters()))
    assert not optimizer.optimizer.state
    frozen_contract = resume_contract(cfg, draft, model)
    assert frozen_contract["dspark_reranker_only"] is True
    joint_cfg = cfg.model_copy(deep=True)
    joint_cfg.training.dspark_reranker_only = False
    joint_contract = resume_contract(joint_cfg, draft, model)
    assert "dspark_reranker_only" not in joint_contract
    assert frozen_contract["dspark_prefix_reranker_objective"] != joint_contract[
        "dspark_prefix_reranker_objective"
    ]

    proposal_logits = []
    hook = draft.prefix_reranker.register_forward_pre_hook(
        lambda _module, args: proposal_logits.append(args[0].detach().clone())
    )
    try:
        for _ in range(2):
            loss, _, _ = model(**inputs)
            assert loss.requires_grad and torch.isfinite(loss)
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith("draft_model.prefix_reranker."):
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                else:
                    assert parameter.grad is None
            assert draft.prefix_reranker.residual_out.weight.grad.abs().sum() > 0
            optimizer.step()
        with torch.no_grad():
            model(**inputs)
    finally:
        hook.remove()
    assert len(proposal_logits) == 3
    assert all(torch.equal(proposal_logits[0], value) for value in proposal_logits[1:])
    for key, value in model.state_dict().items():
        if not key.startswith("draft_model.prefix_reranker."):
            assert torch.equal(value, initial[key]), key
    assert draft.prefix_reranker.residual_out.weight.abs().sum() > 0
    assert any(
        not torch.equal(value, initial[key])
        for key, value in model.state_dict().items()
        if key.startswith("draft_model.prefix_reranker.") and "residual_out" not in key
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_training_forward_alignment_and_joint_updates(dtype):
    torch.manual_seed(5)
    draft = DSparkDraftModel(config())
    model = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=nn.Linear(8, 19, bias=False).requires_grad_(False),
        target_embed_tokens=nn.Embedding(19, 8).requires_grad_(False),
        mask_token_id=0,
        block_size=3,
        attention_backend="sdpa",
        num_anchors=2,
        dspark_ce_loss_alpha=0.1,
        dspark_l1_loss_alpha=0.9,
        dspark_confidence_head_alpha=1.0,
    )

    def anchors(self, seq_len, loss_mask, device):
        return torch.tensor([[0, 2], [0, 2]], device=device), torch.ones(
            2, 2, dtype=torch.bool, device=device
        )

    model._sample_anchor_positions = MethodType(anchors, model)
    inputs = dict(
        input_ids=torch.randint(1, 19, (2, 8)),
        hidden_states=torch.randn(2, 8, 16),
        loss_mask=torch.ones(2, 8),
        target_last_hidden_states=torch.randn(2, 8, 8),
    )
    model.to(dtype=dtype)
    inputs["hidden_states"] = inputs["hidden_states"].to(dtype)
    inputs["target_last_hidden_states"] = inputs["target_last_hidden_states"].to(dtype)
    initial = {k: p.detach().clone() for k, p in draft.named_parameters()}
    assert all(p.requires_grad for p in draft.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=0.01
    )
    for _ in range(2):
        with patch(
            "specforge.algorithms.common.prefix_reranker_loss.prefix_reranker_loss",
            wraps=prefix_reranker_loss,
        ) as call, patch.object(
            model, "_compute_dspark_loss", wraps=model._compute_dspark_loss
        ) as base_call:
            loss, accuracy, metrics = model(**inputs)
        teacher_logits = model.lm_head(inputs["target_last_hidden_states"])
        expected = teacher_logits.argmax(-1)[:, torch.tensor([[0, 1, 2], [2, 3, 4]])]
        torch.testing.assert_close(call.call_args.args[3], expected)
        assert torch.isfinite(loss) and 0 <= accuracy <= 1
        loss.backward()
        for module in (draft.fc, draft.markov_head, draft.confidence_head):
            assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
        assert draft.prefix_reranker.residual_out.weight.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad()
    for prefix in ("layers.", "fc.", "markov_head.", "confidence_head.", "prefix_reranker."):
        assert any(not torch.equal(p, initial[k]) for k, p in draft.named_parameters()
                   if k.startswith(prefix))
    assert set(metrics) == {"log_window"}
    assert metrics["log_window"]["weights"] == {
        "reranker/rank_ce": 1.0, "ce_loss": 0.1, "l1_loss": 0.9, "confidence_loss": 1.0,
    }
    sums, denoms = metrics["log_window"]["sums"], metrics["log_window"]["denoms"]
    assert "tau_probabilistic" in sums
    # Recompute the pre-rerank statistics from the captured loss inputs, without
    # exposing them as extra logging curves.
    _, base_metrics = model._compute_dspark_loss(**base_call.call_args.kwargs)
    for name in ("tau_probabilistic", "accept_rate@0", "accept_rate@1", "accept_rate@2"):
        gain_name = (
            "rerank_tau_gain" if name == "tau_probabilistic"
            else name.replace("accept_rate@", "rerank_accept_rate_gain@")
        )
        torch.testing.assert_close(
            sums[gain_name], sums[name] - base_metrics["sums"][name]
        )
        torch.testing.assert_close(denoms[gain_name], base_metrics["denoms"][name])
        assert not sums[gain_name].requires_grad
    assert not any(name.startswith(("baseline/", "pre_rerank/")) for name in sums)
    assert sums.keys() == denoms.keys()
    model.eval()
    with torch.no_grad():
        eval_loss, _, eval_metrics = model(**inputs)
        assert torch.isfinite(eval_loss)
        assert "log_window" not in eval_metrics
        assert eval_metrics["eval_objective_weights"] == metrics["log_window"]["weights"]


def test_export_round_trip_and_head_baseline_init(tmp_path):
    from specforge.export.to_hf import export_to_hf

    cfg = config()
    model = DSparkDraftModel(cfg)
    with torch.no_grad():
        model.prefix_reranker.residual_out.weight.normal_(std=0.1)
    cfg_path = tmp_path / "draft.json"
    cfg.to_json_file(cfg_path)
    checkpoint = tmp_path / "training_state.pt"
    torch.save(dict(strategy="dspark", draft_state_dict=model.state_dict()), checkpoint)
    out = tmp_path / "export"
    export_to_hf(str(checkpoint), str(cfg_path), str(out))
    copy = DSparkDraftModel.from_pretrained(out)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            copy.state_dict()[key].float(), value.bfloat16().float()
        )
    assert (
        copy.config.dflash_config["prefix_reranker"]
        == cfg.dflash_config["prefix_reranker"]
    )
    state = model.state_dict()
    state.pop("prefix_reranker.position_embed.weight")
    torch.save(dict(strategy="dspark", draft_state_dict=state), checkpoint)
    with pytest.raises(ValueError, match="prefix_reranker.position_embed"):
        export_to_hf(str(checkpoint), str(cfg_path), str(tmp_path / "invalid"))


def test_strategy_and_evaluator_keep_rank_loss_and_metric_names(tmp_path):
    from specforge.eval import Evaluator
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.metric_window import MetricWindow
    from specforge.training.strategies.base import DSparkTrainStrategy

    class Model(nn.Module):
        dspark_loss_mode = "original"
        dspark_ce_loss_alpha = dspark_l1_loss_alpha = 0.0
        dspark_confidence_head_alpha = 0.0

        def __init__(self):
            super().__init__()
            self.scores = nn.Parameter(torch.tensor([[[[1.0, 2.0], [3.0, 2.0]]]]))

        def forward(self, **kwargs):
            ids = torch.tensor([[[[0, 1], [0, 1]]]])
            teacher = torch.tensor([[[1, 1]]])
            mask = torch.ones_like(teacher, dtype=torch.bool)
            return prefix_reranker_loss(
                self.scores,
                ids,
                ids[..., 0],
                teacher,
                teacher,
                mask,
                mask.float(),
                training=self.training,
                candidate_target_probs=torch.full_like(self.scores, 0.25),
            )

    model = Model()
    strategy = DSparkTrainStrategy(model)
    batch = TrainBatch(
        sample_ids=["x"],
        strategy="dspark",
        metadata={},
        tensors={name: torch.zeros(1) for name in strategy.required_features},
    )
    window = MetricWindow()
    step = strategy.forward_loss(batch)
    window.update(step.metrics["log_window"])
    summary = window.summary()
    assert summary["loss_weighted"] == pytest.approx(step.loss.item())
    assert summary["reranker/candidate_recall"] == 1
    model.eval()
    result = Evaluator().run(strategy.forward_loss, [batch])
    assert result["eval/avg_loss"] == pytest.approx(step.loss.item())
    assert "eval/reranker/reference_prefix_length" in result
    assert result["eval/tau_probabilistic"] == pytest.approx(1.75)
    assert summary["tau_probabilistic"] == pytest.approx(1.75)
    assert summary["accept_rate@1"] == pytest.approx(0.5)
    assert result["eval/accept_rate@1"] == pytest.approx(0.5)
    assert summary["rerank_target_agreement_gain"] == pytest.approx(0.5)
    assert result["eval/rerank_teacher_forced_prefix_gain"] == pytest.approx(1.0)

    # Verify actual event tags through the ordinary training tracker, not just
    # intermediate metric dictionaries or the separate evaluation writer.
    from types import SimpleNamespace
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from specforge.tracker import TensorboardTracker
    from specforge.training.tracking import TrackerLogger

    tracker = TensorboardTracker(SimpleNamespace(), str(tmp_path))
    event_dir = tracker.writer.log_dir
    with TrackerLogger(tracker) as logger:
        logger(summary, 10)
        logger(result, 10)
    events = EventAccumulator(event_dir).Reload()
    for tag, expected in {
        "train/accept_rate@1": 0.5,
        "train/tau_probabilistic": 1.75,
        "eval/accept_rate@1": 0.5,
        "eval/tau_probabilistic": 1.75,
        "train/rerank_target_agreement_gain": 0.5,
        "eval/rerank_teacher_forced_prefix_gain": 1.0,
    }.items():
        assert tag in events.Tags()["scalars"]
        points = events.Scalars(tag)
        assert points[0].step == 10
        assert points[0].value == pytest.approx(expected)
    assert not any(
        "/baseline/" in tag or "/pre_rerank/" in tag
        for tag in events.Tags()["scalars"]
    )


def test_sparse_tau_matches_full_vocabulary_overlap_without_target_renormalization():
    from specforge.algorithms.common.dspark_metrics import acceptance_stats

    torch.manual_seed(71)
    scores = torch.randn(2, 2, 3, 2, requires_grad=True)
    candidates = torch.tensor([0, 3]).expand(2, 2, 3, 2)
    target = torch.randn(2, 2, 3, 5).softmax(-1)
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 0, -1] = False
    valid[1, 1] = False
    loss, _, metrics = prefix_reranker_loss(
        scores, candidates, candidates[..., 0], target.argmax(-1), target.argmax(-1),
        valid, valid.float(), training=False,
        candidate_target_probs=target.gather(-1, candidates),
    )
    full_q = torch.zeros_like(target).scatter(-1, candidates, scores.softmax(-1))
    overlap = 1 - 0.5 * (full_q - target).abs().sum(-1)
    sums, denoms = acceptance_stats(overlap, valid)
    for name in sums:
        torch.testing.assert_close(metrics["eval_metric_sums"][name], sums[name])
        torch.testing.assert_close(metrics["eval_metric_denoms"][name], denoms[name])
        assert not metrics["eval_metric_sums"][name].requires_grad
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_example_config_defaults_to_scratch():
    from specforge.config import load_config
    from specforge.algorithms.dspark.providers import build_draft

    cfg = load_config(
        "examples/configs/qwen3.8-flash-next-dspark-prefix-reranker.yaml"
    )
    assert cfg.training.total_steps == 26040 and cfg.training.max_steps == 2604
    assert cfg.training.batch_size * cfg.training.accumulation_steps * 4 == 512
    assert cfg.model.draft_checkpoint_path is None
    assert cfg.training.resume_from is None
    assert cfg.training.dspark_reranker_only is False
    with patch("specforge.algorithms.model_providers.build_registered_draft", return_value="scratch") as build:
        assert build_draft(cfg, config()) == "scratch"
        build.assert_called_once()
    cfg.training.dspark_ce_loss_alpha = cfg.training.dspark_l1_loss_alpha = 0
    with pytest.raises(ValueError, match="baseline CE or L1"):
        build_draft(cfg, config())


def test_frozen_example_warm_starts_baseline_with_a_fresh_run():
    from specforge.config import load_config

    cfg = load_config(
        "examples/configs/qwen3.8-flash-next-dspark-prefix-reranker-frozen.yaml"
    )
    joint = load_config("examples/configs/qwen3.8-flash-next-dspark-prefix-reranker.yaml")
    assert cfg.training.dspark_reranker_only is True
    assert "step2604" in cfg.model.draft_checkpoint_path
    assert cfg.training.resume_from is None
    assert cfg.model.draft_model_config == joint.model.draft_model_config
    assert cfg.training.total_steps == 26040 and cfg.training.max_steps == 2604
    assert cfg.training.batch_size * cfg.training.accumulation_steps * 4 == 512
    assert cfg.run_id != joint.run_id and cfg.output_dir != joint.output_dir


@pytest.mark.parametrize("enabled,checkpoint", [(False, "/baseline/step2604"), (True, None)])
def test_reranker_only_rejects_missing_head_or_initial_weights(enabled, checkpoint):
    from specforge.algorithms.dspark.providers import build_draft
    from specforge.config import load_config

    cfg = load_config("examples/configs/qwen3.8-flash-next-dspark-prefix-reranker.yaml")
    cfg.training.dspark_reranker_only = True
    cfg.model.draft_checkpoint_path = checkpoint
    with patch("specforge.algorithms.model_providers.build_registered_draft") as build:
        with pytest.raises(ValueError, match="reranker"):
            build_draft(cfg, config(enabled))
        build.assert_not_called()


def test_reranker_only_can_resume_without_baseline_warm_start():
    from specforge.algorithms.dspark.providers import build_draft
    from specforge.config import load_config

    cfg = load_config("examples/configs/qwen3.8-flash-next-dspark-prefix-reranker.yaml")
    cfg.training.dspark_reranker_only = True
    cfg.training.resume_from = "/reranker/checkpoint"
    assert cfg.model.draft_checkpoint_path is None
    draft = DSparkDraftModel(config())
    with patch(
        "specforge.algorithms.model_providers.build_registered_draft", return_value=draft
    ):
        assert build_draft(cfg, draft.config) is draft
    assert all(
        parameter.requires_grad == name.startswith("prefix_reranker.")
        for name, parameter in draft.named_parameters()
    )


def test_distributed_rank_loss_uses_global_hit_denominator():
    scores = torch.randn(1, 1, 2, 2, requires_grad=True)
    ids = torch.tensor([[[[0, 1], [0, 1]]]])
    teacher = torch.tensor([[[1, 3]]])
    mask = torch.ones_like(teacher, dtype=torch.bool)
    args = (scores, ids, ids[..., 0], teacher, teacher, mask, mask.float())
    local_loss = prefix_reranker_loss(*args, training=False)[0]
    # This rank has one hit; other rank has three. DDP will average gradients.
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_reduce", side_effect=lambda t: t.add_(3)),
    ):
        distributed_loss = prefix_reranker_loss(*args, training=True)[0]
    torch.testing.assert_close(distributed_loss, local_loss * 2 / 4)


def _ddp_reranker_worker(rank, rendezvous, reranker_only):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(31)
        draft = DSparkDraftModel(config())
        if reranker_only:
            from specforge.algorithms.dspark.providers import build_draft
            from specforge.config import load_config

            cfg = load_config("examples/configs/qwen3.8-flash-next-dspark-prefix-reranker.yaml")
            cfg.training.dspark_reranker_only = True
            cfg.model.draft_checkpoint_path = "/baseline/step2604"
            with patch(
                "specforge.algorithms.model_providers.build_registered_draft", return_value=draft
            ):
                draft = build_draft(cfg, draft.config)
        model = OnlineDSparkModel(
            draft_model=draft,
            target_lm_head=nn.Linear(8, 19, bias=False).requires_grad_(False),
            target_embed_tokens=nn.Embedding(19, 8).requires_grad_(False),
            mask_token_id=0,
            block_size=3,
            attention_backend="sdpa",
            num_anchors=2,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=1.0,
        )
        wrapped = DistributedDataParallel(model)
        frozen_initial = {
            name: parameter.detach().clone()
            for name, parameter in draft.named_parameters()
            if not parameter.requires_grad
        }
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        for step in range(2):
            torch.manual_seed(41 + rank + step)
            # Rank 0 has no supervised positions, but must still participate in
            # loss normalization and every DDP gradient reduction on both steps.
            loss, _, _ = wrapped(
                input_ids=torch.randint(1, 19, (1, 8)),
                hidden_states=torch.randn(1, 8, 16),
                loss_mask=torch.full((1, 8), float(rank)),
                target_last_hidden_states=torch.randn(1, 8, 8),
            )
            loss.backward()
            assert all((p.grad is not None) == p.requires_grad for p in draft.parameters())
            assert all(p.grad is not None for p in draft.prefix_reranker.parameters())
            optimizer.step()
            optimizer.zero_grad()
        for name, parameter in draft.named_parameters():
            if name in frozen_initial:
                assert torch.equal(parameter, frozen_initial[name]), name
        weight = draft.prefix_reranker.residual_out.weight.detach()
        copies = [torch.empty_like(weight) for _ in range(2)]
        dist.all_gather(copies, weight)
        torch.testing.assert_close(copies[0], copies[1])
        assert weight.abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("reranker_only", [False, True], ids=["joint", "reranker_only"])
def test_two_rank_training_with_empty_rank(tmp_path, reranker_only):
    torch.multiprocessing.spawn(
        _ddp_reranker_worker,
        args=(f"file://{tmp_path}/rendezvous", reranker_only),
        nprocs=2,
        join=True,
    )
