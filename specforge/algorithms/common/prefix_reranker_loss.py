"""Sparse greedy ranking objective and explicitly teacher-forced diagnostics."""

import torch
import torch.nn.functional as F

from .dspark_metrics import acceptance_stats


def prefix_reranker_loss(
    scores,
    candidates,
    baseline_ids,
    teacher_ids,
    reference_ids,
    valid,
    weights,
    *,
    training,
    candidate_target_probs=None,
):
    matches = candidates == teacher_ids.unsqueeze(-1)
    hit = matches.any(-1)
    label = matches.long().argmax(-1)
    ce = F.cross_entropy(
        scores.float().reshape(-1, scores.shape[-1]),
        label.reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    rank_weights = weights * valid * hit
    numerator = (ce * rank_weights).sum()
    denominator = rank_weights.sum().detach()
    global_denominator = denominator.clone()
    world_size = 1
    if (
        training
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(global_denominator)
        world_size = torch.distributed.get_world_size()
    loss = numerator * world_size / global_denominator.clamp_min(1e-6)

    with torch.no_grad():
        predicted = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        base_correct = baseline_ids == teacher_ids
        correct = predicted == teacher_ids
        reference_correct = (predicted == reference_ids) & valid
        count = valid.float().sum()
        sums = {"reranker/rank_ce": numerator.detach()}
        denoms = {"reranker/rank_ce": denominator}
        if candidate_target_probs is not None:
            # q is normalized over the selected candidates, zero elsewhere.
            # p remains the ORIGINAL full-vocabulary target probability: never
            # renormalize p over candidates or missing target mass disappears.
            proposal_probs = scores.float().softmax(-1)
            overlap = torch.minimum(
                proposal_probs, candidate_target_probs.float()
            ).sum(-1).clamp(0, 1)
            accept_sums, accept_denoms = acceptance_stats(overlap, valid)
            sums.update(accept_sums)
            denoms.update(accept_denoms)

        def metric(name, values, mask):
            sums[name] = (values.float() * mask).sum()
            denoms[name] = mask.float().sum()

        metric("reranker/candidate_recall", hit, valid)
        metric("reranker/target_accuracy", correct, valid)
        metric("rerank_target_agreement_gain", correct.float() - base_correct.float(), valid)
        metric("reranker/fix_rate", correct, valid & ~base_correct)
        metric("reranker/harm_rate", ~correct, valid & base_correct)
        for i in range(valid.shape[-1]):
            metric(f"reranker/target_accuracy@{i}", correct[..., i], valid[..., i])
            metric(f"reranker/candidate_recall@{i}", hit[..., i], valid[..., i])
        # Reference-prefix length is NOT stochastic distribution overlap or a
        # multi-round target-verified MAL. Keep separate TensorBoard names.
        prefix = reference_correct.float().cumprod(-1)
        base_prefix = ((baseline_ids == reference_ids) & valid).float().cumprod(-1)
        metric("reranker/reference_prefix_length", 1 + prefix.sum(-1), valid.any(-1))
        metric(
            "rerank_teacher_forced_prefix_gain",
            prefix.sum(-1) - base_prefix.sum(-1),
            valid.any(-1),
        )
        accuracy = reference_correct.sum().float() / count.clamp_min(1)
        if training:
            sums["acc"] = reference_correct.sum().float()
            denoms["acc"] = count
            metrics = {"log_window": {
                "sums": sums,
                "denoms": denoms,
                "weights": {"reranker/rank_ce": 1.0},
            }}
        else:
            metrics = {
                "accuracy_denom": count,
                "acc_corrects": [
                    reference_correct[..., i].sum() for i in range(valid.shape[-1])
                ],
                "acc_denoms": [valid[..., i].sum() for i in range(valid.shape[-1])],
                "metric_loss_denoms": [denominator],
                "eval_metric_sums": sums,
                "eval_metric_denoms": denoms,
                "eval_objective_weights": {"reranker/rank_ce": 1.0},
            }
    return loss, accuracy, metrics
