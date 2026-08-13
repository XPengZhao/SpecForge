# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Batch-size-invariant eval aggregation: per-position correct/denom counts are
summed over the whole eval set (all batches, all DP ranks) BEFORE any ratio or
geometric sum. The evaluator's own collective schedule is decided globally, so
empty or scalar-only shards issue the same reductions as their peers; when
``forward_fn`` is itself collective (FSDP), every rank must additionally
iterate the same number of eval batches."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepOutput


class Evaluator:
    """Aggregate a full eval pass into ``eval/*`` metrics.

    ``eval/avg_acc``: for per-position (TTT) strategies, position-0 accuracy
    from eval-set-wide correct/denom counts; for scalar strategies
    (DFlash/Domino), the ``accuracy_denom``-weighted mean of batch accuracy.
    """

    def run(
        self,
        forward_fn: Callable[[TrainBatch], StepOutput],
        batches: Optional[Iterable[TrainBatch]],
    ) -> Dict[str, Any]:
        """Run the pass; returns ``{}`` if zero batches were processed globally.

        Scalar accuracy is weighted by ``metrics['accuracy_denom']`` when present,
        else by the loss-token count — only approximately batch-size invariant
        when the accuracy counts a different token set than the loss. In a mixed
        pass, scalar batches feed avg_loss only; their accuracy is not merged.
        """
        # pp rows: [correct, denom, acceptance_rate*w, ploss*w] per TTT
        # position, float64 so counts stay exact past 2**24.
        pp = None
        scalar_metric_sums: Dict[str, torch.Tensor] = {}
        scalar_metric_denoms: Dict[str, torch.Tensor] = {}
        objective_weights: Dict[str, float] | None = None
        # [loss*w, w, scalar_acc*denom, scalar_denom, n_batches, ar_w, pl_w]
        sums = None

        with torch.no_grad():
            for batch in batches if batches is not None else ():
                out = forward_fn(batch)
                m = out.metrics
                # .mean() normalizes a shape-[1] loss to the 0-dim slot.
                loss = (
                    out.loss.detach().double().mean()
                    if isinstance(out.loss, torch.Tensor)
                    else torch.tensor(float(out.loss), dtype=torch.float64)
                )
                if sums is None:
                    sums = torch.zeros(7, dtype=torch.float64, device=loss.device)
                tokens = self._token_count(batch, m, device=sums.device)
                sums[0] += loss.to(sums.device) * tokens
                sums[1] += tokens
                sums[4] += 1.0

                batch_metric_sums = m.get("eval_metric_sums", {})
                batch_metric_denoms = m.get("eval_metric_denoms", {})
                if batch_metric_sums.keys() != batch_metric_denoms.keys():
                    raise ValueError(
                        "eval metric sums and denominators must have identical keys"
                    )
                for name, value in batch_metric_sums.items():
                    metric_sum = torch.as_tensor(value).detach().double().sum()
                    metric_denom = (
                        torch.as_tensor(batch_metric_denoms[name]).detach().double().sum()
                    )
                    if name not in scalar_metric_sums:
                        scalar_metric_sums[name] = torch.zeros_like(metric_sum)
                        scalar_metric_denoms[name] = torch.zeros_like(metric_denom)
                    scalar_metric_sums[name] += metric_sum
                    scalar_metric_denoms[name] += metric_denom

                batch_objective_weights = {
                    name: float(value)
                    for name, value in m.get("eval_objective_weights", {}).items()
                }
                if batch_objective_weights:
                    if objective_weights is None:
                        objective_weights = batch_objective_weights
                    elif objective_weights != batch_objective_weights:
                        raise ValueError(
                            "eval objective weights must be constant across batches"
                        )

                if "acc_corrects" in m and "acc_denoms" in m:
                    correct = self._stack(m["acc_corrects"])
                    denom = self._stack(m["acc_denoms"])
                    if pp is None:
                        pp = torch.zeros(
                            4,
                            correct.numel(),
                            dtype=torch.float64,
                            device=correct.device,
                        )
                    pp[0] += correct
                    pp[1] += denom
                    w = tokens.to(pp.device)
                    if "acceptance_rates" in m:
                        pp[2] += self._stack(m["acceptance_rates"]) * w
                        sums[5] += tokens
                    if "plosses" in m:
                        pp[3] += self._stack(m["plosses"]) * w
                        sums[6] += tokens
                elif "accuracy" in m:
                    acc = m["accuracy"]
                    acc = (
                        acc.detach().double().mean().to(sums.device)
                        if isinstance(acc, torch.Tensor)
                        else torch.tensor(
                            float(acc), dtype=torch.float64, device=sums.device
                        )
                    )
                    denom = m.get("accuracy_denom")
                    w = (
                        torch.as_tensor(denom).detach().double().sum().to(sums.device)
                        if denom is not None
                        else tokens
                    )
                    sums[2] += acc * w
                    sums[3] += w

        # Fixed global schedule, identical on every rank: (1) SUM the scalar
        # sums, (2) MAX the per-position length — a global decision, so an
        # empty shard still participates — (3) SUM one padded count buffer iff
        # any rank has per-position data.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            device = self._comm_device()
            sums = (
                sums if sums is not None else torch.zeros(7, dtype=torch.float64)
            ).to(device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            local_len = pp.size(1) if pp is not None else 0
            pp_len = torch.tensor([local_len], dtype=torch.int64, device=device)
            dist.all_reduce(pp_len, op=dist.ReduceOp.MAX)
            global_len = int(pp_len.item())
            if global_len > 0:
                buf = torch.zeros(4, global_len, dtype=torch.float64, device=device)
                if pp is not None:
                    buf[:, : pp.size(1)] = pp.to(device)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                pp = buf

        if sums is None or sums[4].item() == 0.0:
            # Zero batches globally: report nothing — fabricated zero metrics
            # would poison best-checkpoint tracking.
            return {}

        metric_names = sorted(scalar_metric_sums)
        if world_size > 1:
            gathered_names: List[List[str]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered_names, metric_names)
            metric_names = sorted({name for names in gathered_names for name in names})
            device = self._comm_device()
            metric_buffer = torch.zeros(
                2,
                len(metric_names),
                dtype=torch.float64,
                device=device,
            )
            for index, name in enumerate(metric_names):
                if name in scalar_metric_sums:
                    metric_buffer[0, index] = scalar_metric_sums[name].to(device)
                    metric_buffer[1, index] = scalar_metric_denoms[name].to(device)
            if metric_names:
                dist.all_reduce(metric_buffer, op=dist.ReduceOp.SUM)

            gathered_weights: List[Dict[str, float] | None] = [
                None for _ in range(world_size)
            ]
            dist.all_gather_object(gathered_weights, objective_weights)
            nonempty_weights = [weights for weights in gathered_weights if weights]
            if nonempty_weights:
                objective_weights = nonempty_weights[0]
                if any(
                    weights != objective_weights
                    for weights in nonempty_weights[1:]
                ):
                    raise ValueError(
                        "eval objective weights must be constant across ranks"
                    )
        else:
            metric_buffer = torch.zeros(2, len(metric_names), dtype=torch.float64)
            for index, name in enumerate(metric_names):
                metric_buffer[0, index] = scalar_metric_sums[name].cpu()
                metric_buffer[1, index] = scalar_metric_denoms[name].cpu()

        loss_x_w, loss_w, acc_sum, acc_w, _n, ar_w, pl_w = sums.tolist()
        avg_loss = loss_x_w / max(loss_w, 1.0)

        if pp is not None:
            pp = pp.cpu()
            per_position_acc = (pp[0] / pp[1].clamp_min(1.0)).tolist()
            simulated_acc_len = self._simulated_acc_len(per_position_acc)
            metrics = {
                "eval/avg_loss": avg_loss,
                "eval/avg_acc": float(per_position_acc[0]),
                "eval/per_position_acc": per_position_acc,
                "eval/simulated_acc_len": simulated_acc_len,
                "eval/simulated_top1_accepted_tokens": simulated_acc_len,
                "eval/simulated_top1_mal": 1.0 + simulated_acc_len,
            }
            for index, value in enumerate(per_position_acc, start=1):
                metrics[f"eval/mtp_{index}_accuracy"] = float(value)
            if ar_w > 0:
                for i, v in enumerate((pp[2] / ar_w).tolist()):
                    metrics[f"eval/acceptance_rate_{i}"] = v
            if pl_w > 0:
                for i, v in enumerate((pp[3] / pl_w).tolist()):
                    metrics[f"eval/ploss_{i}"] = v
            self._add_scalar_metrics(metrics, metric_names, metric_buffer)
            self._set_exact_avg_loss(
                metrics, metric_names, metric_buffer, objective_weights
            )
            overlap = [
                metrics.get(f"eval/mtp_{index}_predicted_acceptance")
                for index in range(1, len(per_position_acc) + 1)
            ]
            if all(value is not None for value in overlap):
                accepted_tokens = self._simulated_acc_len(
                    [float(value) for value in overlap if value is not None]
                )
                metrics["eval/simulated_distribution_accepted_tokens"] = (
                    accepted_tokens
                )
                metrics["eval/simulated_distribution_mal"] = 1.0 + accepted_tokens
            return metrics

        avg_acc = acc_sum / acc_w if acc_w else 0.0
        metrics = {
            "eval/avg_loss": avg_loss,
            "eval/avg_acc": avg_acc,
            "eval/simulated_acc_len": avg_acc,
        }
        self._add_scalar_metrics(metrics, metric_names, metric_buffer)
        self._set_exact_avg_loss(
            metrics, metric_names, metric_buffer, objective_weights
        )
        return metrics

    @staticmethod
    def _set_exact_avg_loss(
        metrics: Dict[str, Any],
        names: List[str],
        values: torch.Tensor,
        weights: Dict[str, float] | None,
    ) -> None:
        if not weights:
            return
        indices = {name: index for index, name in enumerate(names)}
        objective = 0.0
        for name, weight in weights.items():
            if name not in indices:
                raise ValueError(f"missing eval objective component: {name}")
            index = indices[name]
            numerator = float(values[0, index].item())
            denominator = float(values[1, index].item())
            objective += weight * numerator / max(denominator, 1.0)
        metrics["eval/avg_loss"] = objective

    @staticmethod
    def _add_scalar_metrics(
        metrics: Dict[str, Any], names: List[str], values: torch.Tensor
    ) -> None:
        for index, name in enumerate(names):
            denom = float(values[1, index].item())
            if denom <= 0:
                continue
            value = float(values[0, index].item()) / denom
            metrics[f"eval/{name}"] = value
            if name == "l1_loss" or name.startswith("mtp_") and name.endswith("_l1"):
                prefix = name.removesuffix("_l1")
                if name == "l1_loss":
                    prefix = "overall"
                metrics[f"eval/{prefix}_tv"] = 0.5 * value
                metrics[f"eval/{prefix}_predicted_acceptance"] = 1.0 - 0.5 * value

    @staticmethod
    def _stack(values: Iterable[Any]) -> torch.Tensor:
        return torch.stack([torch.as_tensor(v).detach().double() for v in values])

    @staticmethod
    def _comm_device() -> torch.device:
        """Return the bound device required by the active collective backend."""
        backend = str(dist.get_backend()).lower()
        if "nccl" in backend:
            return torch.device("cuda", torch.cuda.current_device())
        if "hccl" in backend:
            from specforge.utils import get_local_device

            device = get_local_device()
            if device.type != "npu":
                raise RuntimeError(
                    "HCCL evaluation requires SPECFORGE_DEVICE=npu and a bound "
                    f"NPU device, got {device}"
                )
            return device
        return torch.device("cpu")

    @staticmethod
    def _simulated_acc_len(per_position_acc: List[float]) -> float:
        """E[accepted tokens] = a0 + a0*a1 + ... over the eval-set-wide
        per-position accuracy (length = ttt_length)."""
        cumulative, total = 1.0, 0.0
        for acc in per_position_acc:
            cumulative *= acc
            total += cumulative
        return total

    @staticmethod
    def _token_count(
        batch: TrainBatch, metrics: Dict[str, Any], device
    ) -> torch.Tensor:
        """This batch's token weight as a 0-dim float64 tensor (no host sync):
        the strategy's loss denoms, else the loss mask, else 1."""
        denoms = metrics.get("metric_loss_denoms")
        if denoms:
            return (
                torch.stack([torch.as_tensor(d).detach().float() for d in denoms])
                .sum()
                .double()
                .to(device)
            )
        loss_mask = batch.tensors.get("loss_mask")
        if isinstance(loss_mask, torch.Tensor):
            return loss_mask.sum().double().to(device)
        return torch.ones((), dtype=torch.float64, device=device)


__all__ = ["Evaluator"]
