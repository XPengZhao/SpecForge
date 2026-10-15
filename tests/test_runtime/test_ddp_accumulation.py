"""Real CPU DDP: count reductions and compare accumulated updates to a global batch."""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput


class _Optimizer:
    def __init__(self, model):
        self.parameters = list(model.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=0.01)
        self.gradients = []
        self.norms = []

    def step(self):
        self.gradients.append(
            torch.cat([p.grad.flatten().clone() for p in self.parameters])
        )
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 0.3)
        self.norms.append(norm.clone())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm


class _Strategy:
    def __init__(self, model):
        self.model = model
        self.sync_flags = []

    def forward_loss(self, batch, ctx=None):
        self.sync_flags.append(self.model.require_backward_grad_sync)
        x, y = batch
        return StepOutput(torch.nn.functional.mse_loss(self.model(x), y), {})


def _count_reduce(state, bucket):
    state["calls"] += 1
    buffer = bucket.buffer()
    dist.all_reduce(buffer)
    buffer.div_(dist.get_world_size())
    future = torch.futures.Future()
    future.set_result(buffer)
    return future


def _batch(rank, window, micro):
    generator = torch.Generator().manual_seed(10000 * window + 1000 * rank + micro)
    return (
        torch.randn(2, 3, generator=generator, dtype=torch.float64),
        torch.randn(2, 2, generator=generator, dtype=torch.float64),
    )


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=60),
    )
    try:
        for accumulation in (1, 4, 128):
            torch.manual_seed(42)
            initial = torch.nn.Linear(3, 2).double().state_dict()

            def backend():
                model = torch.nn.Linear(3, 2).double()
                model.load_state_dict(initial)
                result = FSDPTrainingBackend(
                    ParallelConfig(world_size=2, sharding_strategy="NO_SHARD"),
                    optimizer_factory=_Optimizer,
                )
                result.prepare_model(model, wrap=True)
                state = {"calls": 0}
                result.module.register_comm_hook(state, _count_reduce)
                return result, state

            accumulated, count_a = backend()
            every_micro, count_b = backend()
            strategy = _Strategy(accumulated.module)
            core = TrainerCore(strategy, accumulated, accumulation_steps=accumulation)
            reference = torch.nn.Linear(3, 2).double()
            reference.load_state_dict(initial)
            reference_optimizer = _Optimizer(reference)

            for window in range(2):
                for micro in range(accumulation):
                    batch = _batch(rank, window, micro)
                    result = core.train_step(batch)
                    boundary = micro == accumulation - 1
                    assert result.optimizer_stepped == boundary
                    assert count_a["calls"] == window + int(boundary)
                    assert strategy.sync_flags[-1] == boundary
                    if not boundary:
                        assert len(accumulated.optimizer.gradients) == window

                    x, y = batch
                    loss = torch.nn.functional.mse_loss(every_micro.module(x), y)
                    every_micro.backward(loss / accumulation)
                every_micro.step()
                assert count_b["calls"] == (window + 1) * accumulation

                batches = [
                    _batch(r, window, m) for r in range(2) for m in range(accumulation)
                ]
                x = torch.cat([b[0] for b in batches])
                y = torch.cat([b[1] for b in batches])
                torch.nn.functional.mse_loss(reference(x), y).backward()
                reference_optimizer.step()
                for other in (every_micro.optimizer, reference_optimizer):
                    torch.testing.assert_close(
                        accumulated.optimizer.gradients[-1],
                        other.gradients[-1],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    torch.testing.assert_close(
                        accumulated.optimizer.norms[-1],
                        other.norms[-1],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    for a, b in zip(accumulated.optimizer.parameters, other.parameters):
                        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
                assert core.accumulation_remainder == 0

        # Context exit must restore DDP synchronization even on failure.
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "injected"):
            with accumulated.accumulation_context(is_boundary=False):
                assert not accumulated.module.require_backward_grad_sync
                raise RuntimeError("injected")
        assert accumulated.module.require_backward_grad_sync
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class TestDDPAccumulation(unittest.TestCase):
    def test_reduction_count_gradients_clipping_and_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=True)
