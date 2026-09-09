"""Window weighting, first-step snapshots, and DSpark/controller integration."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.training.metric_window import MetricWindow
from specforge.training.controller import TrainerCore, TrainerController
from specforge.training.strategies.base import StepOutput, DSparkTrainStrategy
from tests.test_runtime.test_trainer import FakeStrategy, FakeBackend, _batch


def payload(num, den):
    return dict(sums={'ce_loss': torch.tensor(float(num), requires_grad=True)},
                denoms={'ce_loss': torch.tensor(float(den))}, weights={'ce_loss': 1.0})


class DDPLoggingModel(torch.nn.Module):
    dspark_ce_loss_alpha = .3
    dspark_l1_loss_alpha = .8
    dspark_kl_loss_alpha = 1.2
    dspark_confidence_head_alpha = .7
    dspark_opd_loss_alpha = .2

    def __init__(self, mode):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.dspark_loss_mode = mode

    def forward(self, **kwargs):
        sums = {name: torch.tensor(2.) for name in
                ('ce_loss', 'l1_loss', 'kl_loss', 'confidence_loss', 'opd_loss',
                 'mtp_1_ce', 'mtp_1_kl')}
        denoms = {name: torch.tensor(1.) for name in sums}
        return self.w * (dist.get_rank() + 1), torch.tensor(.5), {
            'eval_metric_sums': sums, 'eval_metric_denoms': denoms,
        }


def distributed_worker(rank, root):
    dist.init_process_group('gloo', init_method=f'file://{root}/init', rank=rank, world_size=2)
    try:
        window = MetricWindow()
        window.update(payload(2 if rank == 0 else 30, 1 if rank == 0 else 10))
        torch.save(window.summary(), f'{root}/{rank}.pt')
        for mode in ('original', 'kl'):
            model = DDPLoggingModel(mode)
            wrapped = torch.nn.parallel.DistributedDataParallel(model)
            batch = _batch()
            batch.tensors.update({k: torch.ones(1, 2) for k in
                                  DSparkTrainStrategy.required_features})
            strategy = DSparkTrainStrategy(wrapped)
            out = strategy.forward_loss(batch)
            weights = out.metrics['log_window']['weights']
            expected = ({'ce_loss': .3, 'l1_loss': .8} if mode == 'original'
                        else {'kl_loss': 1.2})
            assert weights == {**expected, 'confidence_loss': .7, 'opd_loss': .2}
            out.loss.backward()
            # A bypassed DDP forward would leave rank-local gradients 1 or 2.
            torch.testing.assert_close(model.w.grad, torch.tensor(1.5))
            window.update(out.metrics['log_window'])
            summary = window.summary()
            assert abs(summary['loss_weighted'] - 2 * sum(weights.values())) < 1e-5
            wrapped.eval()
            with torch.no_grad():
                assert 'log_window' not in strategy.forward_loss(batch).metrics
    finally:
        dist.destroy_process_group()


class WindowStrategy(FakeStrategy):
    def forward_loss(self, batch, ctx=None):
        out = super().forward_loss(batch, ctx)
        num = batch.tensors['x'].sum()
        return StepOutput(out.loss, {**out.metrics, 'log_window': payload(num, 1)})


class TestMetricWindow(unittest.TestCase):
    def test_ratio_not_average_of_ratios_and_snapshot_does_not_reset(self):
        window = MetricWindow()
        first = payload(2, 1)
        window.update(first)
        self.assertFalse(window.sums['ce_loss'].requires_grad)
        self.assertIsNone(window.sums['ce_loss'].grad_fn)
        self.assertAlmostEqual(window.summary(reset=False)['ce_loss'], 2)
        window.update(payload(30, 10))
        result = window.summary()
        self.assertAlmostEqual(result['ce_loss'], 32 / 11, places=6)
        self.assertAlmostEqual(result['loss'], 2.5, places=5)
        self.assertAlmostEqual(result['loss_weighted'], 32 / 11, places=6)
        self.assertEqual(result['log_micro_batches'], 2)
        self.assertEqual(first['sums']['ce_loss'].item(), 2)
        self.assertIsNone(window.summary())

    def test_empty_weights_and_zero_denominator(self):
        window = MetricWindow()
        data = payload(0, 0); data['weights'] = {}
        window.update(data)
        self.assertEqual(window.summary()['loss'], 0)

    def test_controller_all_micro_batches_and_final_partial_window(self):
        logs = []
        strategy = WindowStrategy()
        core = TrainerCore(strategy, FakeBackend(strategy.model), accumulation_steps=2)
        controller = TrainerController(core, run_id='window-test', max_steps=5,
                                       log_interval=3, logger=lambda m, s: logs.append((s, m)))
        batches = []
        for value in range(1, 11):
            batch = _batch(); batch.tensors['x'] = torch.tensor([float(value)])
            batches.append(batch)
        controller.fit(batches)
        self.assertEqual([s for s, _ in logs], [1, 3, 5])
        self.assertEqual([m['log_micro_batches'] for _, m in logs], [2, 6, 4])
        self.assertEqual([m['ce_loss'] for _, m in logs], [1.5, 3.5, 8.5])
        self.assertTrue(all(m['grad_norm'] == 1 for _, m in logs))
        self.assertEqual(core.backend.backwards, 10)
        self.assertEqual(core.backend.steps, 5)
        # Logging does not change the backward loss or gradient accumulation.
        self.assertEqual(strategy.model.w.grad.item(), 27.5)

    def test_strategy_exposes_raw_local_statistics(self):
        class Model(torch.nn.Module):
            dspark_loss_mode = 'original'
            dspark_ce_loss_alpha = .1
            dspark_l1_loss_alpha = .9
            dspark_confidence_head_alpha = 1.
            dspark_opd_loss_alpha = 0.

            def __init__(self):
                super().__init__(); self.w = torch.nn.Parameter(torch.ones(()))

            def forward(self, **kw):
                sums = {k: torch.tensor(v) for k, v in
                        {'ce_loss': 20., 'l1_loss': 4., 'confidence_loss': 1.,
                         'mtp_1_ce': 12., 'mtp_2_ce': 8., 'mtp_1_l1': 2.}.items()}
                denoms = {k: torch.tensor(2.) for k in sums}
                return self.w * 99, torch.tensor(.5), dict(
                    eval_metric_sums=sums, eval_metric_denoms=denoms,
                    acc_corrects=[torch.tensor(1.), torch.tensor(1.)],
                    acc_denoms=[torch.tensor(2.), torch.tensor(2.)])
        model = Model()
        batch = _batch()
        batch.tensors.update({k: torch.ones(1, 2) for k in
                              DSparkTrainStrategy.required_features})
        out = DSparkTrainStrategy(model).forward_loss(batch)
        window = MetricWindow(); window.update(out.metrics['log_window'])
        result = window.summary()
        self.assertEqual(result['acc'], .5)
        self.assertEqual(result['mtp_1_loss'], 6.)
        self.assertEqual(result['mtp_2_minus_1_loss'], -2.)
        self.assertAlmostEqual(result['loss_weighted'], 3.3)
        self.assertEqual(out.loss.item(), 99.)
        model.eval()
        self.assertNotIn('log_window', DSparkTrainStrategy(model).forward_loss(batch).metrics)

    def test_natural_end_and_interval_one(self):
        for interval, updates, expected_steps, expected_counts in (
            (10, 1, [1], [2]),
            (10, 2, [1, 2], [2, 4]),
            (1, 2, [1, 2], [2, 2]),
        ):
            logs = []
            strategy = WindowStrategy()
            core = TrainerCore(strategy, FakeBackend(strategy.model), accumulation_steps=2)
            controller = TrainerController(
                core, run_id='end-test', log_interval=interval,
                logger=lambda m, s: logs.append((s, m)),
            )
            controller.fit([_batch() for _ in range(updates * 2)])
            self.assertEqual([s for s, _ in logs], expected_steps)
            self.assertEqual([m['log_micro_batches'] for _, m in logs], expected_counts)
            self.assertEqual(core.metric_window.count, 0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'requires Gloo')
    def test_two_ranks_unequal_denominators(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(distributed_worker, args=(root,), nprocs=2, join=True)
            values = [torch.load(Path(root) / f'{r}.pt', weights_only=True) for r in range(2)]
        self.assertEqual(values[0], values[1])
        self.assertAlmostEqual(values[0]['ce_loss'], 32 / 11, places=6)
        self.assertAlmostEqual(values[0]['loss'], 2, places=5)


if __name__ == '__main__':
    unittest.main()
