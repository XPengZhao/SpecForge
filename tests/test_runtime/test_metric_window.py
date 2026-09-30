"""Window weighting, first-step snapshots, and DSpark/controller integration."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def __init__(self, mode):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.dspark_loss_mode = mode

    def forward(self, **kwargs):
        sums = {name: torch.tensor(2.) for name in
                ('ce_loss', 'l1_loss', 'kl_loss', 'confidence_loss',
                 'mtp_1_loss')}
        denoms = {name: torch.tensor(1.) for name in sums}
        weights = ({'ce_loss': .3, 'l1_loss': .8} if self.dspark_loss_mode == 'original'
                   else {'kl_loss': 1.2})
        weights['confidence_loss'] = .7
        metrics = ({'log_window': dict(sums=sums, denoms=denoms, weights=weights)}
                   if self.training else {
                       'eval_metric_sums': sums, 'eval_metric_denoms': denoms,
                       'eval_objective_weights': weights,
                   })
        return self.w * (dist.get_rank() + 1), torch.tensor(.5), metrics


class DistributedLossModel(torch.nn.Module):
    """Exercise the real DSpark normalization with a small trainable proposal."""

    def __init__(self):
        super().__init__()
        from tests.test_utils.test_dflash_losses import OnlineDSparkModel

        self.logits = torch.nn.Parameter(torch.arange(12.).reshape(1, 2, 2, 3) / 10)
        self.objective = OnlineDSparkModel.__new__(OnlineDSparkModel)
        torch.nn.Module.__init__(self.objective)
        self.objective.block_size = 2
        self.objective.loss_decay_gamma = None
        self.objective.dspark_loss_mode = 'original'
        self.objective.dspark_ce_loss_alpha = 1.
        self.objective.dspark_l1_loss_alpha = 0.
        self.objective.dspark_confidence_head_alpha = 0.
        self.objective.recompute_loss = False

    def forward(self, targets, mask):
        return self.objective._compute_dspark_loss(
            draft_logits=self.logits, target_ids=targets, eval_mask=mask,
            confidence_pred=None, aligned_target_logits=None,
        )


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
            assert weights == {**expected, 'confidence_loss': .7}
            out.loss.backward()
            # A bypassed DDP forward would leave rank-local gradients 1 or 2.
            torch.testing.assert_close(model.w.grad, torch.tensor(1.5))
            window.update(out.metrics['log_window'])
            summary = window.summary()
            assert abs(summary['loss_weighted'] - 2 * sum(weights.values())) < 1e-5
            wrapped.eval()
            with torch.no_grad():
                assert 'log_window' not in strategy.forward_loss(batch).metrics

        for first_rank_count in (0, 1):
            model = DistributedLossModel()
            wrapped = torch.nn.parallel.DistributedDataParallel(model)
            masks = [torch.arange(4).reshape(1, 2, 2) < count
                     for count in (first_rank_count, 3)]
            targets = [(torch.arange(4).reshape(1, 2, 2) + r) % 3 for r in range(2)]
            with mock.patch('torch.distributed.all_reduce', wraps=dist.all_reduce) as reduce:
                loss, stats = wrapped(targets[rank], masks[rank])
                assert reduce.call_count == 1
                assert reduce.call_args.args[0].numel() == 1
            torch.testing.assert_close(stats['denoms']['ce_loss'], masks[rank].sum().float())
            loss.backward()
            reference_logits = model.logits.detach().clone().requires_grad_(True)
            numerator = sum(
                (torch.nn.functional.cross_entropy(
                    reference_logits.reshape(-1, 3), targets[r].reshape(-1),
                    reduction='none',
                ) * masks[r].reshape(-1)).sum()
                for r in range(2)
            )
            reference_loss = numerator / sum(mask.sum() for mask in masks)
            reference_grad, = torch.autograd.grad(reference_loss, reference_logits)
            torch.testing.assert_close(model.logits.grad, reference_grad)
            window = MetricWindow()
            window.update(stats)
            summary = window.summary()
            assert abs(summary['ce_loss'] - reference_loss.item()) < 1e-6
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

            def __init__(self):
                super().__init__(); self.w = torch.nn.Parameter(torch.ones(()))

            def forward(self, **kw):
                sums = {k: torch.tensor(v) for k, v in
                        {'ce_loss': 20., 'l1_loss': 4., 'confidence_loss': 1.,
                         'mtp_1_loss': 12., 'mtp_2_loss': 8.}.items()}
                denoms = {k: torch.tensor(2.) for k in sums}
                sums['acc'], denoms['acc'] = torch.tensor(2.), torch.tensor(4.)
                weights = {'ce_loss': .1, 'l1_loss': .9, 'confidence_loss': 1.}
                metrics = ({'log_window': dict(sums=sums, denoms=denoms, weights=weights)}
                           if self.training else {
                               'eval_metric_sums': sums, 'eval_metric_denoms': denoms,
                               'eval_objective_weights': weights,
                           })
                return self.w * 99, torch.tensor(.5), metrics
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

    def test_window_metrics_bypass_per_step_reduction_and_scalarization(self):
        strategy = WindowStrategy()
        core = TrainerCore(strategy, FakeBackend(strategy.model), accumulation_steps=2)
        with (
            mock.patch('specforge.training.controller._dp_mean_scalars',
                       side_effect=AssertionError('window metrics reduced per step')),
            mock.patch('specforge.training.controller._scalar',
                       wraps=lambda value: float(value.item())) as scalar,
        ):
            first = core.train_step(_batch())
            self.assertEqual(scalar.call_count, 1)  # StepResult.loss only.
            second = core.train_step(_batch())
            self.assertEqual(scalar.call_count, 3)  # Loss and boundary grad norm.
        self.assertFalse(first.optimizer_stepped)
        self.assertTrue(second.optimizer_stepped)
        self.assertEqual(core.metric_window.count, 2)
        self.assertEqual(core.metric_window.summary()['log_micro_batches'], 2)

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
