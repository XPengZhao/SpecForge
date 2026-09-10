import unittest

import torch

from specforge.algorithms.common.dspark_metrics import acceptance_stats
from specforge.training.metric_window import MetricWindow


class TestDSparkAcceptance(unittest.TestCase):
    def test_prefix_and_window_weighting(self):
        rates = torch.tensor([[[.8, .7, .6], [.2, .9, .9], [1., 1., 1.]]],
                             requires_grad=True)
        mask = torch.tensor([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]], dtype=torch.bool)
        sums, denoms = acceptance_stats(rates, mask)
        self.assertFalse(sums['tau_probabilistic'].requires_grad)
        window = MetricWindow()
        window.update(dict(sums=sums, denoms=denoms, weights={}))
        snapshot = window.summary(reset=False)
        self.assertAlmostEqual(snapshot['accept_rate@0'], .5, places=6)
        self.assertAlmostEqual(snapshot['accept_rate@1'], .7, places=6)
        self.assertAlmostEqual(snapshot['accept_rate@2'], .6, places=6)
        self.assertAlmostEqual(snapshot['tau_probabilistic'], (2.696 + 1.2) / 2, places=6)
        sums, denoms = acceptance_stats(torch.ones(1, 1, 3), torch.ones(1, 1, 3, dtype=torch.bool))
        window.update(dict(sums=sums, denoms=denoms, weights={}))
        result = window.summary()
        self.assertAlmostEqual(result['accept_rate@0'], 2 / 3, places=6)
        self.assertAlmostEqual(result['tau_probabilistic'], (2.696 + 1.2 + 4) / 3, places=6)

    def test_empty_blocks(self):
        sums, denoms = acceptance_stats(torch.ones(2, 3, 7), torch.zeros(2, 3, 7, dtype=torch.bool))
        for value in [*sums.values(), *denoms.values()]:
            self.assertEqual(value.item(), 0)


if __name__ == '__main__':
    unittest.main()
