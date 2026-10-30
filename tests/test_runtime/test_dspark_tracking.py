"""Import-light tests for the DSpark dashboard policy."""

import unittest
from unittest import mock

from specforge.training.tracking import (
    TrackerLogger,
    dspark_tensorboard_metrics,
    scalar_metrics,
    training_metric_names,
)


class _Tracker:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))


class DSparkTrackingTest(unittest.TestCase):
    def test_dspark_filter_preserves_internal_and_console_metrics(self):
        metrics = {
            "loss": 1.2, "loss_weighted": 1.3, "ce_loss": 0.0,
            "l1_loss": 0.5, "kl_loss": 0.0, "mtp_1_loss": 0.1,
            "accept_rate@0": 0.8, "mtp_1_accuracy": 0.0, "tau_probabilistic": 3.0,
            "confidence_mae": 0.12, "eval/confidence_mae": 0.0,
            "eval/avg_acc": 0.8, "eval/mtp_1_accuracy": 0.8,
            "eval/per_position_acc": [0.8], "eval/accept_rate@0": 0.9,
            "eval/simulated_acc_len": 2.0, "eval/simulated_top1_mal": 3.0,
        }
        tracker, console = _Tracker(), mock.Mock()
        logger = TrackerLogger(
            tracker, console_logger=console, metric_filter=dspark_tensorboard_metrics
        )
        logger(metrics, 10)
        self.assertEqual(tracker.logged[0], ({
            "train/loss_weighted": 1.3, "train/ce_loss": 0.0,
            "train/l1_loss": 0.5, "train/tau_probabilistic": 3.0,
            "train/confidence_mae": 0.12, "eval/confidence_mae": 0.0,
            "train/position_loss/01_ce": 0.1,
            "train/accept_rate@0": 0.8, "train/top1_acc@0": 0.0,
            "eval/top1_acc@0": 0.8, "eval/accept_rate@0": 0.9,
            "eval/simulated_acc_len": 2.0,
        }, 10))
        full = training_metric_names(scalar_metrics(metrics))
        console.assert_called_once_with(full, 10)
        self.assertIn("eval/simulated_top1_mal", metrics)
        diagnostic = dspark_tensorboard_metrics(full, diagnostics=True)
        self.assertEqual(diagnostic["train/position_loss/01_ce"], 0.1)
        self.assertEqual(diagnostic["train/loss"], 1.2)
        kl = dspark_tensorboard_metrics(full, loss_mode="kl")
        self.assertIn("train/kl_loss", kl)
        self.assertEqual(kl["train/confidence_mae"], 0.12)
        self.assertEqual(kl["eval/confidence_mae"], 0.0)
        self.assertNotIn("train/ce_loss", kl)
        self.assertNotIn("train/l1_loss", kl)

    def test_accuracy_tags_are_zero_based_in_both_modes(self):
        metrics = {
            f"{phase}/mtp_{i + 1}_accuracy": i / 10
            for phase in ("train", "eval") for i in range(7)
        }
        metrics.update({f"{phase}/accept_rate@{i}": 0.9
                        for phase in ("train", "eval") for i in range(7)})
        for mode in ("original", "kl"):
            for diagnostics in (False, True):
                selected = dspark_tensorboard_metrics(
                    metrics, loss_mode=mode, diagnostics=diagnostics)
                self.assertEqual(len(selected), 28)
                for phase in ("train", "eval"):
                    for i in range(7):
                        self.assertEqual(selected[f"{phase}/top1_acc@{i}"], i / 10)
                        self.assertEqual(selected[f"{phase}/accept_rate@{i}"], 0.9)

    def test_all_positions_are_grouped_by_objective_without_duplicate_tags(self):
        for mode, component in [("original", "ce"), ("kl", "kl")]:
            metrics = {}
            for i in range(1, 13):
                metrics[f"train/mtp_{i}_loss"] = i / 10
                metrics[f"eval/mtp_{i}_ce"] = i / 20
                metrics[f"eval/mtp_{i}_kl"] = i / 30
                metrics[f"eval/mtp_{i}_l1"] = i / 40
                metrics[f"train/mtp_{i}_minus_{i-1}_loss"] = 0.1
            selected = dspark_tensorboard_metrics(metrics, loss_mode=mode)
            self.assertEqual(len(selected), 24)
            for i in range(1, 13):
                self.assertEqual(selected[f"train/position_loss/{i:02d}_{component}"], i / 10)
                self.assertEqual(
                    selected[f"eval/position_loss/{i:02d}_{component}"],
                    metrics[f"eval/mtp_{i}_{component}"],
                )
            self.assertEqual(sorted(selected)[0], f"eval/position_loss/01_{component}")
            diagnostic = dspark_tensorboard_metrics(metrics, loss_mode=mode, diagnostics=True)
            self.assertIn("eval/position_loss/01_l1", diagnostic)
            self.assertIn("train/mtp_2_minus_1_loss", diagnostic)

