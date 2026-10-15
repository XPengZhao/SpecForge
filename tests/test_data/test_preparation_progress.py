"""Progress must stay lazy, preserve input semantics, and surface failures."""

import io
import sys
import types
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from specforge.data.preparation_progress import preparation_rows, preparation_stage
from specforge.data.prompt_builder import _materialize_prompt_tasks


class _Bar:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.n = 0
        self.closed = False

    def update(self, amount):
        self.n += amount

    def close(self):
        self.closed = True


class PreparationProgressTest(unittest.TestCase):
    def test_stage_is_visible_before_work_and_records_completion(self):
        output = io.StringIO()
        with redirect_stderr(output):
            with preparation_stage("load_dataset", path="train.jsonl") as summary:
                self.assertIn("stage=load_dataset start", output.getvalue())
                self.assertNotIn("done", output.getvalue())
                summary["samples"] = 12
        self.assertIn("stage=load_dataset done", output.getvalue())
        self.assertIn("samples=12", output.getvalue())
        self.assertIn("elapsed_s=", output.getvalue())

    def test_failure_is_identified_and_original_exception_propagates(self):
        output = io.StringIO()
        error = ValueError("invalid input")
        with redirect_stderr(output), self.assertRaises(ValueError) as caught:
            with preparation_stage("materialize_tasks"):
                raise error
        self.assertIs(caught.exception, error)
        self.assertIn("stage=materialize_tasks failed", output.getvalue())
        self.assertNotIn("stage=materialize_tasks done", output.getvalue())

    def test_early_limit_counts_scanned_rows_and_closes_progress(self):
        bars = []

        def make_bar(**kwargs):
            bar = _Bar(**kwargs)
            bars.append(bar)
            return bar

        def rows():
            yield {"input_ids": [1], "loss_mask": [0]}, "skipped"
            yield {"input_ids": [2, 3, 4], "loss_mask": [0, 1, 1]}, "accepted"
            self.fail("progress must not scan past the accepted-prompt limit")

        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"tqdm": types.SimpleNamespace(tqdm=make_bar)}),
            redirect_stderr(output),
        ):
            result = _materialize_prompt_tasks(
                rows(), max_length=2, min_loss_tokens=1, limit=1
            )
        self.assertEqual(
            result, [{"payload": {"input_ids": [2, 3], "loss_mask": [0, 1]}}]
        )
        self.assertEqual(bars[0].n, 2)
        self.assertIsNone(bars[0].total)
        self.assertTrue(bars[0].closed)
        for value in ("scanned=2", "accepted=1", "skipped=1"):
            self.assertIn(value, output.getvalue())

    def test_unknown_length_input_works_without_tqdm(self):
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"tqdm": None}),
            patch(
                "specforge.data.preparation_progress.time.monotonic",
                side_effect=[0.0, 6.0, 6.0],
            ),
            redirect_stderr(output),
        ):
            with preparation_rows(iter(["row"])) as rows:
                self.assertEqual(list(rows), ["row"])
        self.assertIn("scanned=1", output.getvalue())

    def test_validation_error_closes_bar_and_keeps_failure_visible(self):
        bar = _Bar(total=1)
        output = io.StringIO()
        with (
            patch.dict(
                sys.modules, {"tqdm": types.SimpleNamespace(tqdm=lambda **_: bar)}
            ),
            redirect_stderr(output),
            self.assertRaisesRegex(ValueError, "mismatched"),
        ):
            _materialize_prompt_tasks(
                [({"input_ids": [1], "loss_mask": []}, "bad")],
                max_length=2,
                min_loss_tokens=1,
                limit=None,
                total=1,
            )
        self.assertTrue(bar.closed)
        self.assertIn("stage=materialize_tasks failed", output.getvalue())
