import io
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import torch
from datasets import Dataset

from specforge.data.preprocessing import build_eagle3_dataset
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate


class DummyTokenizer:
    pad_token_id = 0
    unk_token_id = 0
    bos_token = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        return "".join(
            f"<{message['role']}>{message['content']}</eot>" for message in messages
        )

    def __call__(
        self,
        text,
        max_length,
        truncation,
        return_tensors,
        add_special_tokens,
        return_offsets_mapping=False,
    ):
        class Encoding:
            pass

        encoding = Encoding()
        encoding.input_ids = self._ids(text, max_length)[None, :]
        if return_offsets_mapping:
            positions = torch.arange(encoding.input_ids.shape[1])
            encoding.offset_mapping = torch.stack((positions, positions + 1), dim=-1)[
                None, :
            ]
        return encoding

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return self._ids(text, max_length).tolist()

    def _ids(self, text, max_length=None):
        values = [ord(char) % 251 for char in text]
        if max_length is not None:
            values = values[:max_length]
        return torch.tensor(values, dtype=torch.long)


class TestPEagleDataPreprocessing(unittest.TestCase):
    template_name = "unit-test-peagle-data"

    @classmethod
    def setUpClass(cls):
        if cls.template_name not in TEMPLATE_REGISTRY.get_all_template_names():
            TEMPLATE_REGISTRY.register(
                cls.template_name,
                ChatTemplate(
                    assistant_header="<assistant>",
                    user_header="<user>",
                    system_prompt=None,
                    end_of_turn_token="</eot>",
                ),
            )

    def test_minimum_valid_tokens_filters_empty_loss_samples(self):
        dataset = Dataset.from_list(
            [
                {
                    "conversations": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "question"},
                    ]
                },
            ]
        )

        result = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=DummyTokenizer(),
            chat_template=self.template_name,
            max_length=128,
            num_proc=1,
            cache_dir=None,
            cache_key=None,
            minimum_valid_tokens=1,
        )

        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["loss_mask"].sum().item(), 0)

    def test_minimum_valid_tokens_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "minimum_valid_tokens"):
            build_eagle3_dataset(
                dataset=Dataset.from_list([]),
                tokenizer=DummyTokenizer(),
                chat_template=self.template_name,
                minimum_valid_tokens=-1,
            )

    def test_stage_logs_and_cached_tokenization_preserve_outputs(self):
        dataset = Dataset.from_list(
            [
                {
                    "conversations": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {"conversations": [{"role": "user", "content": "question"}]},
            ]
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as cache, redirect_stderr(output):
            options = dict(
                dataset=dataset,
                tokenizer=DummyTokenizer(),
                chat_template=self.template_name,
                max_length=128,
                num_proc=1,
                cache_dir=cache,
                cache_key="unit-test",
                minimum_valid_tokens=1,
            )
            first = build_eagle3_dataset(**options)
            # A second invocation must load cached tokens, not tokenize again.
            with patch.object(
                DummyTokenizer,
                "__call__",
                side_effect=AssertionError("cache was not reused"),
            ):
                second = build_eagle3_dataset(**options)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        for name in ("input_ids", "loss_mask", "attention_mask"):
            self.assertTrue(torch.equal(first[0][name], second[0][name]))
        for stage in ("shuffle_dataset", "tokenize_or_load_cache", "filter_samples"):
            self.assertIn(f"stage={stage} start", output.getvalue())
            self.assertIn(f"stage={stage} done", output.getvalue())
        self.assertIn("kept=1 dropped=1", output.getvalue())

    def test_parallel_tasks_match_serial_order_filtering_and_limit(self):
        from specforge.data.prompt_builder import (
            _materialize_dataset_parallel,
            _materialize_prompt_tasks,
        )

        # Cross several batch boundaries, with skipped rows and truncation.
        records = [
            {"input_ids": [i, i + 1, i + 2], "loss_mask": [0, i % 2, 1]}
            for i in range(145)
        ]
        dataset = Dataset.from_list(records)
        options = dict(max_length=2, min_loss_tokens=1, limit=None)
        serial = _materialize_prompt_tasks(
            ((row, str(i)) for i, row in enumerate(dataset.with_format("torch"))),
            **options,
        )
        parallel = _materialize_dataset_parallel(
            dataset.with_format("numpy"), workers=2, **options
        )
        self.assertEqual(parallel, serial)

        # A malformed speculative row after the cap must not surface an error.
        records[2]["loss_mask"] = []
        dataset = Dataset.from_list(records)
        options["limit"] = 1
        self.assertEqual(
            _materialize_dataset_parallel(dataset, workers=2, **options), serial[:1]
        )
        options["limit"] = None
        with self.assertRaisesRegex(ValueError, "processed dataset row 2.*mismatched"):
            _materialize_dataset_parallel(dataset, workers=2, **options)

    def test_parallel_slices_preserve_shuffled_ragged_rows(self):
        from specforge.data.prompt_builder import (
            _materialize_dataset_parallel,
            _materialize_prompt_tasks,
        )

        dataset = (
            Dataset.from_list(
                [
                    {
                        "input_ids": [[i + j for j in range(2 + i % 7)]],
                        "loss_mask": [[int(j >= i % 4) for j in range(2 + i % 7)]],
                    }
                    for i in range(145)
                ]
            )
            .shuffle(seed=42)
            .select(range(1, 144, 2))
        )
        options = dict(max_length=5, min_loss_tokens=2, limit=23)
        expected = _materialize_prompt_tasks(
            ((row, str(i)) for i, row in enumerate(dataset.with_format("torch"))),
            **options,
        )
        actual = _materialize_dataset_parallel(
            dataset.with_format("numpy"), workers=2, **options
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
