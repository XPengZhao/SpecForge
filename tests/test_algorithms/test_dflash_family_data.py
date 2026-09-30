"""Offline tensor validation shared by DFlash and DSpark."""

import unittest

import torch

from specforge.algorithms.common.dflash_family_data import (
    normalize_offline_dspark_sample,
    normalize_offline_sample,
)


class OfflineNormalizationTest(unittest.TestCase):
    def cases(self):
        ids = torch.arange(5)
        mask = torch.ones(5)
        aux = torch.arange(40).reshape(5, 8)
        last = aux[:, :4]
        yield normalize_offline_sample, dict(
            input_ids=ids,
            loss_mask=mask,
            hidden_states=aux,
        ), {"hidden_states": "hidden_states"}
        yield normalize_offline_dspark_sample, dict(
            input_ids=ids,
            loss_mask=mask,
            aux_hidden_state=aux,
            hidden_state=last,
        ), {
            "aux_hidden_state": "hidden_states",
            "hidden_state": "target_last_hidden_states",
        }

    def test_hidden_states_accept_optional_singleton_batch_axis(self):
        for normalize, raw, keys in self.cases():
            expected = normalize(raw, 3)
            batched = {
                key: value.unsqueeze(0) if key in keys else value
                for key, value in raw.items()
            }
            actual = normalize(batched, 3)
            with self.subTest(normalizer=normalize.__name__):
                for key in expected:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0
                    )
                for raw_key, output_key in keys.items():
                    torch.testing.assert_close(
                        actual[output_key], raw[raw_key][:3].unsqueeze(0)
                    )

    def test_hidden_states_reject_invalid_rank_or_multiple_sequences(self):
        for normalize, raw, keys in self.cases():
            for key in keys:
                for shape in ((5,), (2, 5, 4), (1, 1, 5, 4)):
                    with self.subTest(
                        normalizer=normalize.__name__, key=key, shape=shape
                    ):
                        with self.assertRaisesRegex(
                            ValueError, key + " must have shape"
                        ):
                            normalize({**raw, key: torch.zeros(shape)}, 3)

    def test_lengths_must_match_after_truncation(self):
        for normalize, raw, keys in self.cases():
            for key in raw:
                with self.subTest(normalizer=normalize.__name__, key=key):
                    with self.assertRaisesRegex(
                        ValueError, "mismatched sequence lengths"
                    ):
                        normalize({**raw, key: raw[key][:2]}, 3)
            # A common truncated prefix is valid even if stored tails differ.
            normalize({**raw, "input_ids": raw["input_ids"][:4]}, 3)
