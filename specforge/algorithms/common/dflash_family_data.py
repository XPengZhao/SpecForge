"""Shared DFlash-family normalization and padding adapters."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features

NORMALIZER_ID = "dflash_family_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"


def _normalize_hidden_state(value, *, max_len: int, label: str):
    """Accept one cached sequence, with or without its singleton batch axis."""
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(
            f"{label} must have shape [seq, width] or [1, seq, width], "
            f"got {tuple(value.shape)}"
        )
    return value[:max_len].unsqueeze(0)


def _validate_sequence_lengths(features, *, label: str):
    lengths = {key: value.shape[1] for key, value in features.items()}
    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(
            f"{label} features have mismatched sequence lengths after "
            f"truncation: {details}"
        )


def normalize_offline_sample(raw, max_len: int):
    """Normalize raw DFlash/Domino capture tensors without target projection."""
    normalized = {
        "input_ids": raw["input_ids"][:max_len].unsqueeze(0),
        "loss_mask": raw["loss_mask"][:max_len].unsqueeze(0),
        "hidden_states": _normalize_hidden_state(
            raw["hidden_states"],
            max_len=max_len,
            label="offline DFlash-family hidden_states",
        ),
    }
    _validate_sequence_lengths(normalized, label="offline DFlash-family")
    return normalized


def build_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=("input_ids", "loss_mask", "hidden_states"),
        target_repr=None,
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_offline_normalizer(max_len, **_topology):
    return partial(normalize_offline_sample, max_len=max_len)


def normalize_offline_dspark_sample(raw, max_len: int, dspark_supervision="response"):
    """Map stored target captures to the DSpark training feature names."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].clone().unsqueeze(0)
    if dspark_supervision not in ("response", "full_sequence"):
        raise ValueError(f"Unknown DSpark supervision: {dspark_supervision}")
    if dspark_supervision == "full_sequence":
        if not raw.get("loss_mask_is_token_aligned", False):
            raise ValueError("Full-sequence DSpark requires a token-aligned, unpadded cache (DeepSpec v2)")
        # Before collation: only real cached tokens become supervised.
        # The collator still pads loss_mask with zeros.
        loss_mask.fill_(1)
    if loss_mask.numel() > 0 and not raw.get("loss_mask_is_token_aligned", False):
        loss_mask[0, -1] = 0

    normalized = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": _normalize_hidden_state(
            raw["aux_hidden_state"],
            max_len=max_len,
            label="offline DSpark aux_hidden_state",
        ),
        "target_last_hidden_states": _normalize_hidden_state(
            raw["hidden_state"],
            max_len=max_len,
            label="offline DSpark hidden_state",
        ),
    }
    _validate_sequence_lengths(normalized, label="offline DSpark")
    return normalized


def build_offline_dspark_reader(
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
    from specforge.runtime.data_plane.deepspec_cache import (
        DeepSpecCacheReader, is_deepspec_cache,
    )

    if is_deepspec_cache(hidden_states_path):
        return DeepSpecCacheReader(
            hidden_states_path, run_id=run_id, ttt_length=ttt_length, max_len=max_len,
        )

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy="dspark",
        feature_keys=(
            "input_ids",
            "loss_mask",
            "aux_hidden_state",
            "hidden_state",
        ),
        optional_feature_keys=("loss_mask_is_token_aligned",),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_offline_dspark_normalizer(max_len, dspark_supervision="response", **_topology):
    return partial(
        normalize_offline_dspark_sample, max_len=max_len,
        dspark_supervision=dspark_supervision,
    )


def build_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
            },
            required_keys=("input_ids", "loss_mask", "hidden_states"),
        )

    return collate


def build_dspark_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
                "target_last_hidden_states": 1,
            },
            required_keys=(
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            ),
        )

    return collate


__all__ = [
    "DSPARK_NORMALIZER_ID",
    "NORMALIZER_ID",
    "build_collator",
    "build_offline_dspark_normalizer",
    "build_offline_dspark_reader",
    "build_dspark_collator",
    "build_offline_normalizer",
    "build_offline_reader",
    "normalize_offline_dspark_sample",
    "normalize_offline_sample",
]
