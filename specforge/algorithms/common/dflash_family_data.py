"""Shared DFlash-family normalization and padding adapters."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features

NORMALIZER_ID = "dflash_family_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"


def normalize_offline_sample(raw, max_len: int):
    """Normalize raw DFlash/Domino capture tensors without target projection."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0)
    hidden_states = raw["hidden_states"]
    if hidden_states.dim() == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "offline DFlash-family hidden_states must have shape "
                "[seq, width] or [1, seq, width], got "
                f"{tuple(hidden_states.shape)}"
            )
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.dim() != 2:
        raise ValueError(
            "offline DFlash-family hidden_states must have shape "
            "[seq, width] or [1, seq, width], got "
            f"{tuple(hidden_states.shape)}"
        )
    hidden_states = hidden_states[:max_len].unsqueeze(0)
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline DFlash-family features have mismatched sequence lengths "
            f"after truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"hidden_states={hidden_states.shape[1]}"
        )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
    }


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


def normalize_offline_dspark_sample(raw, max_len: int):
    """Map stored target captures to the DSpark training feature names."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].clone().unsqueeze(0)
    if loss_mask.numel() > 0:
        loss_mask[0, -1] = 0

    def normalize_hidden_state(key):
        value = raw[key]
        if value.dim() == 3:
            if value.shape[0] != 1:
                raise ValueError(
                    f"offline DSpark {key} must have shape [seq, width] or "
                    f"[1, seq, width], got {tuple(value.shape)}"
                )
            value = value.squeeze(0)
        if value.dim() != 2:
            raise ValueError(
                f"offline DSpark {key} must have shape [seq, width] or "
                f"[1, seq, width], got {tuple(value.shape)}"
            )
        return value[:max_len].unsqueeze(0)

    hidden_states = normalize_hidden_state("aux_hidden_state")
    target_last_hidden_states = normalize_hidden_state("hidden_state")
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        hidden_states.shape[1],
        target_last_hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline DSpark features have mismatched sequence lengths after "
            f"truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"hidden_states={hidden_states.shape[1]}, "
            "target_last_hidden_states="
            f"{target_last_hidden_states.shape[1]}"
        )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_last_hidden_states": target_last_hidden_states,
    }


def build_offline_dspark_reader(
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

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
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_offline_dspark_normalizer(max_len, **_topology):
    return partial(normalize_offline_dspark_sample, max_len=max_len)


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
