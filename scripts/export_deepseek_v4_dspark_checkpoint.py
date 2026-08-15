# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge trained DeepSeek-V4 DSpark draft weights into a full HF checkpoint.

SpecForge runtime checkpoints save only ``draft_state_dict``. vLLM DSpark
serving expects a complete DeepSeek-V4-Flash-DSpark Hugging Face directory, so
this script overlays trained ``mtp.*`` tensors onto an existing target HF model
directory and keeps all non-draft target weights unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from specforge.export.checkpoint_io import resolve_training_state


INDEX_FILE = "model.safetensors.index.json"
DEFAULT_SHARD_SIZE_BYTES = 5 * 1024**3
FP4_BLOCK_SIZE = 32
FP4_MAX = 6.0
FP8_MAX = 448.0
UE8M0_EXPONENT_BIAS = 127
UE8M0_MIN_EXPONENT = -127
UE8M0_MAX_EXPONENT = 127
_FP4_POSITIVE_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _parse_size(value: str) -> int:
    text = value.strip().lower()
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda item: -len(item[0])):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(text)


def _copy_or_link_file(source: Path, target: Path, *, link_safetensors: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if link_safetensors and source.suffix == ".safetensors":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _materialize_base_dir(
    base_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    link_safetensors: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)

    for root, dirnames, filenames in os.walk(base_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(base_dir)
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        (output_dir / rel_root).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source = root_path / filename
            target = output_dir / rel_root / filename
            _copy_or_link_file(source, target, link_safetensors=link_safetensors)


def _load_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {INDEX_FILE}: {model_dir}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{index_path} does not contain a weight_map")
    return index


def _load_trained_mtp_tensors(
    checkpoint_path: str,
    *,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    state = resolve_training_state(checkpoint_path)
    draft_state = state.get("draft_state_dict")
    if not isinstance(draft_state, dict):
        raise ValueError(f"checkpoint has no draft_state_dict: {checkpoint_path}")

    tensors: dict[str, torch.Tensor] = {}
    for key, value in draft_state.items():
        if not key.startswith("mtp."):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        tensors[key] = tensor
    if not tensors:
        raise ValueError(f"checkpoint contains no mtp.* tensor: {checkpoint_path}")
    return tensors


def _load_base_tensor(
    model_dir: Path,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(key)
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _ue8m0_exponents(amax: torch.Tensor, *, value_max: float) -> torch.Tensor:
    minimum_amax = value_max * 2.0**-126
    exponent = torch.ceil(
        torch.log2(amax.float().clamp_min(minimum_amax) / value_max)
    )
    return exponent.clamp(UE8M0_MIN_EXPONENT, UE8M0_MAX_EXPONENT)


def _encode_scale_like(
    exponent: torch.Tensor,
    reference_scale: torch.Tensor,
) -> torch.Tensor:
    if reference_scale.dtype == torch.uint8:
        return (exponent + UE8M0_EXPONENT_BIAS).to(torch.uint8)
    return torch.exp2(exponent).to(reference_scale.dtype)


def _quantize_fp4_codes(values: torch.Tensor) -> torch.Tensor:
    abs_values = values.abs()
    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        dtype=abs_values.dtype,
        device=abs_values.device,
    )
    codes = torch.bucketize(abs_values, midpoints)

    # CUDA's cvt.rn.e2m1 uses round-to-nearest-even at exact midpoints.
    lower = codes.clamp_max(len(_FP4_POSITIVE_LEVELS) - 2)
    midpoint = midpoints[lower]
    choose_even_upper = (abs_values == midpoint) & lower.remainder(2).eq(1)
    codes = codes + choose_even_upper
    return codes.to(torch.uint8) + torch.signbit(values).to(torch.uint8) * 8


def _quantize_fp4_like(
    tensor: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 2 or reference_weight.ndim != 2 or reference_scale.ndim != 2:
        raise ValueError(
            "DeepSeek-V4 FP4 export requires rank-2 weight and scale tensors: "
            f"trained={tuple(tensor.shape)}, weight={tuple(reference_weight.shape)}, "
            f"scale={tuple(reference_scale.shape)}"
        )
    expected_shape = (reference_weight.shape[0], reference_weight.shape[1] * 2)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            "DeepSeek-V4 FP4 logical shape mismatch: "
            f"trained={tuple(tensor.shape)}, expected={expected_shape}"
        )
    if (
        reference_scale.shape[0] != tensor.shape[0]
        or tensor.shape[1] % reference_scale.shape[1] != 0
    ):
        raise ValueError(
            "unsupported DeepSeek-V4 FP4 scale geometry: "
            f"trained={tuple(tensor.shape)}, scale={tuple(reference_scale.shape)}"
        )

    block_size = tensor.shape[1] // reference_scale.shape[1]
    if block_size != FP4_BLOCK_SIZE:
        raise ValueError(
            f"DeepSeek-V4 MXFP4 block size must be {FP4_BLOCK_SIZE}, got {block_size}"
        )

    blocks = tensor.float().reshape(tensor.shape[0], -1, block_size)
    exponent = _ue8m0_exponents(blocks.abs().amax(dim=-1), value_max=FP4_MAX)
    scale = torch.exp2(exponent)
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)
    codes = _quantize_fp4_codes(scaled).reshape(tensor.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return (
        packed.to(reference_weight.dtype).contiguous(),
        _encode_scale_like(exponent, reference_scale).contiguous(),
    )


def _scale_grid_shape(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[int, int]:
    if scale.ndim == 1:
        if scale.numel() == weight.shape[0]:
            return weight.shape[0], 1
        return scale.numel(), 1
    if scale.ndim == 2:
        return scale.shape
    raise ValueError(f"unsupported DeepSeek-V4 FP8 scale rank: {scale.ndim}")


def _quantize_fp8_like(
    tensor: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 2 or tuple(tensor.shape) != tuple(reference_weight.shape):
        raise ValueError(
            "DeepSeek-V4 FP8 shape mismatch: "
            f"trained={tuple(tensor.shape)}, expected={tuple(reference_weight.shape)}"
        )

    scale_rows, scale_cols = _scale_grid_shape(tensor, reference_scale)
    row_block = (tensor.shape[0] + scale_rows - 1) // scale_rows
    col_block = (tensor.shape[1] + scale_cols - 1) // scale_cols
    padded_rows = scale_rows * row_block
    padded_cols = scale_cols * col_block
    values = tensor.float()
    if padded_rows != tensor.shape[0] or padded_cols != tensor.shape[1]:
        values = F.pad(
            values,
            (0, padded_cols - tensor.shape[1], 0, padded_rows - tensor.shape[0]),
        )

    blocks = values.reshape(scale_rows, row_block, scale_cols, col_block)
    exponent = _ue8m0_exponents(blocks.abs().amax(dim=(1, 3)), value_max=FP8_MAX)
    scale_grid = torch.exp2(exponent)
    expanded_scale = scale_grid.repeat_interleave(row_block, dim=0)
    expanded_scale = expanded_scale.repeat_interleave(col_block, dim=1)
    quantized = (values / expanded_scale).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized[: tensor.shape[0], : tensor.shape[1]]

    encoded_scale = _encode_scale_like(exponent, reference_scale)
    if reference_scale.ndim == 1:
        encoded_scale = encoded_scale.reshape(reference_scale.shape)
    return (
        quantized.to(reference_weight.dtype).contiguous(),
        encoded_scale.contiguous(),
    )


def _is_fp8_weight(tensor: torch.Tensor) -> bool:
    fp8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
        )
        if dtype is not None
    }
    return tensor.dtype in fp8_dtypes


def _requantize_like_base(
    tensors: dict[str, torch.Tensor],
    base_dir: Path,
    weight_map: dict[str, str],
    *,
    allow_missing_base_keys: bool,
) -> tuple[dict[str, torch.Tensor], int]:
    output: dict[str, torch.Tensor] = {}
    quantized_weights = 0

    for key in tqdm(
        sorted(tensors),
        desc="Requantizing DSpark draft",
        unit="tensor",
    ):
        tensor = tensors.pop(key)
        if key not in weight_map:
            if allow_missing_base_keys:
                output[key] = tensor
                continue
            raise KeyError(key)

        reference_weight = _load_base_tensor(base_dir, weight_map, key)
        scale_key = (
            f"{key[: -len('.weight')]}.scale" if key.endswith(".weight") else None
        )
        if reference_weight.dtype == torch.int8:
            if scale_key is None or scale_key not in weight_map:
                raise ValueError(f"FP4 base tensor has no scale entry: {key}")
            reference_scale = _load_base_tensor(base_dir, weight_map, scale_key)
            output[key], output[scale_key] = _quantize_fp4_like(
                tensor,
                reference_weight,
                reference_scale,
            )
            quantized_weights += 1
        elif _is_fp8_weight(reference_weight):
            if scale_key is None or scale_key not in weight_map:
                raise ValueError(f"FP8 base tensor has no scale entry: {key}")
            reference_scale = _load_base_tensor(base_dir, weight_map, scale_key)
            output[key], output[scale_key] = _quantize_fp8_like(
                tensor,
                reference_weight,
                reference_scale,
            )
            quantized_weights += 1
        else:
            if tuple(tensor.shape) != tuple(reference_weight.shape):
                raise ValueError(
                    f"DeepSeek-V4 tensor shape mismatch for {key}: "
                    f"trained={tuple(tensor.shape)}, "
                    f"expected={tuple(reference_weight.shape)}"
                )
            output[key] = tensor.to(reference_weight.dtype).contiguous()

    return output, quantized_weights


def _write_overlay_shards(
    tensors: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    max_shard_size: int,
) -> dict[str, str]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0

    for key in sorted(tensors):
        tensor = tensors[key]
        tensor_size = _tensor_nbytes(tensor)
        if current and current_size + tensor_size > max_shard_size:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = tensor
        current_size += tensor_size
    if current:
        shards.append(current)

    width = max(5, len(str(len(shards))))
    weight_map: dict[str, str] = {}
    for idx, shard in enumerate(shards, start=1):
        filename = f"model-dspark-draft-{idx:0{width}d}-of-{len(shards):0{width}d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = filename
    return weight_map


def _rewrite_base_shards(
    tensors: dict[str, torch.Tensor],
    base_dir: Path,
    output_dir: Path,
    weight_map: dict[str, str],
) -> int:
    replacements_by_shard: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        shard = weight_map.get(key)
        if shard is None:
            continue
        replacements_by_shard.setdefault(shard, {})[key] = tensor

    for shard, replacements in tqdm(
        sorted(replacements_by_shard.items()),
        desc="Rewriting affected HF shards",
        unit="shard",
    ):
        source = base_dir / shard
        target = output_dir / shard
        with safe_open(source, framework="pt", device="cpu") as handle:
            shard_tensors = {}
            for key in handle.keys():
                shard_tensors[key] = (
                    replacements[key] if key in replacements else handle.get_tensor(key)
                )
            metadata = handle.metadata()

        missing = sorted(set(replacements) - set(shard_tensors))
        if missing:
            raise ValueError(
                f"{shard} does not contain indexed replacement keys: {missing[:20]}"
            )

        temporary = target.with_name(f".{target.name}.tmp")
        save_file(shard_tensors, temporary, metadata=metadata)
        os.replace(temporary, target)

    return len(replacements_by_shard)


def _remove_stale_scale_keys(
    weight_map: dict[str, str],
    replaced_keys: set[str],
) -> list[str]:
    removed: list[str] = []
    for key in sorted(replaced_keys):
        if not key.endswith(".weight"):
            continue
        scale_key = f"{key[: -len('.weight')]}.scale"
        if scale_key in weight_map:
            del weight_map[scale_key]
            removed.append(scale_key)
    return removed


def export_deepseek_v4_dspark_checkpoint(
    *,
    base_hf_model: str,
    checkpoint: str,
    output_dir: str,
    dtype: torch.dtype | None = None,
    quantize_like_base: bool = True,
    max_shard_size: int = DEFAULT_SHARD_SIZE_BYTES,
    overwrite: bool = False,
    copy_base_weights: bool = False,
    allow_missing_base_keys: bool = False,
) -> None:
    base_dir = Path(base_hf_model).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base HF model directory does not exist: {base_dir}")
    if out_dir == base_dir:
        raise ValueError("output_dir must be different from base_hf_model")
    try:
        out_dir.relative_to(base_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output_dir must not be inside base_hf_model")
    try:
        base_dir.relative_to(out_dir)
    except ValueError:
        pass
    else:
        raise ValueError("base_hf_model must not be inside output_dir")

    base_index = _load_index(base_dir)
    trained_tensors = _load_trained_mtp_tensors(
        checkpoint,
        dtype=None if quantize_like_base else dtype,
    )
    trained_tensor_count = len(trained_tensors)
    missing = sorted(set(trained_tensors) - set(base_index["weight_map"]))
    if missing and not allow_missing_base_keys:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(
            "trained draft contains mtp.* keys absent from the base HF index: "
            f"{preview}{suffix}. Pass --allow-missing-base-keys only if the "
            "serving loader is expected to consume newly added keys."
        )
    quantized_weights = 0
    rewritten_shards = 0
    if quantize_like_base:
        trained_tensors, quantized_weights = _requantize_like_base(
            trained_tensors,
            base_dir,
            base_index["weight_map"],
            allow_missing_base_keys=allow_missing_base_keys,
        )

    _materialize_base_dir(
        base_dir,
        out_dir,
        overwrite=overwrite,
        link_safetensors=not copy_base_weights,
    )

    output_index = _load_index(out_dir)
    weight_map = output_index["weight_map"]
    if quantize_like_base:
        rewritten_shards = _rewrite_base_shards(
            trained_tensors,
            base_dir,
            out_dir,
            weight_map,
        )
        new_tensors = {
            key: tensor for key, tensor in trained_tensors.items() if key not in weight_map
        }
        overlay_map = (
            _write_overlay_shards(
                new_tensors,
                out_dir,
                max_shard_size=max_shard_size,
            )
            if new_tensors
            else {}
        )
        stale_scales: list[str] = []
    else:
        overlay_map = _write_overlay_shards(
            trained_tensors,
            out_dir,
            max_shard_size=max_shard_size,
        )
        stale_scales = _remove_stale_scale_keys(weight_map, set(overlay_map))
    weight_map.update(overlay_map)
    output_index.setdefault("metadata", {})["dspark_draft_overlay"] = "specforge"
    output_index["metadata"]["dspark_draft_overlay_format"] = (
        "base" if quantize_like_base else "floating"
    )
    with (out_dir / INDEX_FILE).open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    config_path = out_dir / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        output_config = json.load(handle)
    output_config["dspark_draft_overlay_format"] = (
        "base" if quantize_like_base else "floating"
    )
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(output_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"base HF model: {base_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"output HF model: {out_dir}")
    print(f"trained mtp tensors: {trained_tensor_count}")
    if quantize_like_base:
        print(f"requantized mtp weights: {quantized_weights}")
        print(f"regenerated scale tensors: {quantized_weights}")
        print(f"rewritten base shards: {rewritten_shards}")
        print(f"new overlay tensors: {len(overlay_map)}")
        print("draft storage format: match base HF checkpoint")
    else:
        print(f"overlay tensors written: {len(overlay_map)}")
        print(f"removed stale scale entries: {len(stale_scales)}")
        print("draft storage format: floating overlay")
    print(
        "base safetensors mode: "
        + ("copy" if copy_base_weights else "hardlink-with-copy-fallback")
    )


def _dtype_from_arg(value: str) -> torch.dtype | None:
    if value == "keep":
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise argparse.ArgumentTypeError(
            "dtype must be one of: keep, bfloat16, bf16, float16, fp16, float32, fp32"
        )
    return mapping[value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-hf-model",
        required=True,
        help="Full DeepSeek-V4-Flash-DSpark HF directory that supplies target weights.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="SpecForge output dir, checkpoint dir, or training_state.pt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination full HF directory for vLLM serving.",
    )
    parser.add_argument(
        "--dtype",
        default="keep",
        type=_dtype_from_arg,
        help=(
            "Floating dtype used only with --draft-format floating. "
            "Defaults to 'keep' so strict FP32 DSpark parameters are preserved."
        ),
    )
    parser.add_argument(
        "--draft-format",
        choices=("base", "floating"),
        default="base",
        help=(
            "Storage format for trained draft tensors. 'base' restores the base "
            "checkpoint's FP4/FP8 formats and scales; 'floating' writes a legacy "
            "floating-point overlay."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default=str(DEFAULT_SHARD_SIZE_BYTES),
        help="Maximum overlay shard size, e.g. 5GB or 1073741824.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir before writing.",
    )
    parser.add_argument(
        "--copy-base-weights",
        action="store_true",
        help="Copy base safetensors instead of hardlinking them.",
    )
    parser.add_argument(
        "--allow-missing-base-keys",
        action="store_true",
        help="Allow trained mtp.* keys that are absent from the base HF index.",
    )
    args = parser.parse_args()

    export_deepseek_v4_dspark_checkpoint(
        base_hf_model=args.base_hf_model,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        dtype=args.dtype,
        quantize_like_base=args.draft_format == "base",
        max_shard_size=_parse_size(args.max_shard_size),
        overwrite=args.overwrite,
        copy_base_weights=args.copy_base_weights,
        allow_missing_base_keys=args.allow_missing_base_keys,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
