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
from safetensors.torch import save_file

from specforge.export.checkpoint_io import resolve_training_state


INDEX_FILE = "model.safetensors.index.json"
DEFAULT_SHARD_SIZE_BYTES = 5 * 1024**3


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
    dtype: torch.dtype | None = torch.bfloat16,
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
    trained_tensors = _load_trained_mtp_tensors(checkpoint, dtype=dtype)
    missing = sorted(set(trained_tensors) - set(base_index["weight_map"]))
    if missing and not allow_missing_base_keys:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(
            "trained draft contains mtp.* keys absent from the base HF index: "
            f"{preview}{suffix}. Pass --allow-missing-base-keys only if the "
            "serving loader is expected to consume newly added keys."
        )

    _materialize_base_dir(
        base_dir,
        out_dir,
        overwrite=overwrite,
        link_safetensors=not copy_base_weights,
    )

    overlay_map = _write_overlay_shards(
        trained_tensors,
        out_dir,
        max_shard_size=max_shard_size,
    )

    output_index = _load_index(out_dir)
    weight_map = output_index["weight_map"]
    stale_scales = _remove_stale_scale_keys(weight_map, set(overlay_map))
    weight_map.update(overlay_map)
    output_index.setdefault("metadata", {})["dspark_draft_overlay"] = "specforge"
    with (out_dir / INDEX_FILE).open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"base HF model: {base_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"output HF model: {out_dir}")
    print(f"replaced mtp tensors: {len(overlay_map)}")
    print(f"removed stale scale entries: {len(stale_scales)}")
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
        default="bfloat16",
        type=_dtype_from_arg,
        help="Floating dtype for exported draft tensors. Use 'keep' to preserve checkpoint dtype.",
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
        max_shard_size=_parse_size(args.max_shard_size),
        overwrite=args.overwrite,
        copy_base_weights=args.copy_base_weights,
        allow_missing_base_keys=args.allow_missing_base_keys,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
