# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Package a floating DeepSeek-V4 DSpark export as a draft-only artifact."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FULL_INDEX_FILE = "model.safetensors.index.json"
DRAFT_INDEX_FILE = "dspark-draft.safetensors.index.json"
DRAFT_CONFIG_FILE = "dspark-draft-config.json"
DRAFT_PARAMETER_PREFIX = "mtp."
METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "LICENSE",
)

logger = logging.getLogger(__name__)


def _load_index(source_dir: Path) -> dict[str, Any]:
    index_path = source_dir / FULL_INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"missing source index: {index_path}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{index_path} does not contain a weight_map")
    return index


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _materialize_shard(source: Path, target: Path, *, copy_shards: bool) -> None:
    if copy_shards:
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _validate_draft_shards(
    output_dir: Path,
    draft_weight_map: dict[str, str],
) -> tuple[int, tuple[torch.dtype, ...]]:
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in draft_weight_map.items():
        keys_by_shard[shard].append(key)

    total_size = 0
    storage_dtypes: set[torch.dtype] = set()
    for shard, expected_keys in sorted(keys_by_shard.items()):
        with safe_open(output_dir / shard, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            expected_key_set = set(expected_keys)
            missing = sorted(expected_key_set - actual_keys)
            unexpected = sorted(actual_keys - expected_key_set)
            if missing:
                raise ValueError(f"{shard} is missing draft keys: {missing[:20]}")
            if unexpected:
                raise ValueError(
                    f"{shard} contains non-draft or unindexed keys: {unexpected[:20]}"
                )

            for key in expected_keys:
                tensor = handle.get_tensor(key)
                if not tensor.is_floating_point():
                    raise ValueError(
                        f"draft-only floating artifact contains non-floating tensor "
                        f"{key}: {tensor.dtype}"
                    )
                storage_dtypes.add(tensor.dtype)
                total_size += tensor.numel() * tensor.element_size()

    if not storage_dtypes:
        raise ValueError("draft-only artifact contains no tensors")
    return total_size, tuple(sorted(storage_dtypes, key=str))


def _write_readme(
    output_dir: Path,
    *,
    base_model: str,
    storage_dtypes: tuple[torch.dtype, ...],
    num_tensors: int,
    num_shards: int,
) -> None:
    dtype_names = ", ".join(
        str(storage_dtype).removeprefix("torch.")
        for storage_dtype in storage_dtypes
    )
    readme = f"""# DeepSeek-V4 Flash DSpark Draft

This repository contains only the trained DSpark draft parameters (`mtp.*`).

## Contents

- Base model: `{base_model}`
- Draft tensor prefix: `mtp.`
- Storage dtypes: `{dtype_names}`
- Draft tensors: {num_tensors}
- Draft shards: {num_shards}
- Target weights: not included
- Quantization scales: not included

## Usage

This is a draft-only training artifact, not a standalone Hugging Face model.
It cannot be loaded directly using `AutoModel.from_pretrained()` or served
directly with `vllm serve`.

Users must combine these `mtp.*` tensors with the base Target model and
quantize the Draft according to their inference backend.

Parameter-to-shard mappings are stored in
`{DRAFT_INDEX_FILE}`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def package_deepseek_v4_dspark_draft(
    *,
    source_dir: str,
    output_dir: str,
    base_model: str,
    base_model_revision: str | None,
    overwrite: bool,
    copy_shards: bool,
) -> None:
    """Create a self-describing draft-only artifact from a floating HF export.

    Args:
        source_dir: Full HF export containing floating ``mtp.*`` overlay shards.
        output_dir: Destination directory for the draft-only artifact.
        base_model: Public model ID or path identifying the required Target model.
        base_model_revision: Optional immutable revision of the Target model.
        overwrite: Whether to replace an existing output directory.
        copy_shards: Copy shard data instead of using hardlinks when possible.

    Raises:
        FileNotFoundError: If the source index, a referenced shard, or config is missing.
        ValueError: If the source is not a floating DSpark export.
    """
    source_path = Path(source_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_path}")
    if source_path == output_path:
        raise ValueError("output_dir must be different from source_dir")
    try:
        output_path.relative_to(source_path)
    except ValueError:
        pass
    else:
        raise ValueError("output_dir must not be inside source_dir")

    source_index = _load_index(source_path)
    source_metadata = source_index.get("metadata", {})
    if source_metadata.get("dspark_draft_overlay_format") != "floating":
        raise ValueError(
            "source checkpoint is not a floating DSpark overlay; export it with "
            "--draft-format floating first"
        )

    source_weight_map = source_index["weight_map"]
    draft_weight_map = {
        key: shard
        for key, shard in source_weight_map.items()
        if key.startswith(DRAFT_PARAMETER_PREFIX)
    }
    if not draft_weight_map:
        raise ValueError("source index contains no mtp.* tensors")
    scale_keys = sorted(key for key in draft_weight_map if key.endswith(".scale"))
    if scale_keys:
        raise ValueError(
            "floating draft artifact must not contain quantization scale keys: "
            f"{scale_keys[:20]}"
        )

    shard_names = sorted(set(draft_weight_map.values()))
    for shard in shard_names:
        source_shard = source_path / shard
        if not source_shard.is_file():
            raise FileNotFoundError(f"missing referenced draft shard: {source_shard}")

    config_path = source_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing source config: {config_path}")

    _prepare_output_dir(output_path, overwrite=overwrite)
    for shard in shard_names:
        _materialize_shard(
            source_path / shard,
            output_path / shard,
            copy_shards=copy_shards,
        )

    total_size, storage_dtypes = _validate_draft_shards(
        output_path,
        draft_weight_map,
    )
    storage_dtype_names = [
        str(storage_dtype).removeprefix("torch.")
        for storage_dtype in storage_dtypes
    ]

    draft_index = {
        "metadata": {
            "total_size": total_size,
            "format": "pt",
            "artifact_type": "deepseek-v4-dspark-draft",
            "storage_dtypes": storage_dtype_names,
            "parameter_prefix": DRAFT_PARAMETER_PREFIX,
        },
        "weight_map": dict(sorted(draft_weight_map.items())),
    }
    if len(storage_dtype_names) == 1:
        draft_index["metadata"]["storage_dtype"] = storage_dtype_names[0]
    with (output_path / DRAFT_INDEX_FILE).open("w", encoding="utf-8") as handle:
        json.dump(draft_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    draft_config = {
        "artifact_type": "deepseek-v4-dspark-draft",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "storage_dtypes": storage_dtype_names,
        "parameter_prefix": DRAFT_PARAMETER_PREFIX,
        "num_tensors": len(draft_weight_map),
        "num_shards": len(shard_names),
        "target_included": False,
        "serving_ready": False,
    }
    if len(storage_dtype_names) == 1:
        draft_config["storage_dtype"] = storage_dtype_names[0]
    with (output_path / DRAFT_CONFIG_FILE).open("w", encoding="utf-8") as handle:
        json.dump(draft_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for filename in METADATA_FILES:
        source_file = source_path / filename
        if source_file.is_file():
            shutil.copy2(source_file, output_path / filename)

    _write_readme(
        output_path,
        base_model=base_model,
        storage_dtypes=storage_dtypes,
        num_tensors=len(draft_weight_map),
        num_shards=len(shard_names),
    )

    logger.info("Source HF export: %s", source_path)
    logger.info("Draft-only artifact: %s", output_path)
    logger.info("Draft tensors: %d", len(draft_weight_map))
    logger.info("Draft shards: %d", len(shard_names))
    logger.info("Storage dtypes: %s", ", ".join(storage_dtype_names))
    logger.info("Logical tensor bytes: %d", total_size)
    logger.info(
        "Shard storage mode: %s",
        "copy" if copy_shards else "hardlink-with-copy-fallback",
    )


def main() -> int:
    """Parse command-line arguments and package a draft-only artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Full HF export produced with --draft-format floating.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the draft-only artifact.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Public model ID or path of the required Target model.",
    )
    parser.add_argument(
        "--base-model-revision",
        help="Optional immutable commit or revision of the Target model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir before writing.",
    )
    parser.add_argument(
        "--copy-shards",
        action="store_true",
        help="Copy draft shards instead of hardlinking them.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    package_deepseek_v4_dspark_draft(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        overwrite=args.overwrite,
        copy_shards=args.copy_shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
