# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Select low-acceptance DSpark prompts from a request-wise benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--max-mal",
        type=float,
        default=4.0,
        help="Select requests whose mean acceptance length is below this value.",
    )
    parser.add_argument(
        "--min-drafts",
        type=int,
        default=10,
        help="Require at least this many speculative draft blocks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and thresholds."""
    if not args.benchmark_path.is_file():
        raise ValueError(f"benchmark file does not exist: {args.benchmark_path}")
    if not args.data_path.is_file():
        raise ValueError(f"source data file does not exist: {args.data_path}")
    if args.data_path.resolve() == args.output_path.resolve():
        raise ValueError("--data-path and --output-path must differ")
    if not math.isfinite(args.max_mal) or args.max_mal <= 1.0:
        raise ValueError("--max-mal must be finite and greater than 1")
    if args.min_drafts < 0:
        raise ValueError("--min-drafts must be non-negative")
    if args.output_path.exists() and not args.overwrite:
        raise ValueError(
            f"{args.output_path} exists; pass --overwrite to replace it"
        )


def load_selected_lines(
    benchmark_path: Path,
    *,
    max_mal: float,
    min_drafts: int,
) -> tuple[dict[int, dict[str, float | int]], Counter]:
    """Return source line numbers satisfying the OPD difficulty criteria."""
    selected: dict[int, dict[str, float | int]] = {}
    seen_lines: set[int] = set()
    stats: Counter = Counter()

    with benchmark_path.open(encoding="utf-8") as handle:
        for benchmark_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            stats["benchmark_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid benchmark JSON at line {benchmark_line}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"benchmark line {benchmark_line} is not a JSON object"
                )

            source_line = row.get("line_number")
            if not isinstance(source_line, int) or source_line <= 0:
                raise ValueError(
                    f"benchmark line {benchmark_line} has invalid line_number"
                )
            if source_line in seen_lines:
                raise ValueError(
                    f"benchmark contains duplicate line_number={source_line}"
                )
            seen_lines.add(source_line)

            if row.get("ok") is not True:
                stats["failed_requests"] += 1
                continue
            spec_decode = row.get("spec_decode")
            if not isinstance(spec_decode, dict):
                stats["missing_spec_decode"] += 1
                continue
            mal = spec_decode.get("mean_acceptance_length")
            num_drafts = spec_decode.get("num_drafts")
            if (
                not isinstance(mal, (int, float))
                or isinstance(mal, bool)
                or not math.isfinite(mal)
                or not isinstance(num_drafts, int)
                or isinstance(num_drafts, bool)
                or num_drafts < 0
            ):
                stats["invalid_metrics"] += 1
                continue
            stats["valid_requests"] += 1
            if mal < max_mal and num_drafts >= min_drafts:
                selected[source_line] = {
                    "mean_acceptance_length": float(mal),
                    "num_drafts": num_drafts,
                }

    stats["selected_requests"] = len(selected)
    return selected, stats


def write_selected_rows(
    data_path: Path,
    output_path: Path,
    selected: dict[int, dict[str, float | int]],
) -> int:
    """Copy selected source rows in their original order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    max_source_line = 0
    with (
        data_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for source_line, line in enumerate(source, 1):
            max_source_line = source_line
            if source_line not in selected:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid source JSON at line {source_line}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"source line {source_line} is not a JSON object")
            output.write(line if line.endswith("\n") else line + "\n")
            written += 1

    missing = sorted(line_number for line_number in selected if line_number > max_source_line)
    if missing:
        raise ValueError(
            f"{len(missing)} selected line numbers exceed the source file; "
            f"first missing line_number={missing[0]}"
        )
    if written != len(selected):
        raise RuntimeError(
            f"wrote {written} rows but selected {len(selected)} benchmark rows"
        )
    return written


def main() -> None:
    """Filter the benchmark and write rollout-ready preformatted JSONL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    validate_args(args)
    selected, stats = load_selected_lines(
        args.benchmark_path,
        max_mal=args.max_mal,
        min_drafts=args.min_drafts,
    )
    written = write_selected_rows(args.data_path, args.output_path, selected)
    for name in (
        "benchmark_rows",
        "valid_requests",
        "failed_requests",
        "missing_spec_decode",
        "invalid_metrics",
        "selected_requests",
    ):
        logger.info("%s: %d", name.replace("_", " ").capitalize(), stats[name])
    logger.info("Written rows: %d", written)
    logger.info("Saved selected prompts to %s", args.output_path)


if __name__ == "__main__":
    main()
