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
"""Convert DeepSeek request-log JSONL into preformatted training JSONL."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_MARKER = "<｜Assistant｜>"
DEFAULT_END_MARKER = "<｜end▁of▁sentence｜>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--assistant-marker", default=DEFAULT_ASSISTANT_MARKER)
    parser.add_argument("--end-marker", default=DEFAULT_END_MARKER)
    parser.add_argument(
        "--input-format",
        choices=("json", "python-literal"),
        default="json",
        help="Input record encoding. Use python-literal for lines such as {'prompt': '...'}.",
    )
    return parser.parse_args()


def validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")
    if output_path.suffix != ".jsonl":
        raise ValueError(f"output path must end in .jsonl: {output_path}")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {output_path}")


def validate_markers(assistant_marker: str, end_marker: str) -> None:
    if not assistant_marker:
        raise ValueError("assistant marker must not be empty")
    if not end_marker:
        raise ValueError("end marker must not be empty")


def complete_prompt_prefixes(
    prompt: str,
    *,
    assistant_marker: str,
    end_marker: str,
) -> Iterator[str]:
    """Yield prefixes ending at each complete non-empty assistant turn."""

    search_start = 0
    while True:
        assistant_start = prompt.find(assistant_marker, search_start)
        if assistant_start < 0:
            break

        content_start = assistant_start + len(assistant_marker)
        next_assistant = prompt.find(assistant_marker, content_start)
        response_end = prompt.find(end_marker, content_start)
        if response_end >= 0 and (
            next_assistant < 0 or response_end < next_assistant
        ):
            response = prompt[content_start:response_end]
            if response.strip():
                yield prompt[: response_end + len(end_marker)]

        search_start = content_start


def parse_record(line: str, input_format: str) -> object:
    """Parse one input record in the explicitly selected format."""

    if input_format == "json":
        return json.loads(line)
    if input_format == "python-literal":
        return ast.literal_eval(line)
    raise ValueError(f"unsupported input format: {input_format}")


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    input_format: str,
    prompt_field: str,
    assistant_marker: str,
    end_marker: str,
) -> Counter:
    stats: Counter = Counter()
    seen_prefix_hashes: set[bytes] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(encoding="utf-8") as input_handle,
        output_path.open("x", encoding="utf-8") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            stats["input_rows"] += 1

            try:
                row = parse_record(line, input_format)
            except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
                stats["invalid_records"] += 1
                logger.warning(
                    "Skipping line %d: invalid %s record: %s",
                    line_number,
                    input_format,
                    exc,
                )
                continue

            if not isinstance(row, dict):
                stats["invalid_rows"] += 1
                logger.warning("Skipping line %d: expected a JSON object", line_number)
                continue

            prompt = row.get(prompt_field)
            if not isinstance(prompt, str) or not prompt:
                stats["missing_prompt"] += 1
                logger.warning(
                    "Skipping line %d: %r is not a non-empty string",
                    line_number,
                    prompt_field,
                )
                continue

            found_complete_response = False
            for converted in complete_prompt_prefixes(
                prompt,
                assistant_marker=assistant_marker,
                end_marker=end_marker,
            ):
                found_complete_response = True
                stats["complete_responses"] += 1

                prefix_hash = hashlib.sha256(converted.encode("utf-8")).digest()
                if prefix_hash in seen_prefix_hashes:
                    stats["duplicate_responses"] += 1
                    continue
                seen_prefix_hashes.add(prefix_hash)

                output_handle.write(
                    json.dumps({"text": converted}, ensure_ascii=False) + "\n"
                )
                stats["output_rows"] += 1
                stats["truncated_characters"] += len(prompt) - len(converted)

            if not found_complete_response:
                stats["no_complete_response"] += 1
                logger.warning(
                    "Skipping line %d: no complete assistant response", line_number
                )

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        validate_paths(args.input_path, args.output_path)
        validate_markers(args.assistant_marker, args.end_marker)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    stats = convert_file(
        args.input_path,
        args.output_path,
        input_format=args.input_format,
        prompt_field=args.prompt_field,
        assistant_marker=args.assistant_marker,
        end_marker=args.end_marker,
    )
    logger.info("Input rows: %d", stats["input_rows"])
    logger.info("Invalid records: %d", stats["invalid_records"])
    logger.info("Invalid rows: %d", stats["invalid_rows"])
    logger.info("Missing prompt: %d", stats["missing_prompt"])
    logger.info("No complete response: %d", stats["no_complete_response"])
    logger.info("Complete responses: %d", stats["complete_responses"])
    logger.info("Duplicate responses: %d", stats["duplicate_responses"])
    logger.info("Output rows (unique responses): %d", stats["output_rows"])
    logger.info("Truncated characters: %d", stats["truncated_characters"])
    logger.info("Saved converted data to %s", args.output_path)


if __name__ == "__main__":
    main()
