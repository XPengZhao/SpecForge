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
"""Measure DSpark behavior on exact prefixes from stored target rollouts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specforge.utils import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_MARKER = "<｜Assistant｜>"


@dataclass
class StoredTrajectoryStats:
    """Aggregate stored-trajectory and live-verification measurements."""

    matched_blocks: int
    attempted_requests: int
    unmatched_blocks: int
    missing_trace_blocks: int
    stored_prefix_tokens: int
    live_accepted_tokens: int
    stored_prefix_per_position: list[int]
    stored_correct_per_position: list[int]
    live_accepted_per_position: list[int]
    drafted_per_position: list[int]

    @classmethod
    def create(cls, block_size: int) -> StoredTrajectoryStats:
        """Create zeroed counters for one draft block size."""
        zeros = [0] * block_size
        return cls(
            matched_blocks=0,
            attempted_requests=0,
            unmatched_blocks=0,
            missing_trace_blocks=0,
            stored_prefix_tokens=0,
            live_accepted_tokens=0,
            stored_prefix_per_position=zeros.copy(),
            stored_correct_per_position=zeros.copy(),
            live_accepted_per_position=zeros.copy(),
            drafted_per_position=zeros.copy(),
        )

    def observe(
        self,
        *,
        draft_token_ids: list[int],
        target_token_ids: list[int],
        accepted_length: int,
    ) -> tuple[int, list[bool]]:
        """Record one matched-anchor draft block."""
        compared = min(
            len(draft_token_ids),
            len(target_token_ids),
            len(self.drafted_per_position),
        )
        correctness = [
            draft_token_ids[index] == target_token_ids[index]
            for index in range(compared)
        ]
        prefix_length = 0
        for matched in correctness:
            if not matched:
                break
            prefix_length += 1

        accepted_length = min(max(int(accepted_length), 0), compared)
        self.matched_blocks += 1
        self.stored_prefix_tokens += prefix_length
        self.live_accepted_tokens += accepted_length
        for position in range(compared):
            self.drafted_per_position[position] += 1
            self.stored_correct_per_position[position] += int(correctness[position])
            self.stored_prefix_per_position[position] += int(position < prefix_length)
            self.live_accepted_per_position[position] += int(position < accepted_length)
        return prefix_length, correctness


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4-Flash-DSpark")
    parser.add_argument("--prompt-field", default="text")
    parser.add_argument("--assistant-marker", default=DEFAULT_ASSISTANT_MARKER)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--anchors-per-row", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--max-anchor-attempts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate benchmark arguments."""
    if not args.data_path.is_file():
        raise ValueError(f"data file does not exist: {args.data_path}")
    if args.start_index < 0 or args.limit < 0:
        raise ValueError("--start-index and --limit must be non-negative")
    if args.anchors_per_row <= 0 or args.block_size <= 0:
        raise ValueError("--anchors-per-row and --block-size must be positive")
    if args.max_anchor_attempts <= 0:
        raise ValueError("--max-anchor-attempts must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive for anchor resampling")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.request_timeout <= 0 or args.log_interval <= 0:
        raise ValueError("--request-timeout and --log-interval must be positive")
    if (
        args.output_jsonl is not None
        and args.output_jsonl.exists()
        and not args.overwrite
    ):
        raise ValueError(
            f"output file exists: {args.output_jsonl}; pass --overwrite to replace it"
        )


def tokenize_rollout(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    prompt_field: str,
    assistant_marker: str,
) -> tuple[list[int], int]:
    """Tokenize one stored rollout and locate its final assistant response."""
    text = row.get(prompt_field)
    if not isinstance(text, str) or not text:
        raise ValueError(f"{prompt_field!r} is not a non-empty string")
    marker_position = text.rfind(assistant_marker)
    if marker_position < 0:
        raise ValueError(f"missing assistant marker: {assistant_marker}")
    response_text_start = marker_position + len(assistant_marker)
    prefix = text[:response_text_start]
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    response_start = len(tokenizer.encode(prefix, add_special_tokens=False))
    if response_start >= len(input_ids):
        raise ValueError("stored rollout has no response tokens")
    return [int(token_id) for token_id in input_ids], response_start


def select_anchor_positions(
    *,
    response_start: int,
    response_end: int,
    block_size: int,
    count: int,
) -> list[int]:
    """Select deterministic, evenly spaced anchors with complete labels."""
    first_anchor = response_start
    last_anchor = response_end - block_size - 1
    if last_anchor < first_anchor:
        return []
    available = last_anchor - first_anchor + 1
    kept = min(count, available)
    if kept == 1:
        return [first_anchor + available // 2]
    return sorted(
        {
            first_anchor
            + round(slot * (available - 1) / (kept - 1))
            for slot in range(kept)
        }
    )


def post_completion(
    *,
    endpoint: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    """Request one traced completion from vLLM."""
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "return_token_ids": True,
        "vllm_xargs": {"collect_spec_decode_trace": 1},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"completion response is not an object: {parsed}")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"completion response has no choice: {parsed}")
    return parsed


def extract_token_ids(response: dict[str, Any]) -> list[int]:
    """Extract generated token IDs from a completion response."""
    choices = response["choices"]
    token_ids = choices[0].get("token_ids")
    if not isinstance(token_ids, list):
        spec_decode = response.get("spec_decode")
        token_ids = (
            spec_decode.get("verified_token_ids")
            if isinstance(spec_decode, dict)
            else None
        )
    if not isinstance(token_ids, list):
        raise RuntimeError(f"completion response has no token IDs: {response}")
    return [int(token_id) for token_id in token_ids]


def extract_first_trace(response: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first speculative trace after the target-sampled anchor."""
    spec_decode = response.get("spec_decode")
    trace = spec_decode.get("trace") if isinstance(spec_decode, dict) else None
    if not isinstance(trace, list):
        raise RuntimeError(f"completion response has no speculative trace: {response}")
    entries = [entry for entry in trace if isinstance(entry, dict)]
    if not entries:
        return None
    return min(entries, key=lambda entry: int(entry.get("response_prefix_length", 0)))


def ratios(numerators: list[int], denominators: list[int]) -> list[float]:
    """Return elementwise ratios with zero-denominator protection."""
    return [
        numerator / denominator if denominator else 0.0
        for numerator, denominator in zip(numerators, denominators)
    ]


def summary(stats: StoredTrajectoryStats) -> dict[str, Any]:
    """Build serializable aggregate metrics."""
    matched = max(stats.matched_blocks, 1)
    return {
        "matched_blocks": stats.matched_blocks,
        "attempted_requests": stats.attempted_requests,
        "unmatched_blocks": stats.unmatched_blocks,
        "missing_trace_blocks": stats.missing_trace_blocks,
        "anchor_match_rate": (
            stats.matched_blocks / stats.attempted_requests
            if stats.attempted_requests
            else 0.0
        ),
        "stored_trajectory_mean_acceptance_length": (
            1.0 + stats.stored_prefix_tokens / matched
        ),
        "stored_trajectory_prefix_acceptance": ratios(
            stats.stored_prefix_per_position,
            [matched] * len(stats.stored_prefix_per_position),
        ),
        "stored_trajectory_independent_accuracy": ratios(
            stats.stored_correct_per_position,
            stats.drafted_per_position,
        ),
        "live_trace_mean_acceptance_length": (
            1.0 + stats.live_accepted_tokens / matched
        ),
        "live_trace_prefix_acceptance": ratios(
            stats.live_accepted_per_position,
            [matched] * len(stats.live_accepted_per_position),
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the stored-trajectory prefix benchmark."""
    validate_args(args)
    tokenizer = load_tokenizer(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    endpoint = args.server_url.rstrip("/") + "/v1/completions"
    stats = StoredTrajectoryStats.create(args.block_size)
    output_handle = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        output_handle = args.output_jsonl.open(mode, encoding="utf-8")

    selected_rows = 0
    try:
        with args.data_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number <= args.start_index or not line.strip():
                    continue
                if args.limit and selected_rows >= args.limit:
                    break
                selected_rows += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                input_ids, response_start = tokenize_rollout(
                    tokenizer,
                    row,
                    prompt_field=args.prompt_field,
                    assistant_marker=args.assistant_marker,
                )
                anchors = select_anchor_positions(
                    response_start=response_start,
                    response_end=len(input_ids),
                    block_size=args.block_size,
                    count=args.anchors_per_row,
                )
                for anchor_index, anchor in enumerate(anchors):
                    expected_anchor = input_ids[anchor]
                    matched_response = None
                    attempts = 0
                    for attempt in range(args.max_anchor_attempts):
                        attempts += 1
                        stats.attempted_requests += 1
                        request_seed = (
                            args.seed
                            + line_number * args.anchors_per_row * args.max_anchor_attempts
                            + anchor_index * args.max_anchor_attempts
                            + attempt
                        )
                        response = post_completion(
                            endpoint=endpoint,
                            model=args.model,
                            prompt_token_ids=input_ids[:anchor],
                            max_tokens=args.block_size + 1,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            seed=request_seed,
                            timeout=args.request_timeout,
                        )
                        generated_ids = extract_token_ids(response)
                        if generated_ids and generated_ids[0] == expected_anchor:
                            matched_response = response
                            break

                    record: dict[str, Any] = {
                        "line_number": line_number,
                        "anchor_position": anchor,
                        "attempts": attempts,
                        "expected_anchor_token_id": expected_anchor,
                    }
                    if matched_response is None:
                        stats.unmatched_blocks += 1
                        record["status"] = "anchor_not_matched"
                    else:
                        trace = extract_first_trace(matched_response)
                        if trace is None:
                            stats.missing_trace_blocks += 1
                            record["status"] = "missing_trace"
                        else:
                            draft_token_ids = trace.get("draft_token_ids")
                            accepted_length = trace.get("accepted_length")
                            if not isinstance(draft_token_ids, list) or not isinstance(
                                accepted_length, int
                            ):
                                raise RuntimeError(f"invalid speculative trace: {trace}")
                            draft_token_ids = [
                                int(token_id) for token_id in draft_token_ids
                            ][: args.block_size]
                            target_token_ids = input_ids[
                                anchor + 1 : anchor + 1 + len(draft_token_ids)
                            ]
                            prefix_length, correctness = stats.observe(
                                draft_token_ids=draft_token_ids,
                                target_token_ids=target_token_ids,
                                accepted_length=accepted_length,
                            )
                            record.update(
                                {
                                    "status": "ok",
                                    "draft_token_ids": draft_token_ids,
                                    "target_token_ids": target_token_ids,
                                    "stored_correctness": correctness,
                                    "stored_prefix_length": prefix_length,
                                    "live_accepted_length": accepted_length,
                                    "response_prefix_length": trace.get(
                                        "response_prefix_length"
                                    ),
                                }
                            )
                    if output_handle is not None:
                        output_handle.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        output_handle.flush()
                    if (
                        stats.matched_blocks
                        and stats.matched_blocks % args.log_interval == 0
                    ):
                        logger.info("progress: %s", summary(stats))
    finally:
        if output_handle is not None:
            output_handle.close()

    result = summary(stats)
    logger.info("stored-trajectory benchmark: %s", result)
    return result


def main() -> int:
    """Run the command-line benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run(parse_args())
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
