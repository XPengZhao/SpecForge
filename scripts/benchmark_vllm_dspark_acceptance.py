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
"""Send preformatted prompts to vLLM and measure DSpark acceptance rates."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specforge.utils import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_MARKER = "<｜Assistant｜>"


@dataclass
class SpecDecodeMetrics:
    """Cumulative speculative-decoding counters exposed by vLLM."""

    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    accepted_per_pos: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4-Flash-DSpark")
    parser.add_argument("--prompt-field", default="text")
    parser.add_argument("--assistant-marker", default=DEFAULT_ASSISTANT_MARKER)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer used to save the prompt's final input tokens.",
    )
    parser.add_argument(
        "--input-preview-tokens",
        type=int,
        default=128,
        help="Number of trailing prompt tokens to save in each output record.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Optional stop string. Can be repeated.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional file for request/response summaries.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Completed-request logging interval.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_path.is_file():
        raise ValueError(f"data file does not exist: {args.data_path}")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")
    if args.input_preview_tokens < 0:
        raise ValueError("--input-preview-tokens must be >= 0")
    if (
        args.input_preview_tokens > 0
        and args.output_jsonl is not None
        and not args.tokenizer_path
    ):
        raise ValueError(
            "--tokenizer-path is required when saving token-based input previews"
        )
    if not args.assistant_marker:
        raise ValueError("--assistant-marker must not be empty")
    if args.output_jsonl is not None and args.output_jsonl.exists():
        raise ValueError(f"refusing to overwrite output file: {args.output_jsonl}")


def build_prompt(row: dict[str, Any], *, prompt_field: str, assistant_marker: str) -> str | None:
    text = row.get(prompt_field)
    if not isinstance(text, str):
        return None
    marker_pos = text.rfind(assistant_marker)
    if marker_pos < 0:
        return None
    return text[: marker_pos + len(assistant_marker)]


def iter_prompts(
    data_path: Path,
    *,
    prompt_field: str,
    assistant_marker: str,
    start_index: int,
    limit: int,
):
    emitted = 0
    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number <= start_index:
                continue
            if limit and emitted >= limit:
                break
            if not line.strip():
                yield line_number, None, "empty_line"
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None, "invalid_json"
                continue
            if not isinstance(row, dict):
                yield line_number, None, "invalid_row"
                continue
            prompt = build_prompt(
                row,
                prompt_field=prompt_field,
                assistant_marker=assistant_marker,
            )
            if prompt is None:
                yield line_number, None, "missing_prompt_or_marker"
                continue
            emitted += 1
            yield line_number, prompt, None


def post_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop:
        payload["stop"] = stop

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    elapsed = time.perf_counter() - started
    usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
    choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    spec_decode = parsed.get("spec_decode") if isinstance(parsed, dict) else None
    spec_decode_metrics = None
    if isinstance(spec_decode, dict):
        accepted_per_pos = spec_decode.get("num_accepted_tokens_per_pos", [])
        spec_decode_metrics = SpecDecodeMetrics(
            num_drafts=spec_decode["num_drafts"],
            num_draft_tokens=spec_decode["num_draft_tokens"],
            num_accepted_tokens=spec_decode["num_accepted_tokens"],
            accepted_per_pos=dict(enumerate(accepted_per_pos)),
        )
    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "output_text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "spec_decode_metrics": spec_decode_metrics,
    }


def fetch_spec_decode_metrics(
    server_url: str,
    timeout: float,
) -> SpecDecodeMetrics | None:
    """Read cumulative speculative-decoding counters from vLLM."""
    request = urllib.request.Request(
        server_url.rstrip("/") + "/metrics",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Could not read vLLM metrics: %s", exc)
        return None

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    found = False
    for line in body.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith("#")
            or not line.startswith("vllm:spec_decode")
        ):
            continue
        parts = line.split(None, 1)
        metric_name = parts[0].split("{", 1)[0]
        if len(parts) != 2 or not metric_name.endswith("_total"):
            continue
        try:
            value = int(float(parts[1]))
        except ValueError:
            continue
        found = True
        if "num_drafts" in metric_name:
            num_drafts += value
        elif "num_draft_tokens" in metric_name:
            num_draft_tokens += value
        elif "num_accepted_tokens_per_pos" in metric_name:
            marker = 'position="'
            if marker not in line:
                continue
            start = line.index(marker) + len(marker)
            position = int(line[start : line.index('"', start)])
            accepted_per_pos[position] = accepted_per_pos.get(position, 0) + value
        elif "num_accepted_tokens" in metric_name:
            num_accepted_tokens += value

    if not found:
        logger.warning("No speculative-decoding counters found at /metrics")
        return None
    return SpecDecodeMetrics(
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        accepted_per_pos=accepted_per_pos,
    )


def subtract_metrics(
    before: SpecDecodeMetrics,
    after: SpecDecodeMetrics,
) -> SpecDecodeMetrics:
    """Return counters accumulated between two metrics snapshots."""
    positions = set(before.accepted_per_pos) | set(after.accepted_per_pos)
    return SpecDecodeMetrics(
        num_drafts=after.num_drafts - before.num_drafts,
        num_draft_tokens=after.num_draft_tokens - before.num_draft_tokens,
        num_accepted_tokens=after.num_accepted_tokens - before.num_accepted_tokens,
        accepted_per_pos={
            position: after.accepted_per_pos.get(position, 0)
            - before.accepted_per_pos.get(position, 0)
            for position in positions
        },
    )


def serialize_metrics(metrics: SpecDecodeMetrics) -> dict[str, Any]:
    """Return counters and derived acceptance statistics."""
    acceptance_rate = (
        metrics.num_accepted_tokens / metrics.num_draft_tokens
        if metrics.num_draft_tokens > 0
        else 0.0
    )
    mean_acceptance_length = (
        1.0 + metrics.num_accepted_tokens / metrics.num_drafts
        if metrics.num_drafts > 0
        else 1.0
    )
    per_position = [
        metrics.accepted_per_pos[position] / metrics.num_drafts
        if metrics.num_drafts > 0
        else 0.0
        for position in sorted(metrics.accepted_per_pos)
    ]
    return {
        "num_drafts": metrics.num_drafts,
        "num_draft_tokens": metrics.num_draft_tokens,
        "num_accepted_tokens": metrics.num_accepted_tokens,
        "acceptance_rate": acceptance_rate,
        "mean_acceptance_length": mean_acceptance_length,
        "per_position_acceptance": per_position,
    }


def log_spec_decode_metrics(metrics: SpecDecodeMetrics) -> None:
    """Log exact acceptance statistics for the benchmark interval."""
    values = serialize_metrics(metrics)
    logger.info("DSpark drafts: %d", metrics.num_drafts)
    logger.info("DSpark drafted tokens: %d", metrics.num_draft_tokens)
    logger.info("DSpark accepted tokens: %d", metrics.num_accepted_tokens)
    logger.info("DSpark acceptance rate: %.4f", values["acceptance_rate"])
    logger.info(
        "DSpark mean acceptance length: %.4f",
        values["mean_acceptance_length"],
    )
    logger.info(
        "DSpark per-position acceptance: %s",
        ", ".join(f"{rate:.4f}" for rate in values["per_position_acceptance"]),
    )


def write_jsonl(handle, item: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()


def drain_completed(
    pending: set[Future],
    *,
    output_handle,
    stats: Counter,
    log_interval: int,
    started_at: float,
) -> set[Future]:
    done, pending = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        line_number = getattr(future, "line_number")
        input_tail = getattr(future, "input_tail")
        input_tail_tokens = getattr(future, "input_tail_tokens")
        try:
            result = future.result()
        except Exception as exc:
            stats["failed"] += 1
            logger.warning("line %d failed: %s", line_number, exc)
            write_jsonl(
                output_handle,
                {
                    "line_number": line_number,
                    "ok": False,
                    "input_tail": input_tail,
                    "input_tail_tokens": input_tail_tokens,
                    "error": str(exc),
                },
            )
            continue

        stats["completed"] += 1
        completion_tokens = result.get("completion_tokens")
        if isinstance(completion_tokens, int):
            stats["completion_tokens"] += completion_tokens
        prompt_tokens = result.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            stats["prompt_tokens"] += prompt_tokens
        spec_decode_metrics = result.get("spec_decode_metrics")
        if isinstance(spec_decode_metrics, SpecDecodeMetrics):
            stats["spec_decode_requests"] += 1
        else:
            spec_decode_metrics = None
        output_record = {
            "line_number": line_number,
            "ok": True,
            "elapsed_sec": result["elapsed_sec"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_tail": input_tail,
            "input_tail_tokens": input_tail_tokens,
            "output_text": result["output_text"],
            "finish_reason": result["finish_reason"],
        }
        if spec_decode_metrics is not None:
            output_record["spec_decode"] = serialize_metrics(spec_decode_metrics)
        write_jsonl(
            output_handle,
            output_record,
        )

        completed = stats["completed"]
        if log_interval > 0 and completed % log_interval == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-6)
            logger.info(
                "completed=%d failed=%d skipped=%d req/s=%.3f completion_tok/s=%.3f",
                completed,
                stats["failed"],
                stats["skipped"],
                completed / elapsed,
                stats["completion_tokens"] / elapsed,
            )
    return pending


def run(args: argparse.Namespace) -> Counter:
    endpoint = args.server_url.rstrip("/") + "/v1/completions"
    stats: Counter = Counter()
    started_at = time.perf_counter()
    tokenizer = None
    if args.tokenizer_path:
        tokenizer = load_tokenizer(
            args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )
    output_handle = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output_jsonl.open("x", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            pending: set[Future] = set()
            for line_number, prompt, skip_reason in iter_prompts(
                args.data_path,
                prompt_field=args.prompt_field,
                assistant_marker=args.assistant_marker,
                start_index=args.start_index,
                limit=args.limit,
            ):
                if prompt is None:
                    stats["skipped"] += 1
                    stats[f"skipped_{skip_reason}"] += 1
                    continue

                while len(pending) >= args.concurrency:
                    pending = drain_completed(
                        pending,
                        output_handle=output_handle,
                        stats=stats,
                        log_interval=args.log_interval,
                        started_at=started_at,
                    )

                stats["submitted"] += 1
                input_tail = None
                input_tail_tokens = 0
                if tokenizer is not None and args.input_preview_tokens > 0:
                    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    tail_token_ids = prompt_token_ids[-args.input_preview_tokens :]
                    input_tail = tokenizer.decode(tail_token_ids)
                    input_tail_tokens = len(tail_token_ids)
                future = executor.submit(
                    post_completion,
                    endpoint=endpoint,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=args.stop,
                    timeout=args.request_timeout,
                )
                setattr(future, "line_number", line_number)
                setattr(future, "input_tail", input_tail)
                setattr(future, "input_tail_tokens", input_tail_tokens)
                pending.add(future)

            while pending:
                pending = drain_completed(
                    pending,
                    output_handle=output_handle,
                    stats=stats,
                    log_interval=args.log_interval,
                    started_at=started_at,
                )
    finally:
        if output_handle is not None:
            output_handle.close()

    stats["elapsed_sec"] = time.perf_counter() - started_at
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    validate_args(args)
    metrics_before = fetch_spec_decode_metrics(args.server_url, args.request_timeout)
    stats = run(args)
    metrics_after = fetch_spec_decode_metrics(args.server_url, args.request_timeout)
    logger.info("Finished request benchmark")
    for key in sorted(stats):
        logger.info("%s: %s", key, stats[key])
    if metrics_before is not None and metrics_after is not None:
        log_spec_decode_metrics(subtract_metrics(metrics_before, metrics_after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
