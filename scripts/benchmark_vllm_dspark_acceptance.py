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
"""Send preformatted prompts to vLLM for DSpark acceptance-rate measurement.

The script reports request-side throughput and failures. DSpark acceptance
statistics are emitted by the vLLM server as ``SpecDecoding metrics``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_MARKER = "<｜Assistant｜>"


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
    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "response": parsed,
    }


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
        write_jsonl(
            output_handle,
            {
                "line_number": line_number,
                "ok": True,
                "elapsed_sec": result["elapsed_sec"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
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
    stats = run(args)
    logger.info("Finished request benchmark")
    for key in sorted(stats):
        logger.info("%s: %s", key, stats[key])
    logger.info(
        "Read DSpark acceptance from the vLLM server log: grep 'SpecDecoding metrics' <server.log>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
