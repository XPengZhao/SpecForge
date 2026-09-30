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
"""Generate on-policy training responses with a target vLLM completions server.

Each input row must contain ``conversations`` (role/content). The last
assistant turn is dropped, the remainder is rendered with
``apply_chat_template(..., add_generation_prompt=True)``, and the request goes
to ``/v1/completions``.
Each output row stores the prefix plus the new assistant turn as
``conversations``. Downstream dump and training apply the chat template again.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specforge.utils import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_END_MARKER = "<｜end▁of▁sentence｜>"


@dataclass(frozen=True)
class RolloutJob:
    """One prepared target completion request."""

    source_line: int
    prompt: str
    max_tokens: int
    conversations_prefix: tuple[dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4-Flash-DSpark")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--end-marker", default=DEFAULT_END_MARKER)
    parser.add_argument("--max-length", type=int, default=128000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--drop-truncated", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to apply_chat_template.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate rollout arguments before opening the output file."""
    if not args.data_path.is_file():
        raise ValueError(f"data file does not exist: {args.data_path}")
    if args.data_path.resolve() == args.output_path.resolve():
        raise ValueError("--data-path and --output-path must differ")
    if args.max_length <= 1:
        raise ValueError("--max-length must be greater than 1")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.start_index < 0 or args.limit < 0:
        raise ValueError("--start-index and --limit must be non-negative")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be greater than 0")
    if args.request_timeout <= 0 or args.retry_delay < 0:
        raise ValueError("--request-timeout must be positive and --retry-delay non-negative")
    if args.output_path.exists() and not args.resume:
        raise ValueError(f"{args.output_path} exists; pass --resume to append missing rows")


def load_completed_rows(output_path: Path) -> set[int]:
    """Return source line numbers already present in a rollout file."""
    if not output_path.exists():
        return set()
    completed: set[int] = set()
    with output_path.open(encoding="utf-8") as handle:
        for output_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid existing output JSON at line {output_line}: {exc}"
                ) from exc
            source_line = row.get("source_line_number")
            if not isinstance(source_line, int):
                raise ValueError(
                    f"existing output line {output_line} has no integer source_line_number"
                )
            completed.add(source_line)
    return completed


def _copy_messages(messages: list[Any]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each conversation item must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("conversation items need string role and content")
        copied.append({"role": role, "content": content})
    return copied


def prompt_from_conversations(
    tokenizer: Any,
    conversations: Any,
    *,
    disable_thinking: bool,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Drop the last assistant turn and render a generation prompt."""
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")
    messages = _copy_messages(conversations)
    if messages[-1]["role"] == "assistant":
        prefix = messages[:-1]
    else:
        prefix = messages
    if not prefix or prefix[-1]["role"] != "user":
        raise ValueError("conversation prompt must end on a user turn")
    kwargs: dict[str, Any] = {}
    if disable_thinking:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    prompt = tokenizer.apply_chat_template(
        prefix,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("apply_chat_template returned an empty generation prompt")
    return prompt, tuple(prefix)


def post_completion(
    endpoint: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    end_marker: str,
    timeout: float,
) -> dict[str, Any]:
    """Request one target rollout from the OpenAI-compatible completion API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": [end_marker],
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

    choices = parsed.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"completion response has no choice: {parsed}")
    choice = choices[0]
    output_text = choice.get("text")
    usage = parsed.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    stopped_after_token = (
        choice.get("finish_reason") == "stop"
        and isinstance(completion_tokens, int)
        and completion_tokens > 0
    )
    if not isinstance(output_text, str) or (not output_text and not stopped_after_token):
        raise RuntimeError(f"completion response has empty text: {parsed}")
    return {
        "output_text": output_text,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
    }


def request_with_retries(
    endpoint: str,
    *,
    args: argparse.Namespace,
    prompt: str,
    max_tokens: int,
    source_line: int,
) -> dict[str, Any]:
    """Request a rollout, retrying transient server or transport failures."""
    for attempt in range(1, args.max_retries + 1):
        try:
            return post_completion(
                endpoint,
                model=args.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                end_marker=args.end_marker,
                timeout=args.request_timeout,
            )
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == args.max_retries:
                raise RuntimeError(
                    f"source line {source_line} failed after {attempt} attempts: {exc}"
                ) from exc
            logger.warning(
                "source line %d attempt %d/%d failed: %s",
                source_line,
                attempt,
                args.max_retries,
                exc,
            )
            time.sleep(args.retry_delay)
    raise AssertionError("retry loop completed without returning or raising")


def run(args: argparse.Namespace) -> None:
    """Generate target responses and write resumable conversations JSONL."""
    validate_args(args)
    tokenizer = load_tokenizer(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    completed_rows = load_completed_rows(args.output_path) if args.resume else set()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = args.server_url.rstrip("/") + "/v1/completions"

    selected = 0
    written = 0
    skipped_completed = 0
    skipped_overlong = 0
    dropped_truncated = 0
    started = time.perf_counter()
    mode = "a" if args.resume else "x"

    def consume(
        future: Future[dict[str, Any]],
        job: RolloutJob,
        output_handle,
    ) -> None:
        nonlocal written, dropped_truncated
        result = future.result()
        finish_reason = result["finish_reason"]
        if finish_reason == "length" and args.drop_truncated:
            dropped_truncated += 1
            return

        target_rollout = {
            "finish_reason": finish_reason,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "enable_thinking": not args.disable_thinking,
        }
        assistant_content = result["output_text"]
        if args.end_marker and assistant_content.endswith(args.end_marker):
            assistant_content = assistant_content[: -len(args.end_marker)]
        payload: dict[str, Any] = {
            "conversations": [
                dict(message) for message in job.conversations_prefix
            ]
            + [{"role": "assistant", "content": assistant_content}],
            "source_line_number": job.source_line,
            "target_rollout": target_rollout,
        }
        output_handle.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
        output_handle.flush()
        written += 1
        if args.log_interval > 0 and written % args.log_interval == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            logger.info(
                "written=%d skipped_completed=%d skipped_overlong=%d "
                "dropped_truncated=%d rows/s=%.3f",
                written,
                skipped_completed,
                skipped_overlong,
                dropped_truncated,
                written / elapsed,
            )

    with (
        args.data_path.open(encoding="utf-8") as input_handle,
        args.output_path.open(mode, encoding="utf-8") as output_handle,
        ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        pending: deque[tuple[Future[dict[str, Any]], RolloutJob]] = deque()
        for source_line, line in enumerate(input_handle, 1):
            if source_line <= args.start_index or not line.strip():
                continue
            if args.limit and selected >= args.limit:
                break
            selected += 1
            if source_line in completed_rows:
                skipped_completed += 1
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at source line {source_line}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"source line {source_line} is not a JSON object")
            prompt, conversations_prefix = prompt_from_conversations(
                tokenizer,
                row.get("conversations"),
                disable_thinking=args.disable_thinking,
            )
            prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            available_tokens = args.max_length - prompt_token_count
            if available_tokens <= 0:
                skipped_overlong += 1
                logger.warning(
                    "skip source line %d: prompt has %d tokens, "
                    "reaches --max-length=%d",
                    source_line,
                    prompt_token_count,
                    args.max_length,
                )
                continue
            request_max_tokens = min(args.max_tokens, available_tokens)
            job = RolloutJob(
                source_line=source_line,
                prompt=prompt,
                max_tokens=request_max_tokens,
                conversations_prefix=conversations_prefix,
            )
            future = executor.submit(
                request_with_retries,
                endpoint,
                args=args,
                prompt=job.prompt,
                max_tokens=job.max_tokens,
                source_line=job.source_line,
            )
            pending.append((future, job))
            if len(pending) >= args.concurrency:
                oldest_future, oldest_job = pending.popleft()
                consume(oldest_future, oldest_job, output_handle)

        while pending:
            future, job = pending.popleft()
            consume(future, job, output_handle)

    logger.info("Selected source rows: %d", selected)
    logger.info("Written rollouts: %d", written)
    logger.info("Already completed: %d", skipped_completed)
    logger.info("Skipped overlong prompts: %d", skipped_overlong)
    logger.info("Dropped truncated rollouts: %d", dropped_truncated)
    logger.info("Saved target rollouts to %s", args.output_path)


def main() -> None:
    """Run target rollout generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
