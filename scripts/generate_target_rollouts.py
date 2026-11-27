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
import hashlib
import http.client
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from queue import Empty, Queue
from dataclasses import dataclass
from datetime import timedelta
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
    parser.add_argument(
        "--seed-offset", type=int, help="Sampling seed = offset + input line number"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1,
                        help="For multiple rounds, --output-path is a directory of round-NN.jsonl files")
    parser.add_argument("--first-round", type=int, help="First global round number (1-based); write numbered files")
    parser.add_argument("--seed-stride", type=int, help="Physical input line count for round seeds")
    parser.add_argument("--dp-ranks", type=int, default=1)
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
    if args.rounds < 1 or args.dp_ranks < 1:
        raise ValueError("rounds and dp-ranks must be positive")
    if args.first_round is not None and args.first_round < 1:
        raise ValueError("first-round must be positive")
    if (args.rounds > 1 or args.first_round is not None) and args.seed_offset is None:
        raise ValueError("numbered rounds require seed-offset")
    if args.rounds == 1 and args.first_round is None and args.output_path.exists() and not args.resume:
        raise ValueError(f"{args.output_path} exists; pass --resume to append missing rows")


def prepare_manifest(args, paths, numbered):
    """Bind resumable output to the input bytes and generation settings."""
    state_dir = (args.output_path / ".rollout-state" if numbered else
                 args.output_path.with_name(f".{args.output_path.name}.rollout-state"))
    manifest_path = state_dir / "manifest.json"
    digest = hashlib.sha256()
    selected_count = 0
    physical_lines = 0
    with args.data_path.open("rb") as source:
        for line in source:
            digest.update(line)
            physical_lines += 1
            if physical_lines > args.start_index and (not args.limit or selected_count < args.limit) and line.strip():
                selected_count += 1
    if numbered:
        if args.seed_stride is None:
            args.seed_stride = physical_lines
        if args.seed_stride < physical_lines:
            raise ValueError("seed-stride must cover all physical input lines")
    manifest = {
        "version": 1, "input_sha256": digest.hexdigest(), "input_lines": physical_lines,
        **{key: getattr(args, key) for key in (
            "model", "tokenizer_path", "seed_offset", "seed_stride", "temperature",
            "top_p", "max_length", "max_tokens", "end_marker", "disable_thinking",
            "drop_truncated", "start_index", "limit",
        )},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        changed = [key for key in manifest if old.get(key) != manifest[key]]
        if changed:
            raise ValueError(f"rollout manifest mismatch ({', '.join(changed)}): {manifest_path}")
    else:
        previous = args.output_path.glob("round-*.jsonl") if numbered else paths
        if any(p.exists() and p.stat().st_size for p in previous):
            raise ValueError("existing rollout has no manifest; use a new output path")
        state_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=state_dir, delete=False) as temp:
            json.dump(manifest, temp, indent=2)
        os.replace(temp.name, manifest_path)
    return state_dir / "skipped-overlong.jsonl", selected_count


def load_completed_rows(output_path: Path) -> set[int]:
    """Return source line numbers already present in a rollout file."""
    if not output_path.exists():
        return set()
    completed: set[int] = set()
    with output_path.open("rb+") as handle:
        for output_line, line in enumerate(handle, 1):
            try:
                if not line.strip() and line.endswith(b"\n"):
                    continue
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if not line.endswith(b"\n"):
                    # A killed writer can leave part of its final UTF-8 record.
                    handle.truncate(handle.tell() - len(line))
                    logger.warning("Removed incomplete final record in %s", output_path)
                    break
                raise ValueError(
                    f"invalid existing output JSON at line {output_line}: {exc}"
                ) from exc
            source_line = row.get("source_line_number")
            if not isinstance(source_line, int):
                raise ValueError(
                    f"existing output line {output_line} has no integer source_line_number"
                )
            if source_line in completed:
                raise ValueError(f"duplicate source line {source_line} in {output_path}")
            completed.add(source_line)
            if not line.endswith(b"\n"):
                handle.seek(0, 2)
                handle.write(b"\n")
                break
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
        kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(
        prefix,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("apply_chat_template returned an empty generation prompt")
    return prompt, tuple(prefix)


def read_stream(response, on_ready):
    pieces = []
    usage = {}
    finish = None
    notified = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"stream error: {event['error']}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            text = choice.get("text", "")
            pieces.append(text)
            finish = choice.get("finish_reason") or finish
            if not notified and (text or finish):
                notified = True
                on_ready()
    if finish is None or "completion_tokens" not in usage:
        raise RuntimeError("incomplete rollout stream")
    return {"choices": [{"text": "".join(pieces), "finish_reason": finish}], "usage": usage}


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
    seed: int | None = None,
    rank=None,
    on_ready=None,
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
    if seed is not None:
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    if on_ready is not None:
        payload.update(stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = read_stream(response, on_ready) if on_ready else json.load(response)
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
    round_index=0,
    rank=None,
    on_ready=None,
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
                seed=None if args.seed_offset is None else args.seed_offset + source_line + round_index * (args.seed_stride or 0),
                rank=rank, on_ready=on_ready,
            )
        except (RuntimeError, urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, json.JSONDecodeError) as exc:
            if attempt == args.max_retries:
                raise RuntimeError(
                    f"round {round_index + 1} source line {source_line} failed after {attempt} attempts: {exc}"
                ) from exc
            logger.warning(
                "round %d source line %d attempt %d/%d failed: %s",
                round_index + 1,
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
    first_round = args.first_round or 1
    paths = ([args.output_path] if args.rounds == 1 and args.first_round is None else
             [args.output_path / f"round-{first_round+i:02d}.jsonl" for i in range(args.rounds)])
    if not args.resume and any(p.exists() for p in paths):
        raise ValueError("output exists; use --resume")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    skipped_path, selected_count = prepare_manifest(args, paths, args.rounds > 1 or args.first_round is not None)
    overlong = load_completed_rows(skipped_path)
    selected_lines = set()
    dropped = [set() for _ in paths]
    completed = [load_completed_rows(p) if args.resume else set() for p in paths]
    endpoint = args.server_url.rstrip("/") + "/v1/completions"

    selected = 0
    written = 0
    skipped_completed = 0
    skipped_overlong = 0
    dropped_truncated = 0
    started = time.perf_counter()
    mode = "a" if args.resume else "x"

    resumed = sum(map(len, completed))

    def log_progress():
        processed = written + dropped_truncated
        done = resumed + processed
        total = (selected_count - len(overlong)) * args.rounds
        elapsed = max(time.perf_counter() - started, 1e-6)
        remaining = max(total - done, 0)
        eta = "pending"
        if remaining == 0:
            eta = "0:00:00"
        elif processed:
            eta = str(timedelta(seconds=round(remaining * elapsed / processed)))
        logger.info(
            "progress=%d/%d (%.2f%%) resumed=%d written=%d skipped_completed=%d "
            "skipped_overlong=%d dropped_truncated=%d rows/s=%.3f ETA=%s",
            done, total, 100 * done / total if total else 100, resumed, written,
            skipped_completed, skipped_overlong, dropped_truncated, written / elapsed, eta,
        )

    log_progress()

    def consume(
        future: Future[dict[str, Any]],
        job: RolloutJob,
        index,
    ) -> None:
        nonlocal written, dropped_truncated
        result = future.result()
        finish_reason = result["finish_reason"]
        if finish_reason == "length" and args.drop_truncated:
            dropped[index].add(job.source_line)
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
        output_handle = outputs[index]
        output_handle.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
        output_handle.flush()
        completed[index].add(job.source_line)
        written += 1

    def jobs(input_handle):
        nonlocal selected, skipped_completed, skipped_overlong
        for source_line, line in enumerate(input_handle, 1):
            if source_line <= args.start_index or not line.strip():
                continue
            if args.limit and selected >= args.limit:
                break
            selected += 1
            selected_lines.add(source_line)
            if source_line in overlong:
                skipped_overlong += 1
                continue
            missing = [i for i, rows in enumerate(completed) if source_line not in rows]
            if not missing:
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
                skipped_file.write(json.dumps({"source_line_number": source_line,
                                               "prompt_tokens": prompt_token_count}) + "\n")
                skipped_file.flush()
                overlong.add(source_line)
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
            yield job, missing

    with ExitStack() as stack:
        inputs = stack.enter_context(args.data_path.open(encoding="utf-8"))
        skipped_file = stack.enter_context(skipped_path.open("a", encoding="utf-8"))
        outputs = [stack.enter_context(p.open(mode, encoding="utf-8")) for p in paths]
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=args.concurrency))
        events = Queue()
        ready = deque()
        pending = {}
        loads = [0] * args.dp_ranks
        source = iter(jobs(inputs))
        exhausted = False

        def submit(job, index, rank, followers=()):
            # Retries may deliver a second first-token notification.
            def warmed():
                events.put(("ready", (job, rank, followers)))

            future = executor.submit(
                request_with_retries, endpoint, args=args, prompt=job.prompt,
                max_tokens=job.max_tokens, source_line=job.source_line,
                round_index=first_round - 1 + index, rank=rank if args.dp_ranks > 1 else None,
                on_ready=warmed if followers else None,
            )
            pending[future] = (job, index, rank)
            future.add_done_callback(lambda f: events.put(("done", f)))

        # Charge a prompt's remaining rounds to its rank when assigning it.
        # HTTP requests, including open streams, count against one global cap.
        warming = set()
        while pending or ready or not exhausted:
            try:
                event, value = events.get_nowait()
            except Empty:
                if len(pending) < args.concurrency:
                    if ready:
                        job, index, rank = ready.popleft()
                        submit(job, index, rank)
                        continue
                    if not exhausted:
                        item = next(source, None)
                        if item is not None:
                            job, missing = item
                            rank = min(range(args.dp_ranks), key=loads.__getitem__)
                            loads[rank] += len(missing)
                            if len(missing) > 1:
                                warming.add(job.source_line)
                            submit(job, missing[0], rank, missing[1:])
                            continue
                        exhausted = True
                if not pending:
                    break
                event, value = events.get()
            if event == "ready":
                job, rank, followers = value
                if job.source_line in warming:
                    warming.remove(job.source_line)
                    ready.extend((job, i, rank) for i in followers)
            else:
                job, index, rank = pending.pop(value)
                consume(value, job, index)
                loads[rank] -= 1
                if args.log_interval > 0 and (written + dropped_truncated) % args.log_interval == 0:
                    log_progress()

    if not selected_lines:
        raise ValueError("input selection has no nonempty rows")
    if not overlong.issubset(selected_lines):
        raise ValueError("skipped-overlong contains source lines outside the input selection")
    eligible = selected_lines - overlong
    for index, rows in enumerate(completed):
        expected = eligible - dropped[index] if dropped[index] else eligible
        if rows != expected:
            raise ValueError(f"{paths[index]}: {len(expected - rows)} missing and "
                             f"{len(rows - expected)} unexpected source lines")
        logger.info("%s: verified %d eligible responses", paths[index].name, len(rows))
    log_progress()
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
