#!/usr/bin/env python3
"""Evaluate a fixed random sample of training regen prompts on a vLLM service.

Usage: bash scripts/run_regen_with_acceptance.sh [run-name]
Settings can be overridden with EVAL_* environment variables in that wrapper.

Requires aiohttp. Defaults: 256 records, concurrency 32, non-thinking, T=1.
This measures generation acceptance on training-source prompts, not accuracy.
Counters cover the entire server; keep it otherwise idle during evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import random
from pathlib import Path

import aiohttp

if __package__:
    from .eval_humaneval_vllm import generate, snapshot, summarize, write_json
    from .stat_vllm_acceptance import subtract
else:
    from eval_humaneval_vllm import generate, snapshot, summarize, write_json
    from stat_vllm_acceptance import subtract


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Training regen JSONL (conversations or messages).",
    )
    parser.add_argument("--num-questions", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    parser.add_argument("--metrics-wait-seconds", type=float, default=10)
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    parser.add_argument(
        "--skip-acceptance", action="store_true", help="For target-only runs"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.num_questions, args.max_concurrency, args.max_tokens) < 1:
        parser.error("num-questions, max-concurrency and max-tokens must be positive")
    if not all(
        math.isfinite(getattr(args, name))
        for name in (
            "temperature",
            "top_p",
            "min_p",
            "request_timeout_seconds",
            "metrics_wait_seconds",
        )
    ):
        parser.error("numeric settings must be finite")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or not 0 <= args.min_p <= 1:
        parser.error("temperature >= 0, top-p in (0, 1], min-p in [0, 1] required")
    if args.top_k != -1 and args.top_k < 1:
        parser.error("top-k must be -1 or positive")
    if args.request_timeout_seconds <= 0 or not 0 <= args.metrics_wait_seconds <= 60:
        parser.error("timeout must be positive; metrics wait must be in [0, 60]")
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    try:
        args.chat_template_kwargs = json.loads(args.chat_template_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid chat-template-kwargs JSON: {exc}")
    if not isinstance(args.chat_template_kwargs, dict):
        parser.error("chat-template-kwargs must be a JSON object")
    if args.output_dir.exists():
        parser.error("output-dir already exists; use a fresh directory for each run")
    return args


def extract_prompt(row):
    messages = row.get("conversations", row.get("messages"))
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected a conversation with a final assistant response")
    normalized = []
    roles = {"human": "user", "gpt": "assistant"}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Conversation messages must be objects")
        role = message.get("role", message.get("from"))
        if not isinstance(role, str):
            raise ValueError("Message role must be a string")
        role = roles.get(role, role)
        content = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Expected text-only system/user/assistant messages")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant" or normalized[-2]["role"] != "user":
        raise ValueError("Expected a final user/assistant pair")
    prompt = normalized[:-1]
    if not prompt[-1]["content"].strip():
        raise ValueError("Empty final user message")
    return prompt


def load_tasks(args):
    rng = random.Random(args.sample_seed)
    selected = []
    eligible = skipped = 0
    digest = hashlib.sha256()
    with args.data_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
            try:
                if not isinstance(row, dict):
                    raise ValueError("Expected object")
                messages = extract_prompt(row)
            except ValueError:
                skipped += 1
                continue
            eligible += 1
            task = {
                "task_id": f"regen-line-{line_number}",
                "source_line": line_number,
                "messages": messages,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
            }
            if len(selected) < args.num_questions:
                selected.append(task)
            else:
                slot = rng.randrange(eligible)
                if slot < args.num_questions:
                    selected[slot] = task
    if eligible < args.num_questions:
        raise ValueError(
            f"Need {args.num_questions} eligible records; found {eligible}"
        )
    args.data_sha256 = digest.hexdigest()
    args.eligible_records = eligible
    args.skipped_records = skipped
    print(
        f"Selected {len(selected)} / {eligible} records; skipped {skipped}", flush=True
    )
    return sorted(selected, key=lambda task: task["source_line"])


def request_payload(args, task, index):
    return {
        "model": args.model,
        "messages": task["messages"],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed + index,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": args.chat_template_kwargs,
    }


async def evaluate(args, tasks):
    url = f"{args.host.rstrip('/')}:{args.port}"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        prompt_policy="Drop final assistant; preserve preceding conversation",
        sample_algorithm="reservoir sampling over eligible rows, then source-line order",
        endpoint=url + "/v1/chat/completions",
        seed_policy="seed + task index",
        tasks=tasks,
    )
    for package in ("aiohttp",):
        try:
            config[package + "_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config[package + "_version"] = None
    write_json(out / "config.json", config)
    with (out / "prompts.jsonl").open("x", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    metrics_error = None
    acceptance = None
    async with aiohttp.ClientSession(
        headers=headers, connector=aiohttp.TCPConnector(limit=args.max_concurrency)
    ) as session:
        before = None
        if not args.skip_acceptance:
            # Fail before generating if the requested counters are unavailable.
            before = await snapshot(session, url)
            write_json(out / "before.json", before)
        results, elapsed = await generate(
            session,
            url,
            args,
            tasks,
            out / "responses.jsonl",
            payload_fn=request_payload,
        )
        if before is not None:
            await asyncio.sleep(args.metrics_wait_seconds)
            try:
                after = await snapshot(session, url)
                write_json(out / "after.json", after)
                acceptance = subtract(before, after)
                write_json(out / "acceptance.json", acceptance)
            except Exception as exc:
                metrics_error = f"{type(exc).__name__}: {exc}"

    summary = summarize(results, elapsed)
    summary.update(
        acceptance=acceptance,
        metrics_error=metrics_error,
        acceptance_scope="All requests to this service during the snapshot window",
    )
    summary.pop("pass_at_1", None)
    summary.pop("grading", None)
    summary["valid_measurement"] = not (summary["failed_requests"] or metrics_error)
    write_json(out / "regen.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if acceptance:
        print(
            f"MAL: {acceptance['mal']:.4f}\nAcceptance rate: {acceptance['acceptance_rate']:.2%}"
        )
        print("Position  Accepted  Continuous rate  Conditional rate")
        for row in acceptance["positions"]:
            conditional = (
                f"{row['conditional_rate']:.2%}"
                if row["conditional_rate"] is not None
                else "N/A"
            )
            print(
                f"{row['position']:8d}  {row['accepted']:8d}  {row['continuous_rate']:15.2%}  {conditional:>16}"
            )
    print(f"Results saved to {out}", flush=True)
    return 0 if summary["valid_measurement"] else 1


def main(argv=None):
    args = parse_args(argv)
    tasks = load_tasks(args)
    return asyncio.run(evaluate(args, tasks))


if __name__ == "__main__":
    raise SystemExit(main())
