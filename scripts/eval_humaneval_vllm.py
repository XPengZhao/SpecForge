#!/usr/bin/env python3
"""Generate HumanEval solutions concurrently through an existing vLLM service.

Dependencies: aiohttp, evalplus. Does not execute generated code. Grade the
saved samples.jsonl separately with evalplus.evaluate. Acceptance counters are
server-wide: use an otherwise idle service for the complete measurement window.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import aiohttp

if __package__:
    from .stat_vllm_acceptance import parse_metrics, subtract
else:
    from stat_vllm_acceptance import parse_metrics, subtract


INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--data-path", type=Path, help="Local task_id/prompt/entry_point JSONL"
    )
    parser.add_argument("--num-questions", type=int, default=164)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
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


def load_tasks(args):
    if args.data_path:
        rows = [
            json.loads(line)
            for line in args.data_path.read_text().splitlines()
            if line.strip()
        ]
    else:
        from evalplus.data import get_human_eval_plus

        rows = list(get_human_eval_plus().values())
        if len(rows) != 164:
            raise ValueError(f"Expected 164 HumanEval tasks, got {len(rows)}")
    rows = rows[: args.num_questions]
    ids = set()
    for row in rows:
        for key in ("task_id", "prompt", "entry_point"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Task requires a nonempty {key}")
        if row["task_id"] in ids:
            raise ValueError(f"Duplicate task ID: {row['task_id']}")
        ids.add(row["task_id"])
    if not rows:
        raise ValueError("No tasks to evaluate")
    # Do not send or copy canonical solutions/test cases into the model prompt.
    return [
        {key: row[key] for key in ("task_id", "prompt", "entry_point")} for row in rows
    ]


def request_payload(args, task, index):
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    INSTRUCTION + "\n```python\n" + task["prompt"].strip() + "\n```"
                ),
            }
        ],
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


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def snapshot(session, url):
    async with session.get(
        url + "/metrics", timeout=aiohttp.ClientTimeout(total=30)
    ) as response:
        response.raise_for_status()
        return {"url": url, "series": parse_metrics(await response.text())}


async def generate(session, url, args, tasks, raw_path):
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def one(index, task):
        async with semaphore:
            started = time.perf_counter()
            result = {
                "index": index,
                "task_id": task["task_id"],
                "seed": args.seed + index,
            }
            try:
                async with session.post(
                    url + "/v1/chat/completions",
                    json=request_payload(args, task, index),
                    timeout=aiohttp.ClientTimeout(total=args.request_timeout_seconds),
                ) as response:
                    text = await response.text()
                    result["http_status"] = response.status
                    result["raw_response"] = text
                    response.raise_for_status()
                body = json.loads(text)
                choice = body["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Response has no nonempty assistant content")
                result.update(
                    content=content,
                    finish_reason=choice.get("finish_reason"),
                    usage=body.get("usage") or {},
                )
            except Exception as exc:
                # No automatic retry: a timed-out request may still generate tokens
                # server-side and contaminate both throughput and acceptance counters.
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["latency_seconds"] = time.perf_counter() - started
            return result

    started = time.perf_counter()
    pending = [asyncio.create_task(one(i, row)) for i, row in enumerate(tasks)]
    results = []
    with raw_path.open("x") as stream:
        for future in asyncio.as_completed(pending):
            result = await future
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(result)
            print(
                f"Completed {len(results)}/{len(tasks)}: {result['task_id']}"
                + (" (ERROR)" if "error" in result else ""),
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return sorted(results, key=lambda row: row["index"]), elapsed


def summarize(results, elapsed):
    success = [row for row in results if "error" not in row]
    counts = [row.get("usage", {}).get("completion_tokens") for row in success]
    known = bool(success) and all(type(n) is int and n >= 0 for n in counts)
    total = sum(counts) if known else None
    return {
        "num_questions": len(results),
        "successful_requests": len(success),
        "failed_requests": len(results) - len(success),
        "truncated_responses": sum(
            row.get("finish_reason") == "length" for row in success
        ),
        "generation_seconds": elapsed,
        "questions_per_second": len(success) / elapsed,
        "total_output_tokens": total,
        "output_tokens_per_second": total / elapsed if total is not None else None,
        "mean_output_length": total / len(success) if total is not None else None,
        "pass_at_1": None,
        "grading": "Not executed; run evalplus.evaluate separately on samples.jsonl",
    }


async def evaluate(args, tasks, sanitize):
    url = f"{args.host.rstrip('/')}:{args.port}"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        instruction=INSTRUCTION,
        endpoint=url + "/v1/chat/completions",
        seed_policy="seed + task index",
        tasks=tasks,
    )
    for package in ("evalplus", "aiohttp"):
        try:
            config[package + "_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config[package + "_version"] = None
    write_json(out / "config.json", config)
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
            session, url, args, tasks, out / "responses.jsonl"
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
    # Export one sample per task, including empty solutions for failed requests.
    # Thus downstream pass@1 never silently drops unsuccessful generations.
    sanitization_errors = []
    empty_solutions = 0
    with (out / "samples.jsonl").open("x") as stream:
        for task, result in zip(tasks, results):
            solution = ""
            if "error" not in result:
                try:
                    solution = sanitize(
                        result["content"], entrypoint=task["entry_point"]
                    )
                except Exception as exc:
                    sanitization_errors.append(
                        {"task_id": task["task_id"], "error": str(exc)}
                    )
            empty_solutions += not bool(solution.strip())
            stream.write(
                json.dumps(
                    {"task_id": task["task_id"], "solution": solution},
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary["sanitization_errors"] = sanitization_errors
    summary["empty_solutions"] = empty_solutions
    summary["valid_measurement"] = not (
        summary["failed_requests"] or metrics_error or sanitization_errors
    )
    write_json(out / "humaneval.json", summary)
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
    from evalplus.sanitize import sanitize

    tasks = load_tasks(args)
    return asyncio.run(evaluate(args, tasks, sanitize))


if __name__ == "__main__":
    raise SystemExit(main())
