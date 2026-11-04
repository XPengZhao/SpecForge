#!/usr/bin/env python3
"""Measure two-turn MT-Bench generation and speculative acceptance on vLLM.

Requires aiohttp. Counters are server-wide; use an otherwise idle service.
This measures generation/acceptance, not MT-Bench judge scores.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import aiohttp

if __package__:
    from .eval_humaneval_vllm import snapshot, summarize, write_json
    from .stat_vllm_acceptance import subtract
else:
    from eval_humaneval_vllm import snapshot, summarize, write_json
    from stat_vllm_acceptance import subtract

DATA_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--data-path", type=Path, help="Local MT-Bench question_id/category/turns JSONL"
    )
    parser.add_argument("--num-questions", type=int, default=80)
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
    parser.add_argument("--first-turn-responses", type=Path,
                        help="Reuse a previous run's turn1.responses.jsonl as turn-2 history")
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
        text = args.data_path.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            text = response.read().decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    tasks, ids = [], set()
    for row in rows[:args.num_questions]:
        qid = row.get("question_id")
        turns = row.get("turns")
        if type(qid) not in (int, str) or qid in ids:
            raise ValueError("Missing or duplicate question_id")
        if (not isinstance(turns, list) or len(turns) != 2
                or not all(isinstance(t, str) and t.strip() for t in turns)):
            raise ValueError(f"Question {qid} requires two nonempty turns")
        ids.add(qid)
        tasks.append(dict(question_id=qid, category=row.get("category"), turns=turns))
    if not tasks:
        raise ValueError("No questions to evaluate")
    return tasks


def load_history(path, tasks):
    if path is None:
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    history = {}
    for row in rows:
        qid = row["question_id"]
        if qid in history:
            raise ValueError(f"Duplicate history question: {qid}")
        history[qid] = row
    for task in tasks:
        row = history.get(task["question_id"], {})
        if ("error" in row or not isinstance(row.get("content"), str)
                or not row["content"].strip() or row.get("turns") != task["turns"]):
            raise ValueError(f"Missing, failed, or mismatched history: {task['question_id']}")
    return history


def request_payload(args, task, index, turn, history):
    messages = [{"role": "user", "content": task["turns"][0]}]
    if turn == 2:
        messages += [{"role": "assistant", "content": history[task["question_id"]]["content"]},
                     {"role": "user", "content": task["turns"][1]}]
    return dict(model=args.model, messages=messages, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
                max_tokens=args.max_tokens, seed=args.seed + 2 * index + turn - 1,
                n=1, stream=False, chat_template_kwargs=args.chat_template_kwargs)


async def generate(session, url, args, tasks, turn, history):
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def one(index, task):
        async with semaphore:
            started = time.perf_counter()
            payload = request_payload(args, task, index, turn, history)
            result = dict(index=index, **task, turn=turn, seed=payload["seed"])
            try:
                async with session.post(
                    url + "/v1/chat/completions", json=payload,
                    timeout=aiohttp.ClientTimeout(total=args.request_timeout_seconds),
                ) as response:
                    text = await response.text()
                    result.update(http_status=response.status, raw_response=text)
                    response.raise_for_status()
                body = json.loads(text)
                choice = body["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Response has no nonempty assistant content")
                result.update(content=content, finish_reason=choice.get("finish_reason"),
                              usage=body.get("usage") or {})
            except Exception as exc:
                # Do not retry requests that may still be generating on the server.
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["latency_seconds"] = time.perf_counter() - started
            return result

    started = time.perf_counter()
    results = []
    with (args.output_dir / f"turn{turn}.responses.jsonl").open("x") as stream:
        pending = [asyncio.create_task(one(i, task)) for i, task in enumerate(tasks)]
        for future in asyncio.as_completed(pending):
            result = await future
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(result)
            print(f"Turn {turn}: {len(results)}/{len(tasks)} completed"
                  + (" (ERROR)" if "error" in result else ""), flush=True)
    return sorted(results, key=lambda row: row["index"]), time.perf_counter() - started


def generation_summary(results, elapsed):
    report = summarize(results, elapsed)
    del report["pass_at_1"], report["grading"]
    report["num_requests"] = report.pop("num_questions")
    return report


async def evaluate(args, tasks):
    history = load_history(args.first_turn_responses, tasks)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(tasks=tasks, data_url=DATA_URL if not args.data_path else None,
                  seed_policy="seed + 2 * question index + turn - 1",
                  history_mode="fixed" if history is not None else "generated")
    write_json(out / "config.json", config)
    url = f"{args.host.rstrip('/')}:{args.port}"
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    report = dict(turns={}, acceptance_scope="Server-wide counters; otherwise idle service required",
                  grading="Not performed", valid_measurement=True)
    all_results, total_elapsed, first, after = [], 0.0, None, None
    async with aiohttp.ClientSession(
        headers=headers, connector=aiohttp.TCPConnector(limit=args.max_concurrency)
    ) as session:
        before = None
        if not args.skip_acceptance:
            before = first = await snapshot(session, url)
            write_json(out / "before.json", before)
        for turn in (1, 2):
            results, elapsed = await generate(session, url, args, tasks, turn, history)
            all_results.extend(results)
            total_elapsed += elapsed
            stats = generation_summary(results, elapsed)
            stats["acceptance"] = None
            if before is not None:
                await asyncio.sleep(args.metrics_wait_seconds)
                try:
                    after = await snapshot(session, url)
                    write_json(out / f"turn{turn}.after.json", after)
                    stats["acceptance"] = subtract(before, after)
                    before = after
                except Exception as exc:
                    stats["metrics_error"] = f"{type(exc).__name__}: {exc}"
                    report["valid_measurement"] = False
            report["turns"][str(turn)] = stats
            if stats["failed_requests"]:
                report["valid_measurement"] = False
            if not report["valid_measurement"]:
                break  # Never mix incomplete or timed-out turn-1 requests into turn 2.
            if turn == 1 and history is None:
                history = {row["question_id"]: row for row in results}
    report.update(generation_summary(all_results, total_elapsed))
    report["num_questions"] = len(tasks)
    report["acceptance"] = None
    if first is not None and after is not None and report["valid_measurement"]:
        report["acceptance"] = subtract(first, after)
        write_json(out / "acceptance.json", report["acceptance"])
    write_json(out / "mtbench.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    for label, stats in [*report["turns"].items(), ("overall", report)]:
        if stats.get("acceptance"):
            a = stats["acceptance"]
            print(f"{label}: MAL={a['mal']:.4f}, acceptance={a['acceptance_rate']:.2%}")
            print("Position  Accepted  Continuous rate  Conditional rate")
            for row in a["positions"]:
                conditional = (
                    f"{row['conditional_rate']:.2%}"
                    if row["conditional_rate"] is not None
                    else "N/A"
                )
                print(
                    f"{row['position']:8d}  {row['accepted']:8d}  "
                    f"{row['continuous_rate']:15.2%}  {conditional:>16}"
                )
    print(f"Results saved to {out}", flush=True)
    return 0 if report["valid_measurement"] else 1


def main(argv=None):
    args = parse_args(argv)
    return asyncio.run(evaluate(args, load_tasks(args)))


if __name__ == "__main__":
    raise SystemExit(main())
