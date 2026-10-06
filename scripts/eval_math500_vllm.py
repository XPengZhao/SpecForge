#!/usr/bin/env python3
"""Evaluate MATH-500 through a running vLLM chat endpoint.

Install client dependencies:
    pip install aiohttp datasets 'math-verify[antlr4_13_2]==0.9.0'

Uses a zero-shot prompt and Math-Verify. Saves raw responses before grading.
Speculative counters are server-wide differences; use an otherwise idle server.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "HuggingFaceH4/MATH-500"
INSTRUCTION = (
    "Please reason step by step, and put your final answer within "
    "\\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen38-flash-next")
    parser.add_argument("--data-path", type=Path, help="JSONL with problem and answer.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--metrics-wait-seconds", type=float, default=0.0,
        help="Wait after generation before reading speculative counters (0-60 seconds).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.concurrency, args.max_tokens) < 1:
        parser.error("limit, concurrency and max-tokens must be positive")
    if args.request_timeout <= 0 or args.temperature < 0:
        parser.error("request-timeout must be positive; temperature must be >= 0")
    if not 0 <= args.metrics_wait_seconds <= 60:
        parser.error("metrics-wait-seconds must be in [0, 60]")
    if not 0 < args.top_p <= 1 or not 0 <= args.min_p <= 1:
        parser.error("top-p must be in (0, 1]; min-p must be in [0, 1]")
    args.responses = args.output.with_suffix(".responses.jsonl")
    for path in (args.output, args.responses):
        if path.exists():
            parser.error(f"output already exists: {path}; choose another filename")
    return args


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.data_path is not None:
        with args.data_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        from datasets import load_dataset

        rows = list(load_dataset(DATASET, split="test"))
        if len(rows) != 500:
            raise ValueError(f"Expected 500 MATH-500 rows, found {len(rows)}")
    rows = rows[: args.limit]
    if not rows:
        raise ValueError("No questions to evaluate")
    for index, row in enumerate(rows):
        for field in ("problem", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Row {index} needs a nonempty {field!r} string")
    return rows


def read_spec_metrics(server_url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(
            server_url.rstrip("/") + "/metrics", timeout=15
        ) as response:
            body = response.read().decode()
    except Exception as exc:
        print(f"Speculative metrics unavailable: {exc}", flush=True)
        return None
    totals = {"drafts": 0, "drafted": 0, "accepted": 0, "positions": {}}
    fields = {
        "num_drafts": "drafts",
        "num_draft_tokens": "drafted",
        "num_accepted_tokens": "accepted",
    }
    found = False
    for line in body.splitlines():
        match = re.match(
            r"^(vllm:spec_decode_[a-z_]+)_total(\{.*\})?\s+"
            r"([0-9.eE+-]+)(?:\s|$)", line
        )
        if not match:
            continue
        name, labels, raw = match.groups()
        value = int(float(raw))
        for suffix, field in fields.items():
            if name.endswith("_" + suffix):
                totals[field] += value
                found = True
        if name.endswith("_num_accepted_tokens_per_pos"):
            position = re.search(r'\bposition="(\d+)"', labels or "")
            if position:
                key = int(position.group(1))
                totals["positions"][key] = totals["positions"].get(key, 0) + value
                found = True
    return totals if found else None


def acceptance_summary(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    delta = {key: after[key] - before[key] for key in ("drafts", "drafted", "accepted")}
    positions = sorted(set(before["positions"]) | set(after["positions"]))
    counts = [
        after["positions"].get(i, 0) - before["positions"].get(i, 0)
        for i in positions
    ]
    if min([*delta.values(), *counts], default=0) < 0:
        raise ValueError("Server counters reset during evaluation")
    if delta["drafts"] == 0:
        return None
    if counts and sum(counts) != delta["accepted"]:
        raise ValueError("Per-position counts do not sum to accepted tokens")
    previous = delta["drafts"]
    conditional = []
    for count in counts:
        if count > previous:
            raise ValueError("Continuous-prefix counts must be nonincreasing")
        conditional.append(count / previous if previous else None)
        previous = count
    return {
        "mal": 1 + delta["accepted"] / delta["drafts"],
        "draft_token_acceptance_rate": (
            delta["accepted"] / delta["drafted"] if delta["drafted"] else None
        ),
        "position_labels": positions,
        "continuous_acceptance": [count / delta["drafts"] for count in counts],
        "conditional_acceptance": conditional,
        "counter_deltas": {**delta, "accepted_per_position": counts},
        "scope": "all requests to this server during evaluation",
    }


async def generate(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    import aiohttp

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    headers = {}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    endpoint = args.server_url.rstrip("/") + "/v1/chat/completions"

    async def request(session, index, row):
        async with semaphore:
            prompt = row["problem"] + "\n" + INSTRUCTION
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
            }
            record = {
                "index": index,
                "unique_id": row.get("unique_id", index),
                "problem": row["problem"],
                "gold_answer": row["answer"],
                "subject": row.get("subject"),
                "level": row.get("level"),
                "prompt": prompt,
            }
            started = time.perf_counter()
            try:
                async with session.post(endpoint, json=payload) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:800]}")
                    result = json.loads(body)
                choice = result["choices"][0]
                message = choice["message"]
                record.update(
                    content=message.get("content") or "",
                    reasoning_content=(
                        message.get("reasoning_content") or message.get("reasoning") or ""
                    ),
                    finish_reason=choice.get("finish_reason"),
                    usage=result.get("usage") or {},
                    request_error=None,
                )
            except Exception as exc:
                record.update(
                    content="", reasoning_content="", usage={},
                    finish_reason=None, request_error=str(exc),
                )
            record["request_latency_seconds"] = time.perf_counter() - started
            return record

    started = time.perf_counter()
    records = []
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(
        timeout=timeout, headers=headers, connector=connector
    ) as session:
        tasks = [asyncio.create_task(request(session, i, row)) for i, row in enumerate(rows)]
        with args.responses.open("x", encoding="utf-8") as handle:
            for future in asyncio.as_completed(tasks):
                record = await future
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if len(records) % 10 == 0 or len(records) == len(rows):
                    print(f"Generated {len(records)}/{len(rows)}", flush=True)
    return sorted(records, key=lambda row: row["index"]), time.perf_counter() - started


def grade(records, golds, parse, verify, configs):
    for completed, record in enumerate(records, 1):
        # Prefer final content; use reasoning only when the server returns no content.
        output = record["content"] or record["reasoning_content"]
        try:
            prediction = parse(output, extraction_config=configs) if output else []
            record["invalid"] = not bool(prediction)
            record["parsed_answer"] = [str(item) for item in prediction]
            record["correct"] = bool(
                prediction and not record["request_error"]
                and verify(golds[record["index"]], prediction)
            )
        except Exception as exc:
            record.update(correct=False, invalid=True, grading_error=str(exc))
        if completed % 50 == 0 or completed == len(records):
            print(f"Graded {completed}/{len(records)}", flush=True)


def main() -> int:
    args = parse_args()
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    import aiohttp  # noqa: F401 - fail before sending requests if unavailable.

    configs = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    rows = load_rows(args)
    golds = [parse("$\\boxed{" + row["answer"] + "}$",
                   extraction_config=[LatexExtractionConfig()]) for row in rows]
    for index, gold in enumerate(golds):
        if not gold:
            raise ValueError(f"Cannot parse gold answer at row {index}: {rows[index]['answer']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Running MATH-500: {len(rows)} questions, 0-shot, thinking={args.thinking}, "
        f"temperature={args.temperature}, max_tokens={args.max_tokens}", flush=True,
    )
    before = read_spec_metrics(args.server_url)
    records, latency = asyncio.run(generate(args, rows))
    if before is not None and args.metrics_wait_seconds:
        print(f"Waiting {args.metrics_wait_seconds:g} seconds for metrics to update...", flush=True)
        time.sleep(args.metrics_wait_seconds)
    after = read_spec_metrics(args.server_url)
    grade(records, golds, parse, verify, configs)
    try:
        spec = acceptance_summary(before, after)
        metrics_error = None
    except ValueError as exc:
        spec, metrics_error = None, str(exc)
    total = len(records)
    tokens = sum(record["usage"].get("completion_tokens", 0) for record in records)
    errors = sum(bool(record["request_error"]) for record in records)
    grading_errors = sum("grading_error" in record for record in records)
    correct = sum(record["correct"] for record in records)
    summary = {
        "dataset": DATASET if args.data_path is None else str(args.data_path),
        "num_questions": total,
        "num_shots": 0,
        "model": args.model,
        "thinking": args.thinking,
        "sampling": {key: getattr(args, key) for key in (
            "temperature", "top_p", "top_k", "min_p", "seed", "max_tokens",
        )},
        "concurrency": args.concurrency,
        "correct": correct,
        "accuracy": correct / total,
        "invalid_rate": sum(record["invalid"] for record in records) / total,
        "truncation_rate": sum(record["finish_reason"] == "length" for record in records) / total,
        "request_errors": errors,
        "grading_errors": grading_errors,
        "latency_seconds": latency,
        "total_output_tokens": tokens,
        "output_tokens_per_second": tokens / latency,
        "questions_per_second": total / latency,
        "spec_decode": spec,
        "spec_metrics_error": metrics_error,
        "responses_file": str(args.responses),
        "versions": {
            name: importlib.metadata.version(name) for name in ("math-verify", "aiohttp")
        },
        "timestamp": time.time(),
        "records": records,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Accuracy: {correct / total:.3%} ({correct}/{total})")
    print(f"Invalid rate: {summary['invalid_rate']:.3%}")
    print(f"Truncation rate: {summary['truncation_rate']:.3%}")
    print(f"Request errors: {errors}; grading errors: {grading_errors}")
    print(f"Latency: {latency:.3f} s; output tokens/s: {tokens / latency:.3f}")
    if spec is not None:
        print(f"MAL: {spec['mal']:.4f}")
        print("Conditional acceptance:", ", ".join(
            f"{rate:.2%}" if rate is not None else "N/A"
            for rate in spec["conditional_acceptance"]
        ))
    else:
        print(f"Speculative statistics unavailable{': ' + metrics_error if metrics_error else ''}")
    print(f"Results saved to {args.output}")
    return 1 if errors or grading_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
