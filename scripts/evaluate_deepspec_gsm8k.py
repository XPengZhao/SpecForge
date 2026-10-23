"""Send DeepSpec GSM8K prompts to vLLM with the public reference recipe.

This aligns inputs and request parameters, not the RNG implementation or
terminal-round accounting of the two inference engines.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request


def select_samples(path: Path, count: int, seed: int) -> list[tuple[int, dict]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns or not all(
            isinstance(turn, str) for turn in turns
        ):
            raise ValueError(f"{path}:{line_number}: expected nonempty string turns")
        rows.append((line_number, row))
    if count <= 0 or not rows:
        raise ValueError("num-questions must be positive and dataset must not be empty")
    if len(rows) > count:
        random.Random(seed).shuffle(rows)
        rows = rows[:count]
    return rows


def build_request(tokenizer, row, *, model, seed, max_tokens, stop_token_ids):
    # Match DeepSpec's render_chat_messages -> tokenizer(add_special_tokens=False).
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["turns"][0]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    body = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": seed,
        "n": 1,
        "ignore_eos": False,
    }
    if stop_token_ids is not None:
        body["stop_token_ids"] = stop_token_ids
    return text, body


def send_request(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    record = {"request": body}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not result.get("choices") or "text" not in result["choices"][0]:
            raise ValueError(f"Malformed completions response: {result}")
        record["response"] = result
    except urllib.error.HTTPError as exc:
        record["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - start
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    parser.add_argument("--save-results", type=Path, required=True)
    args = parser.parse_args()
    if args.max_concurrency < 1 or args.max_tokens < 1:
        parser.error("max-concurrency and max-tokens must be positive")

    from transformers import AutoTokenizer
    from tqdm import tqdm

    selected = select_samples(args.dataset_path, args.num_questions, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    generation_path = Path(args.tokenizer) / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    stop_ids = generation_config.get("eos_token_id", tokenizer.eos_token_id)
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    tasks = []
    for index, (source_line, row) in enumerate(tqdm(selected, desc="Preparing prompts")):
        text, body = build_request(
            tokenizer, row, model=args.model, seed=args.seed + index,
            max_tokens=args.max_tokens, stop_token_ids=stop_ids,
        )
        tasks.append((index, source_line, text, body))

    output = args.save_results
    output.parent.mkdir(parents=True, exist_ok=True)
    details_path = output.with_suffix(".requests.jsonl")
    manifest = {
        "protocol": "deepspec-gsm8k",
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset_path.read_bytes()).hexdigest(),
        "selected_source_lines": [line for line, _ in selected],
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "num_questions": len(tasks),
        "max_tokens": args.max_tokens,
        "max_concurrency": args.max_concurrency,
        "details_file": str(details_path),
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

    def run(task):
        index, source_line, text, body = task
        record = send_request(
            f"{args.host.rstrip('/')}:{args.port}", body, args.request_timeout_seconds
        )
        return {"index": index, "source_line": source_line, "rendered_prompt": text, **record}

    start = time.monotonic()
    errors = tokens = finished_by_length = 0
    with details_path.open("x") as stream, ThreadPoolExecutor(args.max_concurrency) as pool:
        for record in tqdm(pool.map(run, tasks), total=len(tasks), desc="Evaluating"):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            errors += int("error" in record)
            response = record.get("response", {})
            tokens += response.get("usage", {}).get("completion_tokens", 0)
            choices = response.get("choices", [])
            finished_by_length += int(bool(choices) and choices[0].get("finish_reason") == "length")
    manifest.update(
        request_errors=errors, completion_tokens=tokens,
        finished_by_length=finished_by_length, elapsed_seconds=time.monotonic() - start,
    )
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if errors:
        raise RuntimeError(f"{errors} requests failed; inspect {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
