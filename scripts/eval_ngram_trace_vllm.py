"""Sample 512 unique training-subset prompts and drive instrumented vLLM.

Same evaluation/sampling options as eval_regen_vllm.py. Extra required option:
--trace-prefix ngram-RUN-. This prefix must match VLLM_NGRAM_TRACE_PREFIX.
Server traces are stored server-side, separately from client responses.
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys

if __package__:
    from . import eval_regen_vllm as regen
else:
    import eval_regen_vllm as regen


def select_unique(args):
    rng = random.Random(args.sample_seed)
    seen, selected = set(), []
    digest = hashlib.sha256()
    eligible = skipped = duplicates = 0
    with args.data_path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            try:
                messages = regen.extract_prompt(row)
            except (ValueError, AttributeError):
                skipped += 1
                continue
            key = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            eligible += 1
            task = dict(
                task_id=f"regen-line-{line}",
                source_line=line,
                messages=messages,
                prompt_sha256=key,
                trace_request_id=f"{args.trace_prefix}{line:09d}",
            )
            if len(selected) < args.num_questions:
                selected.append(task)
            else:
                slot = rng.randrange(eligible)
                if slot < args.num_questions:
                    selected[slot] = task
    if eligible < args.num_questions:
        raise ValueError(f"Need {args.num_questions} unique prompts; found {eligible}")
    args.data_sha256 = digest.hexdigest()
    args.eligible_records, args.skipped_records = eligible, skipped
    args.duplicate_prompts = duplicates
    return sorted(selected, key=lambda x: x["source_line"])


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace-prefix", required=True)
    extra, rest = parser.parse_known_args(argv)
    if not extra.trace_prefix.startswith("ngram-") or not extra.trace_prefix.endswith(
        "-"
    ):
        parser.error(
            "trace-prefix must start with ngram- and end with -; use a fresh prefix per run"
        )
    # Last occurrence wins, allowing normal CLI overrides.
    args = regen.parse_args(
        ["--num-questions", "512", "--max-concurrency", "32", *rest]
    )
    args.trace_prefix = extra.trace_prefix
    tasks = select_unique(args)
    original = regen.request_payload

    def payload(args, task, index):
        value = original(args, task, index)
        value["request_id"] = task["trace_request_id"]
        return value

    regen.request_payload = payload
    args.sample_policy = "reservoir over distinct prompt hashes, then source-line order"
    try:
        return asyncio.run(regen.evaluate(args, tasks))
    finally:
        regen.request_payload = original


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
