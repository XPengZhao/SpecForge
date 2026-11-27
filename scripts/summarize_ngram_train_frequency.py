"""Join pooled online Engram contributions with exact training predecessor counts."""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def training_lines(path):
    total = path.stat().st_size
    started = last_log = time.monotonic()
    rows = 0
    print(f"Scanning training data: {path} ({total:,} bytes)", file=sys.stderr, flush=True)
    with path.open("rb") as source:
        for line in source:
            if line.strip():
                yield line
                rows += 1
            now = time.monotonic()
            if now - last_log >= 30:
                done = source.tell()
                eta = (total - done) * (now - started) / done
                print(f"Scanned {rows:,} records | {done / total:.1%} of bytes | "
                      f"ETA {eta / 60:.1f} min", file=sys.stderr, flush=True)
                last_log = now
    print(f"Scan complete: {rows:,} records. Aggregating results.", file=sys.stderr, flush=True)


def training_rows(args):
    tokenizer = None
    for line in training_lines(args.train_data):
        row = json.loads(line)
        if "input_ids" in row:
            ids, mask = row["input_ids"], row["loss_mask"]
        elif "prompt_token_ids" in row and "completion_token_ids" in row:
            prompt, answer = row["prompt_token_ids"], row["completion_token_ids"]
            ids, mask = prompt + answer, [0] * len(prompt) + [1] * len(answer)
        else:
            if tokenizer is None:
                from transformers import AutoTokenizer
                from specforge.data.preprocessing import preprocess_conversations
                from specforge.data.template import TEMPLATE_REGISTRY

                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
                template = TEMPLATE_REGISTRY.get(args.chat_template)
            messages = row.get("conversations", row.get("messages"))
            if messages is None:
                raise ValueError("Expected conversations, messages or token IDs")
            if not messages:
                continue
            tools = row.get("tools") or []
            extras = {k: [v] for k, v in row.items()
                      if k not in {"id", "messages", "conversations", "tools"}}
            batch = preprocess_conversations(
                tokenizer, [messages], template, args.max_length,
                tools=[json.loads(tools) if isinstance(tools, str) else tools],
                train_only_last_turn=args.train_only_last_turn, **extras,
            )
            ids = batch["input_ids"][0].flatten().tolist()
            mask = batch["loss_mask"][0].flatten().tolist()
        if len(ids) != len(mask):
            raise ValueError("input_ids and loss_mask must have the same length")
        yield ids[:args.max_length], mask[:args.max_length]


def ratio(a, b):
    return a / b if b else None


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-only-last-turn", action="store_true")
    args = parser.parse_args()
    directory = args.run_dir / "summary"
    grams = {}
    print("Loading reached n-grams from diagnostic results...", file=sys.stderr, flush=True)
    with (directory / "ngram_directions.csv").open() as source:
        for row in csv.DictReader(source):
            if row["cohort"] == "reached":
                gram = tuple(json.loads(row["ngram_token_ids"]))
                if gram in grams or len(gram) not in (2, 3):
                    raise ValueError("Expected pooled 2/3-gram direction rows")
                grams[gram] = row
    if not grams:
        raise ValueError("No reached n-grams")
    counts = dict.fromkeys(grams, 0)
    for ids, mask in training_rows(args):
        for i, active in enumerate(mask):
            if active:
                for order in (2, 3):
                    if i >= order:
                        gram = tuple(ids[i-order:i])
                        if gram in counts:
                            counts[gram] += 1

    groups = defaultdict(Counter)
    totals = Counter()
    details = []
    for gram, row in grams.items():
        order, freq = len(gram), counts[gram]
        lower = 1 << (freq.bit_length() - 1) if freq else 0
        g = groups[order, lower]
        n = int(row["events"])
        plus, minus = n * float(row["g_plus"]), n * float(row["g_minus"])
        totals[order] += n
        g.update(events=n, types=1, plus=plus, minus=minus)
        for field in ("improved_events", "harmed_events", "neutral_events"):
            g[field] += int(row[field])
        if n >= 2:
            g.update(repeated=n, rp=plus, rm=minus, mixed=2 * min(plus, minus))
        details.append(dict(ngram_token_ids=row["ngram_token_ids"], train_count=freq,
                            **{k: row[k] for k in ("events", "improved_events",
                               "harmed_events", "neutral_events", "mean_delta_a",
                               "g_plus", "g_minus")}))
    summary = []
    for (order, lo), g in sorted(groups.items()):
        n, net = g["events"], g["plus"] - g["minus"]
        mass = g["rp"] + g["rm"]
        summary.append(dict(order=order, train_count_min=lo,
                            train_count_max=max(0, 2 * lo - 1), types=g["types"],
                            events=n, event_share=n/totals[order],
                            improved=g["improved_events"], harmed=g["harmed_events"],
                            neutral=g["neutral_events"], g_plus_pp=100*g["plus"]/n,
                            g_minus_pp=100*g["minus"]/n, net_pp=100*net/n,
                            contribution_pp=100*net/totals[order],
                            repeated_event_share=g["repeated"]/n,
                            repeated_magnitude_share=ratio(mass, g["plus"] + g["minus"]),
                            bucket_disagreement=ratio(2*min(g["rp"], g["rm"]), mass),
                            within_gram_disagreement=ratio(g["mixed"], mass)))
    write_csv(directory / "ngram_train_frequency.csv", summary)
    write_csv(directory / "ngram_train_frequency_details.csv", details)
    print(f"Results written to {directory}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
