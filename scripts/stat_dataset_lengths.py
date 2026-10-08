#!/usr/bin/env python3
"""Stream a conversations JSONL and plot untruncated token-length CDFs.

One observation is one JSONL record. Prompt/response lengths sum the token
lengths of non-assistant/assistant message bodies respectively; total length
uses the complete non-thinking chat template, including control tokens.
Only the tokenizer is loaded, never model weights. Memory is bounded by the
batch size and the number of distinct lengths, rather than dataset file size.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path


SERIES = ("prompt", "response", "total")


def validate_record(row):
    messages = row.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing or empty conversations")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        if message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        if not isinstance(message.get("content"), str):
            raise ValueError("message content must be a string")
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("no assistant response")
    if not any(m["role"] != "assistant" for m in messages):
        raise ValueError("no prompt messages")
    return messages


def count_batch(batch, tokenizer):
    texts, spans = [], []
    for line_number, row, messages in batch:
        start = len(texts)
        texts.extend(m["content"] for m in messages)
        template_kwargs = {"enable_thinking": False}
        if row.get("tools"):
            template_kwargs["tools"] = row["tools"]
        try:
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                **template_kwargs,
            ))
        except Exception as exc:
            raise ValueError(f"chat template failed at JSONL line {line_number}: {exc}") from exc
        spans.append((line_number, messages, start, len(texts) - 1))
    encoded = tokenizer(
        texts, add_special_tokens=False, truncation=False, padding=False,
        return_attention_mask=False, return_token_type_ids=False, verbose=False,
    )["input_ids"]
    lengths = [len(ids) for ids in encoded]
    for line_number, messages, start, total_index in spans:
        prompt = response = 0
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                response += lengths[start + index]
            else:
                prompt += lengths[start + index]
        assistant_turns = sum(m["role"] == "assistant" for m in messages)
        yield line_number, assistant_turns, prompt, response, lengths[total_index]


def summarize(histogram, threshold):
    """Exact percentiles with linear interpolation, using a length histogram."""
    import bisect

    values = sorted(histogram)
    cumulative, count = [], 0
    for value in values:
        count += histogram[value]
        cumulative.append(count)

    def percentile(q):
        position = (count - 1) * q
        low, high = math.floor(position), math.ceil(position)
        low_value = values[bisect.bisect_right(cumulative, low)]
        high_value = values[bisect.bisect_right(cumulative, high)]
        return low_value + (high_value - low_value) * (position - low)

    above = sum(n for value, n in histogram.items() if value > threshold)
    return {
        "count": count,
        "mean": sum(value * n for value, n in histogram.items()) / count,
        "min": values[0],
        "p50": percentile(0.50), "p90": percentile(0.90),
        "p95": percentile(0.95), "p99": percentile(0.99),
        "max": values[-1],
        "above_threshold_count": above,
        "above_threshold_fraction": above / count,
    }


def save_cdf(histograms, output_dir, threshold, log_x):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    colors = ("#2878B5", "#E07B39", "#38966B")
    labels = ("Prompt (message bodies)", "Response (message bodies)", "Total (chat template)")
    with (output_dir / "cdf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "tokens", "count", "cdf"])
        for name, label, color in zip(SERIES, labels, colors):
            histogram = histograms[name]
            total = sum(histogram.values())
            x, y, cumulative = [], [], 0
            for value in sorted(histogram):
                cumulative += histogram[value]
                probability = cumulative / total
                writer.writerow([name, value, histogram[value], probability])
                x.append(value)
                y.append(probability)
            # Extend the first/last steps so the plot includes CDF = 0 and 1.
            lower = max(0, x[0] - 1)
            ax.step([lower] + x + [max(x[-1] + 1, threshold)],
                    [0] + y + [1], where="post", label=label, color=color,
                    linewidth=1.8, linestyle="--" if name == "total" else "-")
    ax.axvline(threshold, color="#777777", linestyle=":", linewidth=1.2,
               label=f"Training length limit ({threshold:,})")
    if log_x:
        # symlog preserves possible zero lengths while showing the long tail.
        ax.set_xscale("symlog", linthresh=1)
    ax.set(xlabel="Length (tokens, before truncation)", ylabel="Cumulative fraction",
           ylim=(0, 1.02), title="Dataset token-length distributions")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"length_cdf.{suffix}", dpi=200)
    plt.close(fig)


def load_saved_lengths(output_dir):
    """Rebuild histograms from saved counts without loading a tokenizer."""
    histograms = {name: Counter() for name in SERIES}
    with (output_dir / "lengths.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(f"{name}_tokens" for name in SERIES).issubset(reader.fieldnames or []):
            raise ValueError("lengths.csv is missing token-length columns")
        for row_number, row in enumerate(reader, 2):
            for name in SERIES:
                length = int(row[f"{name}_tokens"])
                if length < 0:
                    raise ValueError(f"negative token length at CSV row {row_number}")
                histograms[name][length] += 1
    count = sum(histograms["total"].values())
    if not count:
        raise ValueError("lengths.csv has no records")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    expected = summary.get("counts", {}).get("records_analyzed")
    if expected is not None and expected != count:
        raise ValueError(f"lengths.csv has {count} records but summary.json records {expected}")
    return histograms, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--tokenizer", help="Local model/tokenizer directory or HF ID")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, help="Length limit; defaults to 4096 or the saved threshold")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Stop after this many valid records for a smoke run")
    parser.add_argument("--log-x", action="store_true", help="Use a symlog token-length axis")
    parser.add_argument("--plot-only", action="store_true", help="Plot existing lengths.csv without tokenizing again")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.threshold is not None and args.threshold < 1) or (args.limit is not None and args.limit < 1):
        parser.error("batch size, threshold and limit must be positive")
    if not args.plot_only and (args.input is None or args.tokenizer is None):
        parser.error("--input and --tokenizer are required unless --plot-only is used")
    # Check before starting a potentially long tokenization run.
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing plotting dependency. Run: python -m pip install matplotlib\n"
                         "If statistics already exist, rerun with --plot-only.") from exc
    if args.plot_only:
        histograms, summary = load_saved_lengths(args.output_dir)
        threshold = args.threshold if args.threshold is not None else summary.get("threshold", 4096)
        save_cdf(histograms, args.output_dir, threshold, args.log_x)
        print(f"Reused {sum(histograms['total'].values()):,} saved records; no tokenization performed.")
        print(f"Saved CDF plots to {args.output_dir.resolve()}")
        return
    if args.threshold is None:
        args.threshold = 4096
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    from transformers import AutoTokenizer

    # No trust_remote_code or model loading is required for length statistics.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histograms = {name: Counter() for name in SERIES}
    counts = Counter()
    started = time.monotonic()

    with args.input.open(encoding="utf-8") as source, (args.output_dir / "lengths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["line_number", "assistant_turns", "prompt_tokens", "response_tokens", "total_tokens"])

        def flush(batch):
            previous = counts["records_analyzed"]
            for line_number, turns, prompt, response, total in count_batch(batch, tokenizer):
                writer.writerow([line_number, turns, prompt, response, total])
                for name, length in zip(SERIES, (prompt, response, total)):
                    histograms[name][length] += 1
                counts["records_analyzed"] += 1
                counts["assistant_turns"] += turns
                counts["multi_response_records"] += int(turns > 1)
            elapsed = time.monotonic() - started
            if previous == 0 or counts["records_analyzed"] // 10000 > previous // 10000:
                print(f"Analyzed {counts['records_analyzed']:,} records in {elapsed:.1f}s", file=sys.stderr)

        batch, selected = [], 0
        for line_number, line in enumerate(source, 1):
            counts["lines_read"] += 1
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
                if "status" in row and row["status"] != "success":
                    counts["non_success_records"] += 1
                    continue
                if "status" not in row:
                    counts["records_without_status"] += 1
                messages = validate_record(row)
            except (ValueError, TypeError) as exc:
                counts["invalid_records"] += 1
                if counts["invalid_records"] <= 5:
                    print(f"Skipping line {line_number}: {exc}", file=sys.stderr)
                continue
            batch.append((line_number, row, messages))
            selected += 1
            if len(batch) == args.batch_size:
                flush(batch)
                batch.clear()
            if args.limit is not None and selected >= args.limit:
                break
        if batch:
            flush(batch)
    if not counts["records_analyzed"]:
        raise SystemExit("No valid records found; check the JSONL format and status fields.")

    summary = {
        "input": str(args.input.resolve()), "tokenizer": args.tokenizer,
        "unit": "one JSONL record (all turns)",
        "prompt_definition": "sum of individually tokenized non-assistant message bodies",
        "response_definition": "sum of individually tokenized assistant message bodies",
        "total_definition": "complete chat template, enable_thinking=False, add_generation_prompt=False",
        "truncation": False, "threshold": args.threshold, "limit": args.limit,
        "counts": dict(counts),
        "lengths": {name: summarize(hist, args.threshold) for name, hist in histograms.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_cdf(histograms, args.output_dir, args.threshold, args.log_x)
    print("\n| Series | Mean | P50 | P90 | P95 | P99 | Max |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, stats in summary["lengths"].items():
        print(f"| {name} | " + " | ".join(f"{stats[key]:,.1f}" for key in
              ("mean", "p50", "p90", "p95", "p99", "max")) + " |")
    total_stats = summary["lengths"]["total"]
    print(f"\nTotal > {args.threshold:,}: {total_stats['above_threshold_count']:,} / "
          f"{total_stats['count']:,} ({total_stats['above_threshold_fraction']:.2%})")
    print(f"Record counts: {dict(counts)}")
    print(f"Saved CSVs, summary and CDF plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
