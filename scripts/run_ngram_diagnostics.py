#!/usr/bin/env python3
"""Start vLLM, evaluate a dataset, summarize Engram on/off, and stop.

Uses the sweep service lifecycle; requires the instrumented vLLM branch and
an exported Engram checkpoint. CUDA_VISIBLE_DEVICES selects the GPU.
"""

import argparse
import json
import math
import signal
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from .sweep_dspark_checkpoints import REPO, read_metrics, serving, stop_group
else:
    from sweep_dspark_checkpoints import REPO, read_metrics, serving, stop_group


DEFAULT_COUNTS = dict(
    regen=512, gsm8k=1319, humaneval=164, mtbench=80, math500=500, aime=30
)


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="Exported HF Engram directory"
    )
    p.add_argument("--dataset", choices=DEFAULT_COUNTS, default="regen")
    p.add_argument(
        "--data-path",
        type=Path,
        help="Dataset JSONL; GSM8K also accepts train/test directory",
    )
    p.add_argument("--data-dir", type=Path, help="Prepared sweep dataset directory")
    p.add_argument("--num-shots", type=int, default=0, help="GSM8K few-shot count")
    p.add_argument("--target", type=Path, default=Path("/data1/models/Qwen3-4B"))
    p.add_argument("--vllm-dir", type=Path, default=Path.home() / "vllm")
    p.add_argument("--python", type=Path, help="Evaluator Python; default vLLM .venv")
    p.add_argument(
        "--output-root", type=Path, default=Path.home() / "outputs/ngram-diagnostics"
    )
    p.add_argument("--port", type=int, default=11443)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--num-prompts", type=int, help="Dataset default count unless overridden"
    )
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument(
        "--trace-topk",
        type=int,
        default=32,
        help="Residual logging width, not sampling top-k",
    )
    p.add_argument("--max-events", type=int, default=2000000)
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--min-prompts", type=int, default=3)
    p.add_argument(
        "--direction-epsilon",
        type=float,
        default=0.001,
        help="Meaningful delta A threshold; .001 = 0.1 percentage point",
    )
    p.add_argument(
        "--direction-min-prompt-events",
        type=int,
        default=3,
        help="Deprecated: ignored by event-based direction classification",
    )
    p.add_argument(
        "--direction-min-prompts",
        type=int,
        default=5,
        help="Deprecated: ignored by event-based direction classification",
    )
    args = p.parse_args(argv)
    if args.num_prompts is None:
        args.num_prompts = DEFAULT_COUNTS[args.dataset]
    if args.data_path and args.data_dir:
        p.error("Use either --data-path or --data-dir")
    if args.data_dir:
        args.data_path = args.data_dir / (
            "gsm8k" if args.dataset == "gsm8k" else f"{args.dataset}.jsonl"
        )
    if args.dataset == "regen" and args.data_path is None:
        p.error("regen requires --data-path or --data-dir")
    if args.num_shots < 0 or (args.dataset != "gsm8k" and args.num_shots):
        p.error("--num-shots is a nonnegative GSM8K-only option")
    if args.data_path:
        args.data_path = args.data_path.expanduser().resolve()
        if args.dataset == "gsm8k" and args.data_path.is_dir():
            required = [args.data_path / "test.jsonl"]
            if args.num_shots:
                required.append(args.data_path / "train.jsonl")
        else:
            required = [args.data_path]
            if args.dataset == "gsm8k" and args.num_shots:
                p.error("GSM8K few-shot local data requires a train/test directory")
        for path in required:
            if not path.is_file():
                p.error(f"Missing dataset file: {path}")
    for name in ("checkpoint", "target", "vllm_dir", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = (
        (args.python or args.vllm_dir / ".venv/bin/python").expanduser().resolve()
    )
    for path in (
        args.checkpoint / "config.json",
        args.target / "config.json",
        args.vllm_dir / ".venv/bin/vllm",
        args.python,
    ):
        if not path.is_file():
            p.error(f"Missing file: {path}")
    positive = (
        args.concurrency,
        args.num_prompts,
        args.max_tokens,
        args.max_model_len,
        args.startup_timeout,
        args.request_timeout,
        args.max_events,
        args.min_count,
        args.min_prompts,
    )
    if not all(math.isfinite(x) and x > 0 for x in positive):
        p.error("Counts, lengths, and timeouts must be positive")
    if not 1 <= args.port <= 65535 or not 1 <= args.trace_topk <= 256:
        p.error("Invalid port or trace-topk (1..256)")
    if (
        not math.isfinite(args.temperature)
        or args.temperature < 0
        or not 0 < args.top_p <= 1
    ):
        p.error("temperature must be >=0 and top-p must be in (0,1]")
    if args.top_k != -1 and args.top_k < 1:
        p.error("top-k must be -1 or positive")
    if not math.isfinite(args.direction_epsilon) or args.direction_epsilon < 0:
        p.error("direction-epsilon must be finite and nonnegative")
    if min(args.direction_min_prompt_events, args.direction_min_prompts) < 1:
        p.error("direction prompt thresholds must be positive")
    return args


def commands(a, root, prefix):
    client = [
        str(a.python),
        str(REPO / "scripts/eval_ngram_trace_vllm.py"),
        "--trace-prefix",
        prefix,
        "--host",
        "http://127.0.0.1",
        "--port",
        str(a.port),
        "--model",
        "qwen3-4b",
        "--num-questions",
        str(a.num_prompts),
        "--max-concurrency",
        str(a.concurrency),
        "--sample-seed",
        str(a.sample_seed),
        "--seed",
        str(a.seed),
        "--temperature",
        str(a.temperature),
        "--top-p",
        str(a.top_p),
        "--top-k",
        str(a.top_k),
        "--max-tokens",
        str(a.max_tokens),
        "--request-timeout-seconds",
        str(a.request_timeout),
        "--output-dir",
        str(root / "eval"),
    ]
    if a.data_path:
        client += ["--data-path", str(a.data_path)]
    if a.dataset != "regen":
        client[1] = str(REPO / "scripts/eval_benchmark_ngram_trace.py")
        client += [
            "--dataset",
            a.dataset,
            "--vllm-dir",
            str(a.vllm_dir),
            "--num-shots",
            str(a.num_shots),
        ]
    summary = [
        str(a.python),
        str(REPO / "scripts/summarize_ngram_trace.py"),
        "--eval-dir",
        str(root / "eval"),
        "--trace-dir",
        str(root / "trace"),
        "--output-dir",
        str(root / "summary"),
        "--min-count",
        str(a.min_count),
        "--min-prompts",
        str(a.min_prompts),
    ]
    summary += [
        "--direction-epsilon",
        str(a.direction_epsilon),
        "--direction-min-prompt-events",
        str(a.direction_min_prompt_events),
        "--direction-min-prompts",
        str(a.direction_min_prompts),
    ]
    return client, summary


def run_logged(command, log):
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait()
        except BaseException:
            stop_group(process)
            raise
    if code:
        raise RuntimeError(f"Command failed with code {code}; see {log}")


def run(a):
    a.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ngram-", dir=a.output_root)).resolve()
    prefix = root.name + "-"
    client, summary = commands(a, root, prefix)
    extra_env = dict(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_NGRAM_TRACE_DIR=str(root / "trace"),
        VLLM_NGRAM_TRACE_PREFIX=prefix,
        VLLM_NGRAM_TRACE_PAIRED="1",
        VLLM_NGRAM_TRACE_TOPK=str(a.trace_topk),
        VLLM_NGRAM_TRACE_MAX_EVENTS=str(a.max_events),
        VLLM_NGRAM_TRACE_CANDIDATES=None,
    )
    manifest = dict(
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        trace_env=extra_env,
        client_command=client,
        summary_command=summary,
    )
    (root / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"Results: {root}", flush=True)
    try:
        print("Starting vLLM with paired Engram tracing...", flush=True)
        with serving(
            a, a.checkpoint, "probabilistic", root, extra_env=extra_env
        ) as server:
            print(
                f"Evaluating {a.num_prompts} prompts; log: {root / 'eval.log'}",
                flush=True,
            )
            run_logged(client, root / "eval.log")
            if server.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during evaluation; see {root / 'serve.log'}"
                )
        # Stop the server first so all trace records are flushed before reading.
        print("Summarizing contributions...", flush=True)
        run_logged(summary, root / "summary.log")
        stats = json.loads((root / "summary/summary.json").read_text())
        if not stats["contributions"]["enabled"]:
            raise RuntimeError("No paired events; use the updated vLLM branch")
        result = dict(
            status="ok",
            dataset=a.dataset,
            checkpoint=str(a.checkpoint),
            path=str(root),
            **read_metrics(a.dataset, root / "eval"),
            contributions=stats["contributions"],
            requests_with_drafts=stats["requests_with_drafts"],
        )
    except BaseException as exc:
        (root / "result.json").write_text(
            json.dumps(dict(status="failed", error=str(exc)), indent=2)
        )
        raise
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return root


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    run(options())
