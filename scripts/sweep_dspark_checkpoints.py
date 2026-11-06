#!/usr/bin/env python3
"""Sweep exported Qwen3 DSpark checkpoints on one dedicated vLLM GPU/port.

Example: CUDA_VISIBLE_DEVICES=0 python3 scripts/sweep_dspark_checkpoints.py
Use --list to preview checkpoints, --steps to select exports, --datasets to select tasks.
Each run has its own directory. summary.jsonl contains one result per dataset.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("gsm8k", "humaneval", "mtbench", "math500", "aime")
DATASETS = (*DEFAULT_DATASETS, "regen")
TOP_P = 1.0
TOP_K = -1
DRAFT_TOKENS = 7
# Each mode sets the target temperature and the vLLM draft sampling method.
MODES = {"greedy": ("0", "greedy"), "probabilistic": ("1", "probabilistic"),
         "greedy-t1": ("1", "greedy")}


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, default=Path("/data3/models/Qwen3-4B-DSpark"))
    p.add_argument("--target", type=Path, default=Path("/data1/models/Qwen3-4B"))
    p.add_argument("--pattern", default="*step*", help="Directory glob under --model-root (step2658 or step-2658)")
    p.add_argument("--data-dir", type=Path, help="Local dataset files prepared by prepare_dspark_eval_data.py")
    p.add_argument("--steps", nargs="+", type=int)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DEFAULT_DATASETS)
    p.add_argument("--modes", nargs="+", choices=MODES, default=["greedy", "probabilistic"])
    p.add_argument("--port", type=int, default=11443)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--vllm-dir", type=Path, default=Path.home() / "vllm")
    p.add_argument("--python", type=Path, help="Evaluator Python (defaults to vLLM .venv)")
    p.add_argument("--output-root", type=Path, default=Path.home() / "logs/qwen3-4b/eval/sweeps")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if not 1 <= args.port <= 65535 or min(args.max_model_len, args.concurrency, args.startup_timeout) <= 0:
        p.error("Specify a valid port and positive concurrency, model length, and startup timeout")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.modes)) != len(args.modes):
        p.error("Specify each dataset and mode only once")
    if args.data_dir:
        args.data_dir = args.data_dir.resolve()
        for dataset in args.datasets:
            files = (["gsm8k/train.jsonl", "gsm8k/test.jsonl"] if dataset == "gsm8k"
                     else [f"{dataset}.jsonl"])
            for filename in files:
                if not (args.data_dir / filename).is_file():
                    p.error(f"Missing local dataset file: {args.data_dir / filename}")
    return args


def checkpoints(a):
    found = []
    for path in a.model_root.glob(a.pattern):
        match = re.search(r"(?:^|[-_])step-?(\d+)(?=$|[-_])", path.name)
        if path.is_dir() and match:
            step = int(match[1])
            if a.steps is None or step in a.steps:
                found.append((step, path.absolute()))
    missing = set(a.steps or ()) - {step for step, _ in found}
    if not found or missing:
        raise ValueError(f"No matching checkpoints or missing steps {sorted(missing)} under {a.model_root}")
    return sorted(found)


def stop_group(process):
    # vLLM owns additional worker processes. Only signal the group we started.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            process.poll()
            os.killpg(process.pid, 0)
            time.sleep(0.25)
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def serving(a, checkpoint, mode, directory):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", a.port))
    spec = dict(method="dspark", model=str(checkpoint), num_speculative_tokens=DRAFT_TOKENS,
                draft_sample_method=MODES[mode][1], attention_backend="FLASH_ATTN",
                enable_adaptive_verification=False)
    cmd = [str(a.vllm_dir / ".venv/bin/vllm"), "serve", str(a.target),
           "--host", "127.0.0.1", "--port", str(a.port), "--served-model-name", "qwen3-4b",
           "--dtype", "bfloat16", "--max-model-len", str(a.max_model_len),
           "--max-num-seqs", str(a.concurrency), "--speculative-config", json.dumps(spec)]
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    (directory / "serve.json").write_text(json.dumps(dict(command=cmd, gpu=env["CUDA_VISIBLE_DEVICES"]), indent=2))
    log = directory / "serve.log"
    with log.open("a") as stream:
        server = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            deadline = time.monotonic() + a.startup_timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {server.returncode}; see {log}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"vLLM startup timed out; see {log}")
            yield server
        finally:
            stop_group(server)


def run_dataset(a, mode, dataset, out):
    out.mkdir()
    python = str(a.python or a.vllm_dir / ".venv/bin/python")
    env = dict(os.environ, VLLM_DIR=str(a.vllm_dir), PYTHON_BIN=python,
               SPECFORGE_DIR=str(REPO), EVAL_SERVER_URL=f"http://127.0.0.1:{a.port}", EVAL_PORT=str(a.port), EVAL_HOST="http://127.0.0.1",
               EVAL_MODEL="qwen3-4b", EVAL_OUTPUT_DIR=str(out), EVAL_TEMPERATURE=MODES[mode][0],
               EVAL_GSM8K_SAVE_DETAILS="true", EVAL_TOP_P=str(TOP_P), EVAL_TOP_K=str(TOP_K), EVAL_MIN_P="0", EVAL_SKIP_ACCEPTANCE="false",
               EVAL_USE_CHAT_COMPLETIONS="true", EVAL_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}',
               EVAL_MAX_CONCURRENCY=str(a.concurrency))
    if a.data_dir:
        if dataset == "gsm8k":
            env["TMPDIR"] = str(a.data_dir / "gsm8k")
        else:
            env["EVAL_DATA_PATH"] = str(a.data_dir / f"{dataset}.jsonl")
    cmd = ["bash", str(REPO / f"scripts/run_{dataset}_with_acceptance.sh"), mode]
    with (out / "eval.log").open("w") as log:
        client = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            returncode = client.wait()
        except BaseException:
            stop_group(client)
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, cmd)
    return read_metrics(dataset, out)


def mean_token_lengths(dataset, directory):
    filenames = {
        "mtbench": ("turn1.responses.jsonl", "turn2.responses.jsonl"),
        "math500": ("math500.responses.jsonl",),
        "aime": ("aime.responses.jsonl",),
    }.get(dataset, ("responses.jsonl",))
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    requests = 0
    for filename in filenames:
        path = directory / filename
        if not path.is_file():
            return dict(mean_input_tokens=None, mean_output_tokens=None)
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("error") or record.get("request_error"):
                    continue
                response = record.get("response", {}) if dataset == "gsm8k" else record
                usage = response.get("usage") or {}
                requests += 1
                for key, total in totals.items():
                    value = usage.get(key)
                    # A partial total would understate the per-request mean.
                    if type(value) is not int or value < 0:
                        totals[key] = None
                    elif total is not None:
                        totals[key] += value
    def mean(key):
        return totals[key] / requests if requests and totals[key] is not None else None

    return dict(mean_input_tokens=mean("prompt_tokens"), mean_output_tokens=mean("completion_tokens"))


def read_metrics(dataset, out):
    is_math = dataset in ("aime", "math500")
    filename = f"{dataset}.json" if is_math else "acceptance.json"
    files = list(out.rglob(filename))
    if len(files) != 1:
        raise ValueError(f"Expected one {filename} in {out}, found {len(files)}")
    if dataset == "gsm8k":
        result = json.loads(files[0].with_name("gsm8k.json").read_text())
        errors = result["diagnostics"]["request_errors"]
        if errors != 0:
            raise ValueError(f"GSM8K reported {errors} failed requests")
    acc = json.loads(files[0].read_text())
    if is_math:
        spec = acc["spec_decode"]
        if not spec or spec["draft_token_acceptance_rate"] is None:
            raise ValueError(f"Missing {dataset} acceptance counters")
        # Math evaluators preserve the zero-based vLLM counter labels.
        positions = {i + 1: rate for i, rate in zip(spec["position_labels"], spec["continuous_acceptance"])}
        metrics = dict(mal=spec["mal"], acceptance_rate=spec["draft_token_acceptance_rate"])
    else:
        positions = {row["position"]: row["continuous_rate"] for row in acc["positions"]}
        metrics = dict(mal=acc["mal"], acceptance_rate=acc["acceptance_rate"])
    metrics.update({f"pos-{i}": positions.get(i) for i in range(1, DRAFT_TOKENS + 1)})
    metrics.update(mean_token_lengths(dataset, files[0].parent))
    if dataset == "gsm8k":
        metrics["num_shots"] = result["num_shots"]
    return metrics


def main():
    a = options()
    jobs = checkpoints(a)
    for step, path in jobs:
        print(f"step={step} {path}", flush=True)
    if a.list:
        return 0
    a.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="sweep-", dir=a.output_root)).resolve()
    print(f"Results: {root}", flush=True)
    failed = False
    for step, checkpoint in jobs:
        for mode in a.modes:
            directory = root / checkpoint.name / mode
            directory.mkdir(parents=True)
            print(f"Starting {checkpoint.name} {mode}", flush=True)
            pending = list(a.datasets)
            while pending:
                with ExitStack() as stack:
                    startup_error = None
                    try:
                        if not (checkpoint / "config.json").is_file():
                            raise ValueError(f"Missing config.json in {checkpoint}")
                        server = stack.enter_context(serving(a, checkpoint, mode, directory))
                    except (OSError, RuntimeError, ValueError) as exc:
                        startup_error = str(exc)
                    while pending:
                        dataset = pending.pop(0)
                        out = directory / dataset
                        row = dict(checkpoint=checkpoint.name, step=step, mode=mode, dataset=dataset,
                                   temperature=float(MODES[mode][0]), top_p=TOP_P, top_k=TOP_K,
                                   draft_sample_method=MODES[mode][1], model=checkpoint.name, path=str(out))
                        try:
                            if startup_error is not None:
                                raise RuntimeError(startup_error)
                            print(f"Running {dataset}; log: {out}/eval.log", flush=True)
                            metrics = run_dataset(a, mode, dataset, out)
                            if server.poll() is not None:
                                raise RuntimeError("vLLM exited during evaluation")
                            row.update(status="ok", **metrics)
                        except (OSError, ValueError, KeyError, TypeError,
                                subprocess.CalledProcessError, RuntimeError) as exc:
                            failed = True
                            row.update(status="failed", error=str(exc), mal=None, acceptance_rate=None,
                                       mean_input_tokens=None, mean_output_tokens=None)
                            row.update({f"pos-{i}": None for i in range(1, DRAFT_TOKENS + 1)})
                            if startup_error is not None:
                                row["path"] = str(directory)
                        with (root / "summary.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        print(json.dumps(row), flush=True)
                        if row["status"] == "failed" and startup_error is None:
                            # Restart before measuring the next dataset: failed clients
                            # can leave outstanding requests in the counters' window.
                            break
    return int(failed)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
