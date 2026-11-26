"""Run the sweep's original benchmark evaluator with request-ID tracing.

Only the HTTP request_id is added. Prompts, scoring, sampling, and MT-Bench
turn-two history remain owned by the existing evaluators.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import aiohttp

if __package__:
    from . import eval_humaneval_vllm as human
    from . import eval_math500_vllm as math_eval
    from . import eval_mtbench_vllm as mtbench
    from .stat_vllm_acceptance import subtract
else:
    import eval_humaneval_vllm as human
    import eval_math500_vllm as math_eval
    import eval_mtbench_vllm as mtbench
    from stat_vllm_acceptance import subtract


class RequestTrace:
    def __init__(self, directory, prefix, dataset, server_url):
        self.directory, self.prefix, self.dataset = directory, prefix, dataset
        self.server_url = server_url.rstrip("/")
        self.tasks = []
        self.stream = None

    def payload(self, url, payload):
        if str(url) not in (
            self.server_url + "/v1/chat/completions",
            self.server_url + "/v1/completions",
        ):
            return payload
        if payload.get("n", 1) != 1 or payload.get("stream", False):
            raise ValueError("Trace requires one non-streamed response per request")
        messages = payload.get("messages")
        prompt = messages if messages is not None else payload["prompt"]
        digest = hashlib.sha256(
            json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        index = len(self.tasks)
        if index >= 10**9:
            raise ValueError("Too many trace requests")
        rid = f"{self.prefix}{index:09d}"
        # Keep both MT-Bench turns in the same prompt fold for residual analysis.
        family = messages[:1] if self.dataset == "mtbench" else prompt
        family_hash = hashlib.sha256(
            json.dumps(family, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        task = dict(
            trace_request_id=rid,
            prompt_sha256=digest,
            prompt_family_sha256=family_hash,
            dataset=self.dataset,
            turn=1 + sum(m["role"] == "assistant" for m in messages) if messages else 1,
        )
        result = dict(payload, request_id=rid)
        if self.stream is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.stream = (self.directory / "trace_requests.jsonl").open("x")
        self.stream.write(
            json.dumps(dict(**task, payload=result), ensure_ascii=False) + "\n"
        )
        self.stream.flush()
        self.tasks.append(task)
        return result

    @contextmanager
    def capture(self):
        original = aiohttp.ClientSession.post

        def post(session, url, **kwargs):
            if "json" in kwargs:
                kwargs["json"] = self.payload(url, kwargs["json"])
            return original(session, url, **kwargs)

        aiohttp.ClientSession.post = post
        try:
            yield
        finally:
            aiohttp.ClientSession.post = original
            if self.stream is not None:
                self.stream.close()

    def finish(self, valid):
        self.directory.mkdir(parents=True, exist_ok=True)
        human.write_json(
            self.directory / "trace_manifest.json",
            dict(
                dataset=self.dataset,
                trace_prefix=self.prefix,
                tasks=self.tasks,
                valid_measurement=bool(valid and self.tasks),
                prompt_policy="Original benchmark evaluator; only request_id added",
            ),
        )


@contextmanager
def arguments(args):
    original = sys.argv
    sys.argv = [original[0], *args]
    try:
        yield
    finally:
        sys.argv = original


def evaluator_arguments(a):
    common = [
        "--model",
        a.model,
        "--temperature",
        str(a.temperature),
        "--top-p",
        str(a.top_p),
        "--top-k",
        str(a.top_k),
        "--seed",
        str(a.seed),
        "--max-tokens",
        str(a.max_tokens),
    ]
    data = ["--data-path", str(a.data_path)] if a.data_path else []
    if a.dataset in ("math500", "aime"):
        return (
            common
            + data
            + [
                "--dataset",
                a.dataset,
                "--server-url",
                f"{a.host}:{a.port}",
                "--limit",
                str(a.num_questions),
                "--concurrency",
                str(a.max_concurrency),
                "--thinking",
                "off",
                "--request-timeout",
                str(a.request_timeout_seconds),
                "--metrics-wait-seconds",
                "10",
                "--output",
                str(a.output_dir / f"{a.dataset}.json"),
            ]
        )
    common += [
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--num-questions",
        str(a.num_questions),
        "--max-concurrency",
        str(a.max_concurrency),
        "--request-timeout-seconds",
        str(a.request_timeout_seconds),
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if a.dataset == "gsm8k":
        return common + [
            "--use-chat-completions",
            "--num-shots",
            str(a.num_shots),
            "--save-results",
            str(a.output_dir / "gsm8k.json"),
            "--save-details",
            str(a.output_dir / "responses.jsonl"),
        ]
    return common + data + ["--output-dir", str(a.output_dir)]


async def snapshot(url):
    async with aiohttp.ClientSession() as session:
        return await human.snapshot(session, url)


def run_gsm8k(a, argv):
    path = a.vllm_dir / "tests/evals/gsm8k/gsm8k_eval.py"
    spec = importlib.util.spec_from_file_location("gsm8k_trace_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    # gsm8k_eval imports vllm.assets; mirror sweep's vLLM working directory.
    sys.path.insert(0, str(a.vllm_dir))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if a.data_path:

            def load_local():
                directory = a.data_path.is_dir()
                test = a.data_path / "test.jsonl" if directory else a.data_path
                if a.num_shots and not directory:
                    raise ValueError("Few-shot GSM8K requires train/test directory")
                train = (
                    list(module.read_jsonl(str(a.data_path / "train.jsonl")))
                    if a.num_shots
                    else []
                )
                return train, list(module.read_jsonl(str(test)))

            module.load_gsm8k_data = load_local
        a.output_dir.mkdir(parents=True, exist_ok=False)
        url = f"{a.host}:{a.port}"
        before = asyncio.run(snapshot(url))
        human.write_json(a.output_dir / "before.json", before)
        with arguments(argv):
            module.main()
        time.sleep(10)
        after = asyncio.run(snapshot(url))
        human.write_json(a.output_dir / "after.json", after)
        human.write_json(a.output_dir / "acceptance.json", subtract(before, after))
        report = json.loads((a.output_dir / "gsm8k.json").read_text())
        return report["diagnostics"]["request_errors"] == 0
    finally:
        sys.path.pop(0)


def run(a):
    trace = RequestTrace(a.output_dir, a.trace_prefix, a.dataset, f"{a.host}:{a.port}")
    valid = False
    try:
        with trace.capture():
            argv = evaluator_arguments(a)
            if a.dataset == "gsm8k":
                valid = run_gsm8k(a, argv)
            elif a.dataset in ("math500", "aime"):
                with arguments(argv):
                    code = math_eval.main()
                report = json.loads((a.output_dir / f"{a.dataset}.json").read_text())
                valid = (
                    code == 0
                    and report["request_errors"] == 0
                    and report["spec_decode"] is not None
                    and not report["spec_metrics_error"]
                )
            else:
                evaluator = human if a.dataset == "humaneval" else mtbench
                code = evaluator.main(argv)
                report = json.loads((a.output_dir / f"{a.dataset}.json").read_text())
                valid = code == 0 and report["valid_measurement"]
    finally:
        trace.finish(valid)
    return 0 if valid and trace.tasks else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        choices=("gsm8k", "humaneval", "mtbench", "math500", "aime"),
    )
    p.add_argument("--trace-prefix", required=True)
    p.add_argument("--host", default="http://127.0.0.1")
    p.add_argument("--port", type=int, default=11443)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--data-path", type=Path)
    p.add_argument("--vllm-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-questions", type=int, required=True)
    p.add_argument("--num-shots", type=int, default=0)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Unused: benchmarks follow original evaluator ordering",
    )
    p.add_argument("--request-timeout-seconds", type=float, default=1800)
    a = p.parse_args(argv)
    if not a.trace_prefix.startswith("ngram-") or not a.trace_prefix.endswith("-"):
        p.error("Invalid trace prefix")
    if a.output_dir.exists():
        p.error("Output directory already exists")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
