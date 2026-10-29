"""Client contract tests: real local HTTP, bounded concurrency, no code execution."""

import asyncio
import json

import pytest
from aiohttp import web

from scripts.eval_humaneval_vllm import evaluate, load_tasks, parse_args


def metrics(rounds, drafted, accepted, positions):
    prefix = "vllm:spec_decode_"
    lines = [
        f'{prefix}num_drafts_total{{model_name="test"}} {rounds}',
        f'{prefix}num_draft_tokens_total{{model_name="test"}} {drafted}',
        f'{prefix}num_accepted_tokens_total{{model_name="test"}} {accepted}',
    ]
    lines += [
        f'{prefix}num_accepted_tokens_per_pos_total{{position="{i}",model_name="test"}} {n}'
        for i, n in enumerate(positions)
    ]
    return "\n".join(lines)


@pytest.mark.parametrize("failure", [None, "http", "metrics_reset"])
@pytest.mark.parametrize("concurrency,task_count", [(2, 4), (32, 40)])
def test_concurrent_client_preserves_sampling_metrics_and_failed_tasks(
    tmp_path, failure, concurrency, task_count
):
    async def run():
        active = peak = snapshots = 0
        payloads = []

        async def completion(request):
            nonlocal active, peak
            body = await request.json()
            payloads.append(body)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            if failure == "http" and body["seed"] == 43:
                return web.json_response({"error": "test failure"}, status=500)
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {"content": "def solution():\n    return 1"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 10},
                }
            )

        async def metric_response(request):
            nonlocal snapshots
            snapshots += 1
            text = (
                metrics(10, 20, 15, [10, 5])
                if snapshots == 1
                else metrics(14, 28, 21, [14, 7])
            )
            if snapshots > 1 and failure == "metrics_reset":
                text = metrics(1, 2, 1, [1, 0])
            return web.Response(text=text)

        app = web.Application()
        app.router.add_post("/v1/chat/completions", completion)
        app.router.add_get("/metrics", metric_response)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        out = tmp_path / "results"
        args = parse_args(
            [
                "--port",
                str(port),
                "--output-dir",
                str(out),
                "--max-concurrency",
                str(concurrency),
                "--metrics-wait-seconds",
                "0",
            ]
        )
        tasks = [
            {
                "task_id": f"HumanEval/{i}",
                "prompt": "def solution():\n    pass",
                "entry_point": "solution",
            }
            for i in range(task_count)
        ]
        try:
            code = await evaluate(args, tasks, lambda content, entrypoint: content)
        finally:
            await runner.cleanup()
        assert code == int(failure is not None)
        assert peak == concurrency
        assert len(payloads) == task_count  # Failed requests must not be retried.
        assert sorted(p["seed"] for p in payloads) == list(range(42, 42 + task_count))
        for payload in payloads:
            assert (
                payload["temperature"],
                payload["top_p"],
                payload["top_k"],
                payload["min_p"],
            ) == (0.7, 0.8, 20, 0)
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert payload["n"] == 1 and not payload["stream"]
        report = json.loads((out / "humaneval.json").read_text())
        assert (
            report["total_output_tokens"] == (task_count - int(failure == "http")) * 10
        )
        assert report["failed_requests"] == int(failure == "http")
        assert report["pass_at_1"] is None
        assert report["valid_measurement"] == (failure is None)
        if failure == "metrics_reset":
            assert report["acceptance"] is None
            assert "Counters decreased" in report["metrics_error"]
        else:
            assert report["acceptance"]["mal"] == 2.5
            assert report["acceptance"]["positions"][1]["continuous_rate"] == 0.5
            assert report["acceptance"]["positions"][1]["conditional_rate"] == 0.5
        samples = [
            json.loads(line)
            for line in (out / "samples.jsonl").read_text().splitlines()
        ]
        assert [row["task_id"] for row in samples] == [row["task_id"] for row in tasks]
        assert (samples[1]["solution"] == "") == (failure == "http")
        assert len((out / "responses.jsonl").read_text().splitlines()) == task_count

    asyncio.run(run())


def test_local_tasks_exclude_reference_answers_and_reject_duplicates(tmp_path):
    path = tmp_path / "tasks.jsonl"
    task = dict(
        task_id="HumanEval/0",
        prompt="def f(): pass",
        entry_point="f",
        canonical_solution="secret",
    )
    path.write_text(json.dumps(task) + "\n")
    args = parse_args(["--data-path", str(path), "--output-dir", str(tmp_path / "out")])
    assert "canonical_solution" not in load_tasks(args)[0]
    path.write_text((json.dumps(task) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_tasks(args)


@pytest.mark.parametrize(
    "option,value",
    [("--top-k", "0"), ("--temperature", "nan"), ("--max-concurrency", "0")],
)
def test_invalid_settings_fail_before_requests(tmp_path, option, value):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path / "out"), option, value])
