"""Exercise two-turn histories and counter windows against a local HTTP service."""
import asyncio
import json

import pytest
from aiohttp import web

from scripts.eval_mtbench_vllm import evaluate, load_history, parse_args
from tests.test_scripts.test_eval_humaneval_vllm import metrics


@pytest.mark.parametrize("mode", ["generated", "fixed", "failure", "reset"])
def test_two_turn_measurement(tmp_path, mode):
    async def run():
        requests, snapshots = [], 0
        active = peak = 0

        async def complete(request):
            nonlocal active, peak
            body = await request.json()
            requests.append(body)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            if mode == "failure" and body["seed"] == 42:
                return web.json_response({}, status=500)
            return web.json_response({"choices": [{"message": {"content": "generated"},
                                      "finish_reason": "stop"}],
                                      "usage": {"completion_tokens": 10}})

        async def counters(request):
            nonlocal snapshots
            snapshots += 1
            n = snapshots * 10
            if mode == "reset" and snapshots == 2:
                n = 1
            return web.Response(text=metrics(n, 2*n, n, [n, 0]))

        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        app.router.add_get("/metrics", counters)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        tasks = [dict(question_id=i, category="test", turns=[f"q{i}", "followup"])
                 for i in range(3)]
        args = parse_args(["--output-dir", str(tmp_path / "out"), "--port", str(port),
                           "--max-concurrency", "2", "--metrics-wait-seconds", "0"])
        if mode == "fixed":
            args.first_turn_responses = tmp_path / "history.jsonl"
            args.first_turn_responses.write_text("\n".join(
                json.dumps(dict(**task, content="fixed")) for task in tasks))
        try:
            code = await evaluate(args, tasks)
        finally:
            await runner.cleanup()
        failed = mode in ("failure", "reset")
        assert code == int(failed)
        assert peak == 2
        assert len(requests) == (3 if failed else 6)
        for body in requests:
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            assert body["temperature"] == 0.7
            if len(body["messages"]) == 3:
                assert body["messages"][1] == {"role": "assistant", "content":
                                               "fixed" if mode == "fixed" else "generated"}
                assert body["messages"][2]["content"] == "followup"
        report = json.loads((args.output_dir / "mtbench.json").read_text())
        assert report["valid_measurement"] == (not failed)
        if not failed:
            assert report["total_output_tokens"] == 60
            assert report["acceptance"]["rounds"] == 20
            assert report["acceptance"]["mal"] == 2
            assert report["acceptance"]["acceptance_rate"] == 0.5
            assert report["turns"]["1"]["acceptance"]["rounds"] == 10
            assert report["turns"]["2"]["acceptance"]["rounds"] == 10
    asyncio.run(run())


def test_fixed_history_rejects_different_questions(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(dict(question_id=1, turns=["wrong", "b"], content="answer")))
    with pytest.raises(ValueError, match="mismatched"):
        load_history(path, [dict(question_id=1, turns=["a", "b"])])
