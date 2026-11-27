import http.client
import json
import sys
from threading import Event
from unittest.mock import Mock

import pytest

from scripts import generate_target_rollouts as rollout


def test_slow_first_request_does_not_block_refill_or_resume(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "round.jsonl"
    rows = [{"conversations": [{"role": "user", "content": str(i)}]} for i in range(1, 5)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", [
        "rollout", "--data-path", str(source), "--output-path", str(output),
        "--tokenizer-path", "unused", "--concurrency", "2", "--seed-offset", "42",
    ])
    args = rollout.parse_args()
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = lambda messages, **kw: messages[0]["content"]
    tokenizer.encode.return_value = [1]
    monkeypatch.setattr(rollout, "load_tokenizer", lambda *a, **kw: tokenizer)
    third_started = Event()

    def complete(endpoint, *, prompt, seed, **kwargs):
        assert seed == 42 + int(prompt)
        if prompt == "1":
            assert third_started.wait(5), "slow first request prevented refill"
        elif prompt == "3":
            third_started.set()
        return dict(output_text=prompt, finish_reason="stop", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(rollout, "post_completion", complete)
    rollout.run(args)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert saved[0]["source_line_number"] == 2
    assert sorted(row["source_line_number"] for row in saved) == [1, 2, 3, 4]
    output.write_text(json.dumps(saved[0]) + "\n")
    args.resume = True
    rollout.run(args)
    assert sorted(json.loads(line)["source_line_number"] for line in output.read_text().splitlines()) == [1, 2, 3, 4]


def test_grouped_rounds_stream_before_finish_and_resume(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    # Round 1 of row 1 already exists. Round 2 must warm the rank, and round 3
    # must start before round 2 finishes, even if its first token is retried.
    source = tmp_path / "input.jsonl"
    out = tmp_path / "out"
    out.mkdir()
    source.write_text(json.dumps({"conversations": [{"role": "user", "content": "test"}]}) + "\n")
    existing = '{"source_line_number":1}\n'
    monkeypatch.setattr(sys, "argv", [
        "rollout", "--data-path", str(source), "--output-path", str(out),
        "--tokenizer-path", "unused", "--concurrency", "2", "--rounds", "3",
        "--dp-ranks", "8", "--seed-offset", "42", "--seed-stride", "1", "--resume",
    ])
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "test"
    tokenizer.encode.return_value = [1]
    monkeypatch.setattr(rollout, "load_tokenizer", lambda *a, **kw: tokenizer)
    follower = Event()
    seen = []

    def complete(endpoint, *, seed, rank, on_ready, **kwargs):
        seen.append((seed, rank))
        if seed == 44:
            on_ready()
            on_ready()
            assert follower.wait(5)
        else:
            follower.set()
        return dict(output_text=str(seed), finish_reason="stop", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(rollout, "post_completion", complete)
    args = rollout.parse_args()
    rollout.prepare_manifest(args, [], numbered=True)
    (out / "round-01.jsonl").write_text(existing)
    rollout.run(args)
    rollout.run(args)
    assert "progress=1/3 (33.33%) resumed=1 written=0" in caplog.text
    assert "progress=3/3 (100.00%) resumed=3 written=0" in caplog.text
    assert "ETA=0:00:00" in caplog.text
    assert seen == [(44, 0), (45, 0)]
    tokenizer.encode.assert_called_once()
    assert (out / "round-01.jsonl").read_text() == existing
    for index, seed in [(2, 44), (3, 45)]:
        record = json.loads((out / f"round-{index:02d}.jsonl").read_text())
        assert record["conversations"][-1]["content"] == str(seed)


def test_completion_stream_parses_text_and_final_usage(monkeypatch):
    import io

    chunks = [
        {"choices": [{"text": "", "finish_reason": None}]},
        {"choices": [{"text": "你好", "finish_reason": None}]},
        {"choices": [{"text": "！", "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    ]
    def open_stream(req, timeout):
        assert dict((k.lower(), v) for k, v in req.header_items())["x-data-parallel-rank"] == "3"
        assert json.loads(req.data)["stream_options"]["include_usage"]
        return io.BytesIO(("".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                           + "data: [DONE]\n\n").encode())

    monkeypatch.setattr(rollout.urllib.request, "urlopen", open_stream)
    notify = Mock()
    result = rollout.post_completion("http://localhost/test", model="qwen", prompt="p",
                                     max_tokens=5, temperature=1, top_p=1, end_marker="END",
                                     timeout=5, rank=3, on_ready=notify)
    notify.assert_called_once()
    assert result == dict(output_text="你好！", finish_reason="stop", prompt_tokens=4, completion_tokens=2)


def test_second_machine_uses_global_round_numbers_and_seeds(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    source.write_text('\n{"conversations":[{"role":"user","content":"test"}]}\n')
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "rollout", "--data-path", str(source), "--output-path", str(out),
        "--tokenizer-path", "unused", "--first-round", "6", "--rounds", "5",
        "--seed-stride", "2", "--seed-offset", "42", "--resume",
    ])
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "test"
    tokenizer.encode.return_value = [1]
    monkeypatch.setattr(rollout, "load_tokenizer", lambda *a, **kw: tokenizer)
    def complete(endpoint, *, seed, on_ready, **kwargs):
        if on_ready:
            on_ready()
        return dict(output_text=str(seed), finish_reason="stop", prompt_tokens=1, completion_tokens=1)
    monkeypatch.setattr(rollout, "post_completion", complete)
    rollout.run(rollout.parse_args())
    assert len(list(out.glob("round-*.jsonl"))) == 5
    for r in range(6, 11):
        row = json.loads((out / f"round-{r:02d}.jsonl").read_text())
        assert row["conversations"][-1]["content"] == str(42 + 2 + (r - 1) * 2)


def test_manifest_overlong_and_exact_coverage(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    out = tmp_path / "out"
    source.write_text("".join(json.dumps({"conversations": [{"role": "user", "content": text}]}) + "\n"
                              for text in ["x", "long"]))
    monkeypatch.setattr(sys, "argv", [
        "rollout", "--data-path", str(source), "--output-path", str(out),
        "--tokenizer-path", "unused", "--max-length", "4", "--rounds", "2",
        "--seed-offset", "42", "--resume",
    ])
    args = rollout.parse_args()
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = lambda messages, **kw: messages[0]["content"]
    tokenizer.encode.side_effect = lambda text, **kw: list(text)
    monkeypatch.setattr(rollout, "load_tokenizer", lambda *a, **kw: tokenizer)

    def complete(endpoint, *, on_ready, **kw):
        if on_ready:
            on_ready()
        return dict(output_text="answer", finish_reason="length", prompt_tokens=1, completion_tokens=3)

    monkeypatch.setattr(rollout, "post_completion", complete)
    rollout.run(args)
    rollout.run(args)
    assert tokenizer.encode.call_count == 2
    assert json.loads((out / ".rollout-state/skipped-overlong.jsonl").read_text()) == {
        "source_line_number": 2, "prompt_tokens": 4}
    result = out / "round-01.jsonl"
    saved = result.read_text()
    assert json.loads(saved)["target_rollout"]["finish_reason"] == "length"
    args.temperature += 1
    with pytest.raises(ValueError, match="temperature"):
        rollout.run(args)
    args.temperature -= 1
    original = source.read_text()
    source.write_text(original.replace('"x"', '"y"'))
    with pytest.raises(ValueError, match="input_sha256"):
        rollout.run(args)
    source.write_text(original)
    result.write_text(saved + saved)
    with pytest.raises(ValueError, match="duplicate source line"):
        rollout.run(args)
    result.write_text(saved + '{"source_line_number":2}\n')
    with pytest.raises(ValueError, match="1 unexpected"):
        rollout.run(args)
    (out / ".rollout-state/manifest.json").unlink()
    with pytest.raises(ValueError, match="no manifest"):
        rollout.run(args)


def test_transport_retries_preserve_round_and_seed(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", [
        "rollout", "--data-path", "unused", "--output-path", "unused-out",
        "--tokenizer-path", "unused", "--seed-stride", "10", "--seed-offset", "42",
        "--retry-delay", "0",
    ])
    post = Mock(side_effect=[http.client.IncompleteRead(b"x"), ConnectionResetError(), {"ok": True}])
    monkeypatch.setattr(rollout, "post_completion", post)
    result = rollout.request_with_retries("url", args=rollout.parse_args(), prompt="p",
                                          max_tokens=1, source_line=2, round_index=5, rank=3)
    assert result == {"ok": True}
    assert [call.kwargs["seed"] for call in post.call_args_list] == [94] * 3
    assert "round 6 source line 2" in caplog.text
