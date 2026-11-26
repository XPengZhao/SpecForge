import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.eval_ngram_trace_vllm import select_unique
from scripts.summarize_ngram_trace import records, main as summarize


def test_selection_is_unique_stable_and_records_subset_identity(tmp_path):
    path = tmp_path / "subset.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": str(i)},
                {"role": "assistant", "content": str(r)},
            ]
        }
        for i in range(30)
        for r in range(10)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = SimpleNamespace(
        data_path=path, sample_seed=42, num_questions=12, trace_prefix="ngram-run-"
    )
    a = select_unique(args)
    b = select_unique(args)
    assert a == b and len(a) == 12
    assert len({r["prompt_sha256"] for r in a}) == 12
    assert args.duplicate_prompts == 270 and args.eligible_records == 30
    assert len(args.data_sha256) == 64
    assert all(r["trace_request_id"].startswith("ngram-run-") for r in a)


def fixture_run(tmp_path):
    ev = tmp_path / "eval"
    ev.mkdir()
    traces = tmp_path / "traces"
    traces.mkdir()
    tasks = [
        dict(
            trace_request_id=f"ngram-run-{i:09d}",
            prompt_sha256="0" * 15 + ("0" if i == 1 else "1") + "0" * 48,
        )
        for i in (1, 2)
    ]
    (ev / "config.json").write_text(
        json.dumps(dict(trace_prefix="ngram-run-", tasks=tasks))
    )
    (ev / "regen.json").write_text(json.dumps(dict(valid_measurement=True)))
    events = []
    for task in tasks:
        for pos in (0, 1):
            events.append(
                dict(
                    type="event",
                    request_id="chatcmpl-" + task["trace_request_id"],
                    anchor_position=100,
                    position=pos,
                    reached=pos == 0,
                    accepted=False,
                    first_rejection=pos == 0,
                    raw_l1=0.4,
                    ngram2=[1, 2],
                    ngram3=[0, 1, 2],
                    top_token_ids=[7],
                    top_residual=[0.2],
                    fixed=[
                        dict(ngram_token_ids=[1, 2], token_ids=[7], residual=[-0.1])
                    ],
                )
            )
    (traces / "trace-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    return ev, traces, tasks, events


def test_summary_excludes_after_rejection_and_scores_fixed_candidates(tmp_path):
    ev, traces, _, _ = fixture_run(tmp_path)
    out = tmp_path / "out"
    summarize(
        [
            "--eval-dir",
            str(ev),
            "--trace-dir",
            str(traces),
            "--output-dir",
            str(out),
            "--min-count",
            "1",
            "--min-prompts",
            "1",
        ]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["events"] == 4 and summary["reached_events"] == 2
    results = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    row = next(x for x in results if len(x["ngram_token_ids"]) == 2)
    assert row["position"] == 0 and row["first_rejections"] == 2
    assert row["heldout"]["mean_residual"] == [-0.1]
    assert row["heldout"]["selected_consistency"] == pytest.approx(0.25)


def test_duplicate_worker_records_dedup_and_conflicts_fail(tmp_path):
    _, traces, tasks, events = fixture_run(tmp_path)
    (traces / "trace-b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    mapping = {x["trace_request_id"]: x for x in tasks}
    assert (
        len(list(records(sorted(traces.glob("*.jsonl")), mapping, "ngram-run-"))) == 4
    )
    events[0]["raw_l1"] = 0.9
    (traces / "trace-b.jsonl").write_text(json.dumps(events[0]) + "\n")
    with pytest.raises(ValueError, match="Conflicting"):
        list(records(sorted(traces.glob("*.jsonl")), mapping, "ngram-run-"))


def test_contributions_split_cancellation_cohorts_and_local_oracle(tmp_path):
    import csv
    from scripts.ngram_contributions import ContributionSummary

    stats = ContributionSummary(tmp_path)
    for i, (on, off, reached) in enumerate(
        ((0.8, 0.5, True), (0.4, 0.6, True), (0.9, 0.1, False))
    ):
        stats.add(
            dict(
                contribution=dict(a_on=on, a_off=off, delta_a=on - off),
                reached=reached,
                position=0,
                trace_request_id=str(i),
                prompt_hash=str(i),
                ngram2=[1, 2],
                ngram3=[0, 1, 2],
            )
        )
    summary = stats.finish(1, 1)
    reached = summary["overall"]["reached"]
    assert reached["events"] == 2
    assert reached["g_plus"] == pytest.approx(0.15)
    assert reached["g_minus"] == pytest.approx(0.1)
    assert reached["mean_delta_a"] == pytest.approx(0.05)
    assert reached["oracle_a"] == pytest.approx(0.7)
    assert reached["oracle_gain_over_on"] == pytest.approx(0.1)
    assert reached["improved_events"] == reached["harmed_events"] == 1
    assert summary["overall"]["after_rejection"]["mean_delta_a"] == pytest.approx(0.8)
    rows = list(csv.DictReader((tmp_path / "contributions.csv").open()))
    gram = next(
        r
        for r in rows
        if r["scope"] == "ngram"
        and r["group_key"] == "[1, 2]"
        and r["cohort"] == "reached"
    )
    assert int(gram["prompts"]) == 2
    assert float(gram["mean_delta_a"]) == pytest.approx(0.05)


def test_mixed_paired_and_legacy_events_fail(tmp_path):
    from scripts.ngram_contributions import ContributionSummary

    stats = ContributionSummary(tmp_path)
    row = dict(
        contribution=dict(a_on=0.8, a_off=0.5, delta_a=0.3),
        reached=True,
        position=0,
        trace_request_id="a",
        prompt_hash="a",
        ngram2=[1, 2],
        ngram3=None,
    )
    stats.add(row)
    row.pop("contribution")
    stats.add(row)
    with pytest.raises(ValueError, match="Mixed paired/unpaired"):
        stats.finish(1, 1)


def test_paired_trace_flows_into_summary_and_csv(tmp_path):
    import csv

    ev, traces, _, events = fixture_run(tmp_path)
    for row in events:
        row["contribution"] = dict(a_on=0.8, a_off=0.5, delta_a=0.3)
    (traces / "trace-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    out = tmp_path / "out"
    summarize(
        [
            "--eval-dir",
            str(ev),
            "--trace-dir",
            str(traces),
            "--output-dir",
            str(out),
            "--min-count",
            "1",
            "--min-prompts",
            "1",
        ]
    )
    result = json.loads((out / "summary.json").read_text())["contributions"]
    assert result["enabled"] and result["paired_events"] == 4
    assert result["overall"]["reached"]["events"] == 2
    assert result["overall"]["reached"]["mean_delta_a"] == pytest.approx(0.3)
    rows = list(csv.DictReader((out / "contributions.csv").open()))
    assert {r["scope"] for r in rows} == {"overall", "position", "request", "ngram"}


@pytest.mark.parametrize("fail_client", [False, True])
def test_one_click_lifecycle_and_paired_configuration(
    tmp_path, monkeypatch, fail_client
):
    from contextlib import contextmanager
    from scripts import run_ngram_diagnostics as runner

    for name in ("draft", "target"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    (tmp_path / "vllm/.venv/bin").mkdir(parents=True)
    for name in ("python", "vllm"):
        (tmp_path / "vllm/.venv/bin" / name).touch()
    (tmp_path / "data.jsonl").write_text("{}\n")
    args = runner.options(
        [
            "--checkpoint",
            str(tmp_path / "draft"),
            "--target",
            str(tmp_path / "target"),
            "--vllm-dir",
            str(tmp_path / "vllm"),
            "--data-path",
            str(tmp_path / "data.jsonl"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    events = []

    @contextmanager
    def serving(a, ckpt, mode, directory, *, extra_env):
        assert mode == "probabilistic" and ckpt == args.checkpoint
        assert extra_env["VLLM_NGRAM_TRACE_PAIRED"] == "1"
        assert extra_env["VLLM_NGRAM_TRACE_CANDIDATES"] is None
        events.append("start")
        try:
            yield SimpleNamespace(poll=lambda: None)
        finally:
            events.append("stop")

    def run_logged(cmd, log):
        if "eval_ngram_trace_vllm.py" in cmd[1]:
            events.append("eval")
            assert cmd[cmd.index("--num-questions") + 1] == "512"
            assert cmd[cmd.index("--max-concurrency") + 1] == "32"
            assert cmd[cmd.index("--temperature") + 1] == "0.7"
            if fail_client:
                raise RuntimeError("client failed")
        else:
            assert events[-1] == "stop"
            events.append("summary")
            out = Path(cmd[cmd.index("--output-dir") + 1])
            out.mkdir()
            (out / "summary.json").write_text(
                json.dumps(
                    dict(contributions=dict(enabled=True), requests_with_drafts=512)
                )
            )

    monkeypatch.setattr(runner, "serving", serving)
    monkeypatch.setattr(runner, "run_logged", run_logged)
    monkeypatch.setattr(runner, "read_metrics", lambda *args: dict(mal=5.0))
    if fail_client:
        with pytest.raises(RuntimeError, match="client failed"):
            runner.run(args)
        assert events == ["start", "eval", "stop"]
        root = next(args.output_root.iterdir())
        assert json.loads((root / "result.json").read_text())["status"] == "failed"
    else:
        root = runner.run(args)
        assert events == ["start", "eval", "stop", "summary"]
        result = json.loads((root / "result.json").read_text())
        assert result["status"] == "ok" and result["mal"] == 5.0


@pytest.mark.parametrize(
    "dataset,count",
    [
        ("gsm8k", 1319),
        ("humaneval", 164),
        ("mtbench", 80),
        ("math500", 500),
        ("aime", 30),
        ("regen", 512),
    ],
)
def test_dataset_launcher_defaults_and_native_arguments(tmp_path, dataset, count):
    from scripts import run_ngram_diagnostics as runner
    from scripts import eval_benchmark_ngram_trace as bridge

    for name in ("draft", "target"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    (tmp_path / "vllm/.venv/bin").mkdir(parents=True)
    for name in ("python", "vllm"):
        (tmp_path / "vllm/.venv/bin" / name).touch()
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    argv = [
        "--dataset",
        dataset,
        "--checkpoint",
        str(tmp_path / "draft"),
        "--target",
        str(tmp_path / "target"),
        "--vllm-dir",
        str(tmp_path / "vllm"),
    ]
    if dataset == "regen":
        argv += ["--data-path", str(data)]
    args = runner.options(argv)
    assert args.num_prompts == count
    client, _ = runner.commands(args, tmp_path / "run", "ngram-test-")
    assert client[client.index("--num-questions") + 1] == str(count)
    if dataset == "regen":
        assert client[1].endswith("eval_ngram_trace_vllm.py")
        return
    assert client[1].endswith("eval_benchmark_ngram_trace.py")
    assert client[client.index("--dataset") + 1] == dataset
    a = SimpleNamespace(
        dataset=dataset,
        model="qwen3-4b",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=42,
        max_tokens=4096,
        data_path=None,
        host="http://127.0.0.1",
        port=11443,
        num_questions=count,
        max_concurrency=32,
        request_timeout_seconds=1800,
        num_shots=0,
        output_dir=tmp_path / "eval",
    )
    native = bridge.evaluator_arguments(a)
    if dataset in ("math500", "aime"):
        with bridge.arguments(native):
            parsed = bridge.math_eval.parse_args()
        assert parsed.dataset == dataset and parsed.limit == count
    elif dataset != "gsm8k":
        module = bridge.human if dataset == "humaneval" else bridge.mtbench
        parsed = module.parse_args(native)
        assert parsed.num_questions == count
    else:
        assert native[native.index("--num-shots") + 1] == "0"
        assert "--use-chat-completions" in native and "--save-details" in native


def test_benchmark_capture_preserves_mtbench_history_and_restores_transport(
    tmp_path, monkeypatch
):
    import aiohttp
    from scripts import eval_benchmark_ngram_trace as bridge

    sent = []

    def original(session, url, **kwargs):
        sent.append(kwargs["json"])
        return "response"

    monkeypatch.setattr(aiohttp.ClientSession, "post", original)
    trace = bridge.RequestTrace(
        tmp_path, "ngram-mt-", "mtbench", "http://localhost:1234"
    )
    args = SimpleNamespace(
        model="q",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        max_tokens=100,
        seed=42,
        chat_template_kwargs={"enable_thinking": False},
    )
    task = dict(question_id=5, turns=["first question", "second question"])
    first = bridge.mtbench.request_payload(args, task, 0, 1, {})
    second = bridge.mtbench.request_payload(
        args, task, 0, 2, {5: dict(content="generated first answer")}
    )
    originals = json.loads(json.dumps([first, second]))
    with trace.capture():
        for payload in (first, second):
            assert (
                aiohttp.ClientSession.post(
                    None, "http://localhost:1234/v1/chat/completions", json=payload
                )
                == "response"
            )
    assert aiohttp.ClientSession.post is original
    assert [first, second] == originals
    assert sent[1]["messages"][1]["content"] == "generated first answer"
    assert [
        dict((k, v) for k, v in p.items() if k != "request_id") for p in sent
    ] == originals
    assert sent[0]["request_id"] != sent[1]["request_id"]
    assert (
        trace.tasks[0]["prompt_family_sha256"] == trace.tasks[1]["prompt_family_sha256"]
    )
    assert [t["turn"] for t in trace.tasks] == [1, 2]
    trace.finish(True)
    assert json.loads((tmp_path / "trace_manifest.json").read_text())[
        "valid_measurement"
    ]
    with pytest.raises(RuntimeError):
        with trace.capture():
            raise RuntimeError("stop")
    assert aiohttp.ClientSession.post is original


def test_summary_accepts_benchmark_manifest_and_separates_turns(tmp_path):
    import csv

    ev, traces, tasks, events = fixture_run(tmp_path)
    (ev / "config.json").unlink()
    (ev / "regen.json").unlink()
    for i, task in enumerate(tasks):
        task.update(dataset="mtbench", turn=i + 1, prompt_family_sha256="f" * 64)
    (ev / "trace_manifest.json").write_text(
        json.dumps(dict(tasks=tasks, trace_prefix="ngram-run-", valid_measurement=True))
    )
    for row in events:
        row["contribution"] = dict(a_on=0.8, a_off=0.5, delta_a=0.3)
    (traces / "trace-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    out = tmp_path / "out"
    summarize(
        [
            "--eval-dir",
            str(ev),
            "--trace-dir",
            str(traces),
            "--output-dir",
            str(out),
            "--min-count",
            "1",
            "--min-prompts",
            "1",
        ]
    )
    rows = list(csv.DictReader((out / "contributions.csv").open()))
    assert {r["group_key"] for r in rows if r["scope"] == "turn"} == {"1", "2"}


def test_mtbench_original_evaluator_runs_both_turns_with_trace(tmp_path, monkeypatch):
    import aiohttp
    from scripts import eval_benchmark_ngram_trace as bridge

    data = tmp_path / "mtbench.jsonl"
    data.write_text(
        json.dumps(dict(question_id=1, category="test", turns=["hello", "continue"]))
        + "\n"
    )
    sent = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def text(self):
            return json.dumps(
                dict(
                    choices=[
                        dict(message=dict(content="first answer"), finish_reason="stop")
                    ],
                    usage=dict(prompt_tokens=3, completion_tokens=2),
                )
            )

    def post(session, url, **kwargs):
        sent.append(kwargs["json"])
        return Response()

    async def snapshot(*args):
        return {}

    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    monkeypatch.setattr(bridge.mtbench, "snapshot", snapshot)
    monkeypatch.setattr(
        bridge.mtbench,
        "subtract",
        lambda *args: dict(mal=2.0, acceptance_rate=0.5, positions=[]),
    )
    original = bridge.evaluator_arguments
    monkeypatch.setattr(
        bridge,
        "evaluator_arguments",
        lambda a: original(a) + ["--metrics-wait-seconds", "0"],
    )
    out = tmp_path / "eval"
    code = bridge.main(
        [
            "--dataset",
            "mtbench",
            "--trace-prefix",
            "ngram-mt-",
            "--vllm-dir",
            str(tmp_path),
            "--data-path",
            str(data),
            "--num-questions",
            "1",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0 and len(sent) == 2
    assert sent[1]["messages"] == [
        dict(role="user", content="hello"),
        dict(role="assistant", content="first answer"),
        dict(role="user", content="continue"),
    ]
    manifest = json.loads((out / "trace_manifest.json").read_text())
    assert manifest["valid_measurement"] and len(manifest["tasks"]) == 2
    assert (out / "acceptance.json").exists()
    assert aiohttp.ClientSession.post is post


def test_event_direction_pools_positions_turns_and_accepts_one_prompt(tmp_path):
    import csv
    from scripts.ngram_contributions import ContributionSummary

    # Old prompt thresholds must not affect event-based labels.
    stats = ContributionSummary(
        tmp_path, direction_min_prompts=99, direction_min_prompt_events=999
    )

    def add(gram, delta, count, prompt="same", reached=True):
        for i in range(count):
            stats.add(
                dict(
                    contribution=dict(a_on=0.5 + delta, a_off=0.5, delta_a=delta),
                    reached=reached,
                    position=i % 7,
                    turn=1 + i % 2,
                    trace_request_id=prompt + str(i % 2),
                    prompt_hash=prompt,
                    ngram2=gram,
                    ngram3=None,
                )
            )

    add([1, 1], 0.02, 10)
    add([2, 2], -0.02, 10)
    add([3, 3], 0.02, 6)
    add([3, 3], -0.02, 4)
    add([4, 4], 0.0001, 10)
    add([5, 5], 0.002, 9)
    add([5, 5], -0.1, 1)
    add([6, 6], 0.05, 9)
    add([1, 1], -0.02, 10, reached=False)
    report = stats.finish(10, 100)
    rows = list(csv.DictReader((tmp_path / "ngram_directions.csv").open()))
    reached = {r["ngram_token_ids"]: r for r in rows if r["cohort"] == "reached"}
    assert {k: v["classification"] for k, v in reached.items()} == {
        "[1, 1]": "mostly_helpful",
        "[2, 2]": "mostly_harmful",
        "[3, 3]": "mixed_effect",
        "[4, 4]": "weak_effect",
        "[5, 5]": "inconclusive",
        "[6, 6]": "insufficient_data",
    }
    assert all(r["prompts"] == "1" for r in reached.values())
    assert "position" not in rows[0] and "turn" not in rows[0]
    mixed = reached["[3, 3]"]
    assert float(mixed["positive_event_fraction"]) == pytest.approx(0.6)
    assert float(mixed["negative_event_fraction"]) == pytest.approx(0.4)
    assert float(mixed["mean_delta_a"]) == pytest.approx(0.004)
    assert (
        next(r for r in rows if r["cohort"] == "after_rejection")["classification"]
        == "mostly_harmful"
    )
    coverage = next(
        r
        for r in report["ngram_directions"]["coverage_by_order"]
        if r["cohort"] == "reached"
    )
    assert coverage["events"] == 59 and coverage["sufficient_events"] == 50
    assert coverage["event_coverage"] == pytest.approx(50 / 59)
    assert coverage["negative_loss_coverage"] == pytest.approx(1.0)
    details = list(csv.DictReader((tmp_path / "ngram_prompt_contributions.csv").open()))
    assert {r["turn"] for r in details} == {"1", "2"}
    assert (tmp_path / "ngram_position_turn_contributions.csv").is_file()


def test_direction_prompt_fractions_are_auxiliary_and_family_is_deduplicated(tmp_path):
    import csv
    from scripts.ngram_contributions import ContributionSummary

    stats = ContributionSummary(tmp_path)
    for prompt, deltas in [("a", [0.02] * 6), ("b", [-0.02] * 3), ("sparse", [0.1])]:
        for i, delta in enumerate(deltas):
            stats.add(
                dict(
                    contribution=dict(a_on=0.5 + delta, a_off=0.5, delta_a=delta),
                    reached=True,
                    position=i % 7,
                    turn=1 + i % 2,
                    trace_request_id=prompt + str(i % 2),
                    prompt_hash=prompt,
                    ngram2=[1, 2],
                    ngram3=None,
                )
            )
    stats.finish(10, 100)
    row = next(csv.DictReader((tmp_path / "ngram_directions.csv").open()))
    assert row["prompts"] == "3" and row["events"] == "10"
    assert float(row["positive_prompt_fraction"]) == pytest.approx(2 / 3)
    assert float(row["positive_event_fraction"]) == pytest.approx(0.7)
    assert row["classification"] == "mixed_effect"
