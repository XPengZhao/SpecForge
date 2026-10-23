"""Guard the official prompt/sample recipe and its acceptance wrapper."""

import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import io


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "deepspec_gsm8k_eval", ROOT / "scripts/evaluate_deepspec_gsm8k.py"
)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class RecipeTest(unittest.TestCase):
    def test_evaluation_records_per_sample_seeds_and_local_eos_ids(self):
        import contextlib
        import sys

        class Tokenizer:
            eos_token_id = 8

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

            def __call__(self, text, **kwargs):
                return {"input_ids": [int(text)]}

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = directory / "data.jsonl"
            data.write_text('\n'.join(json.dumps({"turns": [str(i)]}) for i in range(4)))
            (directory / "generation_config.json").write_text('{"eos_token_id": [9, 10]}')
            output = directory / "result.json"
            modules = {
                "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: Tokenizer())),
                "tqdm": SimpleNamespace(tqdm=lambda iterable, **_: iterable),
            }

            def send(url, body, timeout):
                return {"request": body, "response": {"choices": [{"text": "answer", "finish_reason": "stop"}], "usage": {"completion_tokens": 2}}}

            argv = ["eval", "--dataset-path", str(data), "--tokenizer", str(directory), "--model", "target", "--num-questions", "3", "--save-results", str(output)]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), patch.object(EVAL, "send_request", side_effect=send), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(EVAL.main(), 0)
            summary = json.loads(output.read_text())
            records = [json.loads(line) for line in output.with_suffix(".requests.jsonl").read_text().splitlines()]
            self.assertEqual(summary["request_errors"], 0)
            self.assertEqual(summary["completion_tokens"], 6)
            self.assertEqual([record["request"]["seed"] for record in records], [980406, 980407, 980408])
            self.assertEqual([record["source_line"] for record in records], summary["selected_source_lines"])
            self.assertTrue(all(record["request"]["stop_token_ids"] == [9, 10] for record in records))

    def test_sample_order_matches_deepspec_shuffle_then_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gsm8k.jsonl"
            rows = [{"turns": [f"question {i}"]} for i in range(1319)]
            path.write_text("\n".join(map(json.dumps, rows)))
            expected = rows.copy()
            random.Random(980406).shuffle(expected)
            actual = EVAL.select_samples(path, 500, 980406)
            self.assertEqual([row for _, row in actual], expected[:500])
            self.assertEqual(len(set(line for line, _ in actual)), 500)
            self.assertEqual([row for _, row in EVAL.select_samples(path, 2000, 1)], rows)

    def test_prompt_uses_only_first_turn_and_exact_non_thinking_template(self):
        class Tokenizer:
            def apply_chat_template(inner, messages, **kwargs):
                self.assertEqual(messages, [{"role": "user", "content": "original prompt"}])
                self.assertEqual(kwargs, dict(tokenize=False, add_generation_prompt=True, enable_thinking=False))
                return "rendered prompt"

            def __call__(inner, text, **kwargs):
                self.assertEqual(text, "rendered prompt")
                self.assertEqual(kwargs, {"add_special_tokens": False})
                return {"input_ids": [4, 5, 6]}

        text, request = EVAL.build_request(
            Tokenizer(), {"turns": ["original prompt", "ignored"]},
            model="target", seed=980409, max_tokens=2048, stop_token_ids=[7, 8],
        )
        self.assertEqual(text, "rendered prompt")
        self.assertEqual(request["prompt"], [4, 5, 6])
        self.assertEqual(request["seed"], 980409)
        self.assertEqual(request["stop_token_ids"], [7, 8])
        self.assertEqual((request["temperature"], request["top_p"], request["top_k"], request["min_p"]), (1, 1, -1, 0))
        self.assertNotIn("stop", request)

    def test_http_error_is_recorded_instead_of_counted_as_success(self):
        error = urllib.error.HTTPError("url", 400, "bad", {}, io.BytesIO(b"bad request"))
        with patch.object(EVAL.urllib.request, "urlopen", side_effect=error):
            record = EVAL.send_request("http://localhost:8000", {"seed": 1}, 10)
        self.assertIn("400", record["error"])
        self.assertIn("bad request", record["error"])
        self.assertNotIn("response", record)

    def test_wrapper_selects_official_defaults_without_requiring_vllm_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            capture = directory / "calls.jsonl"
            fake_python = directory / "python"
            # Use the current interpreter for a dependency-free argument recorder.
            import sys
            fake_python.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['CAPTURE'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            )
            fake_python.chmod(0o755)
            data = directory / "gsm8k.jsonl"
            data.write_text('{"turns":["test"]}\n')
            env = {k: v for k, v in os.environ.items() if not k.startswith("EVAL_")}
            env.update(
                EVAL_PROTOCOL="deepspec", EVAL_DEEPSPEC_DATA_PATH=str(data),
                EVAL_OUTPUT_DIR=str(directory / "output"), EVAL_METRICS_WAIT_SECONDS="0",
                PYTHON_BIN=str(fake_python), SPECFORGE_DIR=str(ROOT),
                VLLM_DIR=str(directory / "missing-vllm"), CAPTURE=str(capture),
            )
            run = subprocess.run(["bash", str(ROOT / "scripts/run_gsm8k_with_acceptance.sh"), "test"], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            evaluation = calls[1]
            self.assertTrue(evaluation[0].endswith("evaluate_deepspec_gsm8k.py"))
            for name, value in [("--num-questions", "500"), ("--max-tokens", "2048"), ("--seed", "980406"), ("--max-concurrency", "1")]:
                self.assertEqual(evaluation[evaluation.index(name) + 1], value)
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
