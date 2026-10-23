# Qwen3-4B DSpark: DeepSpec evaluation inputs through vLLM

`EVAL_PROTOCOL=deepspec` keeps the acceptance-counter workflow of
`scripts/run_gsm8k_with_acceptance.sh`, using the public DeepSpec GSM8K recipe
instead of the default vLLM 5-shot evaluator. The default mode remains unchanged.

Use `eval_datasets/gsm8k.jsonl` from an official
[DeepSpec checkout](https://github.com/deepseek-ai/DeepSpec). No DeepSpec Python
installation is needed. Keep its Git revision with the experiment; the evaluator
also records the dataset SHA256 and selected source lines.

```bash
cd "$HOME/SpecForge"

SPECFORGE_DIR="$HOME/SpecForge" \
PYTHON_BIN="$HOME/vllm/.venv/bin/python" \
EVAL_PROTOCOL=deepspec \
EVAL_DEEPSPEC_DATA_PATH="$HOME/DeepSpec/eval_datasets/gsm8k.jsonl" \
EVAL_TOKENIZER_PATH="/data1/models/Qwen3-4B" \
EVAL_HOST="http://127.0.0.1" \
EVAL_PORT=11443 \
EVAL_MODEL="qwen3-4b" \
EVAL_OUTPUT_DIR="$HOME/logs/qwen3-4b/eval/gsm8k" \
bash scripts/run_gsm8k_with_acceptance.sh qwen3-4b-official-deepspec
```

The service must already use the matching BF16 Qwen3-4B target, official
`deepseek-ai/dspark_qwen3_4b_block7` draft, 7 speculative tokens,
`draft_sample_method="probabilistic"`, standard rejection sampling and disabled
adaptive verification. Client flags do not configure draft sampling. Run against
an otherwise idle service because `/metrics` counters are global.

The client defaults to:

- 500 samples, selected with `random.Random(980406).shuffle` then sliced;
- each row's first `turns` entry, preserving its existing reasoning instruction;
- one user message, no few-shot examples, `enable_thinking=False`;
- local tokenizer `apply_chat_template(tokenize=False, add_generation_prompt=True)`
  followed by tokenization with `add_special_tokens=False`;
- `/v1/completions` with prompt token IDs, avoiding an extra server chat template;
- temperature 1, top-p 1, top-k -1, min-p 0, neutral repetition/presence/frequency
  penalties, and EOS IDs from the local target's `generation_config.json` when present;
- maximum 2048 generated tokens, one concurrent request, seed `980406 + sample_index`.

`EVAL_NUM_QUESTIONS`, `EVAL_MAX_TOKENS`, `EVAL_SEED`, and
`EVAL_MAX_CONCURRENCY` can override those defaults. For a short plumbing check,
set `EVAL_NUM_QUESTIONS=8 EVAL_MAX_TOKENS=64`; such a run is not a paper comparison.
Sampling and chat environment variables from the legacy mode are ignored and
the script prints the fixed DeepSpec sampling settings.

Each run saves the input manifest and summary in `gsm8k.json`, exact rendered
prompts/token IDs, per-sample seeds and responses in `gsm8k.requests.jsonl`, plus
before/after metrics and `acceptance.json`. Request failures fail the run instead
of being treated as successful empty answers. This dataset has no answer labels,
so the evaluator reports acceptance and output diagnostics, not GSM8K accuracy.

Matching the recipe does not imply matching random draws between PyTorch and
vLLM or exactly matching EOS/length terminal-round accounting. The public main
branch is not a pinned artifact of the paper's original run. This mode is not a
guarantee that MAL must equal 6.11.
