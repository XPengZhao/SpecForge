# MT-Bench acceptance evaluation

Run against an existing vLLM service, using the same `/metrics` counter deltas
as the GSM8K and HumanEval scripts. Requires `aiohttp` in the client environment.
The default workload is the first 80 questions from FastChat MT-Bench, two turns
per question, with temperature 0.7, top-p 0.8, top-k 20, seed 42, up to 4096
output tokens per turn, and 32 concurrent requests. Thinking is disabled by default.
This measures generation and speculative acceptance; it does not run a quality judge.

```bash
cd "$HOME/SpecForge"
bash scripts/run_mtbench_with_acceptance.sh qwen3-4b-baseline
```

Set `EVAL_MODEL` to the server's `--served-model-name`. As in HumanEval, override
settings using `EVAL_MAX_CONCURRENCY`, `EVAL_MAX_TOKENS`, `EVAL_TEMPERATURE`,
`EVAL_TOP_P`, `EVAL_TOP_K`, `EVAL_MIN_P`, `EVAL_SEED`, `EVAL_CHAT_TEMPLATE_KWARGS`,
`EVAL_HOST`, `EVAL_PORT`, `PYTHON_BIN`, and `EVAL_OUTPUT_DIR`.
Use `EVAL_DATA_PATH=/path/to/question.jsonl` for offline input instead of downloading
FastChat's question file. The exact selected questions and settings are saved in
`results/config.json`. No additional system message is inserted.

Each round runs all questions concurrently (up to the limit), then waits for the
metrics refresh before proceeding. Round 2 uses the assistant response from round 1.
To compare drafters on identical second-turn contexts, reuse one run's histories:

```bash
EVAL_FIRST_TURN_RESPONSES=/path/to/baseline/results/turn1.responses.jsonl \
  bash scripts/run_mtbench_with_acceptance.sh qwen3-4b-markov-mha
```

This still evaluates both rounds; the supplied responses replace only the context
used for round 2. Question IDs and both question texts must match.
Use the same local question file and sampling settings across runs.

Outputs under `$HOME/logs/qwen3-4b/eval/mtbench/<run>/results/`:

- `mtbench.json`: per-turn and overall acceptance, MAL, output throughput and lengths.
- `acceptance.json`: overall counter deltas and per-position acceptance.
- `turn1.responses.jsonl`, `turn2.responses.jsonl`: answers, usage, latency and errors.
- `before.json`, `turn1.after.json`, `turn2.after.json`: counter snapshots.
- `config.json`: reproducible input questions and client settings.

Overall acceptance is calculated from total counters, not an unweighted mean of
turn-level rates. Generation throughput excludes the deliberate metrics waits.
MAL follows the existing scripts: `1 + accepted_tokens / draft_rounds`.
Per-position numbering follows their existing one-based convention.

Use an otherwise idle service: these counters cover all requests, not just this
client. A failed request or inconsistent metrics marks the run invalid and stops
before the next round, without retries. `EVAL_SKIP_ACCEPTANCE=true` supports a
target-only generation run. Draft sampling is controlled by the running service.
