# Qwen3-4B DSpark HumanEval evaluation

`scripts/run_humaneval_with_acceptance.sh` sends one chat-completion request
per HumanEval task to an existing vLLM service. It maintains up to **32 concurrent
requests across tasks**, rather than asking for 32 samples of one task.
The default dataset contains all 164 HumanEval tasks, obtained through EvalPlus.
Each task uses the EvalPlus-style instruction to return a self-contained Python
script in a Markdown code block. No reference solutions or tests enter the prompt.

## Run against an existing server

Keep the baseline or Markov MLA `vllm serve` configuration, including its CUDA
Graph settings. Set `--served-model-name qwen3-4b` and `--max-num-seqs 32` on the
server for the current comparison. Draft sampling is controlled by the service's
`draft_sample_method`; the client controls **target** sampling.

Create a separate client environment so evaluation dependencies do not change
the custom vLLM installation:

```bash
uv venv "$HOME/.venvs/humaneval" --python 3.12
uv pip install --python "$HOME/.venvs/humaneval/bin/python" aiohttp evalplus

cd "$HOME/SpecForge"
PYTHON_BIN="$HOME/.venvs/humaneval/bin/python" \
EVAL_MODEL=qwen3-4b \
EVAL_MAX_CONCURRENCY=32 \
EVAL_TEMPERATURE=0.7 EVAL_TOP_P=0.8 EVAL_TOP_K=20 \
bash scripts/run_humaneval_with_acceptance.sh markov-mla-probabilistic
```

The run name is only a label; it does not select the draft checkpoint or draft
sampler. Restart the service with the corresponding checkpoint/sampler before
each comparison. Use an otherwise idle service: acceptance counters include all
requests handled between the two snapshots.

The wrapper uses the same `EVAL_*` sampling variables as the GSM8K wrapper:

| Variable | Default |
| --- | --- |
| `EVAL_HOST`, `EVAL_PORT` | `http://127.0.0.1`, `8000` (host without port or `/v1`) |
| `EVAL_MODEL` | `qwen3-4b` |
| `EVAL_MAX_CONCURRENCY` | `32` |
| `EVAL_NUM_QUESTIONS` | `164` |
| `EVAL_MAX_TOKENS` | `4096` |
| `EVAL_TEMPERATURE`, `EVAL_TOP_P`, `EVAL_TOP_K`, `EVAL_MIN_P` | `0.7`, `0.8`, `20`, `0` |
| `EVAL_SEED` | `42`; request seed is seed + task index |
| `EVAL_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking": false}` |
| `EVAL_REQUEST_TIMEOUT_SECONDS` | `600` per request, after admission |
| `EVAL_METRICS_WAIT_SECONDS` | `10`; excluded from generation timing |
| `EVAL_OUTPUT_DIR` | `$HOME/logs/qwen3-4b/eval/humaneval` |
| `EVAL_DATA_PATH` | Optional local JSONL with `task_id`, `prompt`, `entry_point` |
| `EVAL_SKIP_ACCEPTANCE` | `false`; set `true` for target-only measurements |

Use `OPENAI_API_KEY` if the endpoint requires authentication. It is not written
to the saved configuration. For T=1 without truncation set temperature/top-p/top-k
to `1`, `1`, `-1`; for target greedy use `0`, `1`, `-1`.

The Python entry point exposes equivalent flags:

```bash
"$HOME/.venvs/humaneval/bin/python" scripts/eval_humaneval_vllm.py \
  --host http://127.0.0.1 --port 8000 --model qwen3-4b \
  --max-concurrency 32 --temperature 1 --top-p 1 --top-k -1 \
  --output-dir "$HOME/logs/qwen3-4b/eval/humaneval/new-run"
```

Output directories must be new. There is no resume mode or automatic request
retry, so cached outputs and duplicate requests cannot silently affect metrics.

## Outputs and grading

The shell wrapper prints its unique run directory and writes `humaneval.log`.
Inside `results/`:

- `config.json`: sampling settings, seed policy, prompt/task inputs and client
  package versions. Record the server command, checkpoint and GPU separately.
- `responses.jsonl`: raw API responses, token usage, finish reasons, request
  latency and any errors; written as each request completes.
- `before.json`, `after.json`: snapshots in the existing GSM8K acceptance format.
- `acceptance.json`: counter differences, MAL, overall acceptance, and per-position
  accepted counts, continuous rates and conditional rates.
- `humaneval.json`: generation wall time, questions/s, output tokens/s, MOL,
  failed/truncated request counts, empty solutions and acceptance results.
- `samples.jsonl`: one sanitized `task_id`/`solution` record per task, suitable
  for EvalPlus. Failed requests retain an empty solution rather than disappearing
  from the evaluation denominator.

Metrics use the same `stat_vllm_acceptance.py` parser and subtraction logic as
GSM8K: `MAL = 1 + accepted / rounds`, continuous rate is position count / rounds,
and conditional rate is position count / preceding position count (rounds for
position 1). Generation timing includes client dispatch/response collection and
raw-response saving, but excludes dataset download, metrics waiting, sanitizing
and code execution. Output throughput uses API `completion_tokens`; if usage
is missing it is reported as null, not estimated from text.

The generator **does not execute generated code or report pass@1**. In an isolated
evaluation container/environment, grade the saved samples separately:

```bash
evalplus.evaluate --dataset humaneval --samples /path/to/run/results/samples.jsonl
```

EvalPlus reports HumanEval and HumanEval+ correctness. A failed generation or
metrics error causes a nonzero script exit and `valid_measurement=false`; preserve
the artifacts for diagnosis and rerun cleanly before comparing throughput/MAL.
In particular, timed-out requests may still be executing server-side. Truncated
or empty-code outputs remain in the grading set and are reported explicitly.

For comparisons, keep prompts, task order, sampling settings, max tokens,
concurrency, client version and server settings fixed. HumanEval has only 164
tasks, so use repeated runs before interpreting small throughput/MAL differences.
