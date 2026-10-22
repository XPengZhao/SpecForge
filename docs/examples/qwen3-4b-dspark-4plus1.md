# Dense DSpark 4+1 late conditioning

Based on `dspark-v1.5`, with no MoE. The auxiliary target layers remain
`[1, 9, 17, 25, 33]`. Existing capture, loss, optimizer, gradient accumulation
and managed-local checkpoint resume are reused.

## Model and position convention

`block_size=7` means one real bonus/anchor and six proposed tokens. Decoder
slot p predicts token p+1: inference uses the FIRST six final-layer outputs;
the seventh output retains the baseline's additional next-token supervision.
The future bank contains decoder-slot states, not sampled tokens.

Before verifying a block beginning at t, the first four layers see only target
features at positions <t and twelve pure MASK slots at t+2 through t+13.
All twelve slots interact bidirectionally. No anchor embedding, current-block
target feature, or acceptance boundary enters this stage.

After accepting r proposals (r=0..6), the new bonus is at b=t+r+1. Select six
bank states at slots b+1 through b+6 (bank offsets r through r+5). The final
layer processes `[adapted bonus embedding, six states]` at positions b..b+6.
Its K/V include target features <b and the entire seven-slot block. Rejected
suffix features and the unavailable target hidden state of the bonus are not
visible. Queries are all seven slots; block attention remains bidirectional.

MASK uses ONE shared trainable hidden-size vector, initialized with the model's
initializer range. RoPE distinguishes positions. There are no bank-index or
acceptance-boundary-specific embeddings. Bonus uses a trainable bias-free linear
adapter because it has not passed through the first four layers. The final
layer is an ordinary dense decoder layer with residuals and FFN, without gating.

## Training

Use the original dense online command with a new run/output/control directory,
and change only:

```bash
model.draft_model_config=configs/qwen3-4b-dspark-4plus1.json
```

For a one-step managed-local smoke run using the existing server paths:

```bash
ATTEMPT="dspark-4plus1-smoke-$(date +%Y%m%d-%H%M%S)" \
  bash scripts/run_qwen3_4b_dspark_online.sh smoke \
  model.draft_model_config=configs/qwen3-4b-dspark-4plus1.json \
  data.build_dataset_num_proc=128 \
  data.prompt_build_num_proc=128 \
  runtime.producer_lease=256
```

For four training GPUs, the previous settings remain:

```bash
training.batch_size=1 \
training.accumulation_steps=128 \
training.learning_rate=3e-4 \
training.total_steps=26160 \
training.max_steps=2616 \
runtime.in_flight_high_watermark=1280 \
runtime.in_flight_low_watermark=1024
```

For every sampled final anchor b, training samples a lag uniformly from 0..7
and sets the old context cutoff to max(b-lag,0). Lag zero (1/8 probability)
trains bootstrap: old cutoff equals b, and bank slots start at b+1, exactly as
in prompt-prefill inference. Positive lags use bank start old_cutoff+2.
Near the sequence start clamping changes the induced boundary distribution;
anchor zero is always bootstrap with an empty context and bank slots 1..12.
Evaluation uses lag 7, clamped
at zero, for a deterministic conservative delayed-context diagnostic. These are
teacher-forced boundary simulations, not sampled from live acceptance lengths.
Real acceptance must be measured with generation.

Every training anchor builds a twelve-state bank; different training anchors
remain attention-isolated. Only one six-state slice is supervised per bank in
each forward. Padding anchors are masked from loss. The bank must not be detached:
all five layers, MASK and bonus adapter train jointly. This increases first-stage
query count relative to ordinary seven-slot DSpark and may cost more training
memory/time. Start with the launcher's smoke mode before a full run.

Start from scratch; a baseline/MoE checkpoint is not an exact resume of this
architecture. Same-architecture resume retains the existing optimizer/LR/data
cursor contract, with an additional late-conditioning semantics marker.
The bootstrap fix changes that marker to v2; exact resume from a v1 run is
intentionally rejected because its boundary sampling distribution differs.

## Export and synchronous inference

```bash
specforge export --to hf \
  --checkpoint /path/to/run-step2616 \
  --draft-config configs/qwen3-4b-dspark-4plus1.json \
  --output-dir /data1/models/dspark-4plus1-step2616

PYTHONPATH=. python scripts/generate_dspark_4plus1.py \
  --target /data1/models/Qwen3-4B \
  --draft /data1/models/dspark-4plus1-step2616 \
  --prompt 'What is 17 times 23?' \
  --max-new-tokens 256
```

The reference runs batch-one greedy generation with the target and draft on the
same device. Target verification uses a KV cache; accepted target features are
retained, rejected suffixes are cropped. Draft context K/V are recomputed each
round for simplicity. Initial prompt prefill has a separate bootstrap bank;
subsequent banks are computed BEFORE verification and reused after its result.
There is no CUDA overlap, multi-loop refinement, stochastic rejection sampling,
or vLLM integration. Random-sampling temperatures are explicitly rejected.

Output includes generated text, verification rounds, accepted/drafted proposal
counts and `mal=1+accepted/rounds`. Terminal EOS/length truncation can shorten
actual emitted output; MAL describes verified rounds, not exact emitted tokens
per round. Reference latency is not a prediction of overlapped serving latency.
