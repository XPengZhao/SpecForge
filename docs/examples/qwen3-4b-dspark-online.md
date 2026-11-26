# Qwen3-4B DSpark online training on eight GPUs

This recipe runs a frozen Qwen3-4B teacher on GPU 0 and DSpark training on
GPUs 1–7. Mooncake buffers a bounded window of features in host DRAM, so a full
hidden-state cache on SSD is unnecessary. The 1+7 split is a starting point,
not a measured throughput guarantee.

## Install and select inputs

Use the same checkout on the Linux GPU server. For CUDA 13.0, install the
repository dependencies and CUDA 13 Mooncake client in the active environment:

```bash
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e . --torch-backend=cu130 --prerelease=allow
uv pip install mooncake-transfer-engine-cuda13
bash scripts/apply_sglang_spec_capture_patch.sh
```

`mooncake_master` must be on PATH. See the
[installation guide](../get_started/installation.md) for environment setup.

DSpark accepts conversation JSONL directly. It does not require EAGLE3's
32,000-token vocabulary mapping or a separate preparation command. Point
`data.train_data_path` at your downloaded file; for example:

```bash
DATA_PATH="$HOME/datasets/open-perfectblend-qwen3-4b-regen/data/train.jsonl"
```

The file must have the conversation format supported by SpecForge, or
preformatted text with `data.is_preformatted=true`. The producer uses the Qwen
template to tokenize, truncate at 4,096 tokens, and construct response loss
masks. The saved responses are teacher-forced inputs; this does not regenerate
answers. Already-tokenized JSONL is also supported through `data.prompts_path`
with `data.train_data_path=`.

## Inspect and run

Run from the repository root. Inspect the plan:

```bash
specforge train \
  -c examples/configs/qwen3-4b-dspark-1server-dp7-disaggregated.yaml \
  data.train_data_path="$DATA_PATH" \
  --plan
```

The plan resolves model configuration, so Hugging Face access or cached model
files are needed. For a local target model, also pass
`model.target_model_path=/path/to/Qwen3-4B`.

Start with 20 optimizer steps and fresh attempt paths:

```bash
ATTEMPT=qwen3-4b-dspark-online-smoke-001
specforge train \
  -c examples/configs/qwen3-4b-dspark-1server-dp7-disaggregated.yaml \
  data.train_data_path="$DATA_PATH" \
  training.max_steps=20 training.save_interval=10 training.log_interval=1 \
  run_id="$ATTEMPT" output_dir="outputs/$ATTEMPT" \
  deployment.disaggregated.control_dir="outputs/$ATTEMPT/control" \
  deployment.disaggregated.consumer_state_dir="outputs/$ATTEMPT/consumer-state"
```

The supervisor owns Mooncake, the patched SGLang teacher, and the producer and
trainer processes. It checks service readiness and stops owned services on
completion or failure. Service logs are in `outputs/$ATTEMPT/control/logs/`.
Each new run requires fresh control and consumer-state paths.

During startup, `[data-prep]` logs in the training terminal identify each
preparation stage with `start`, `done`, or `failed`, elapsed seconds, and sample
counts. Stages cover tokenizer/processor loading, draft configuration, input
inspection, dataset loading, subset selection, shuffling, tokenization/cache
loading, filtering, prompt-task construction, and capture metadata/pipeline
resolution. Tokenization and filtering retain their dataset progress bars.
`Building prompt tasks` reports scanned samples, rate, and ETA when the total
is known; completion logs report accepted and skipped samples. Tokenized JSONL
is streamed with an unknown total to avoid an extra full-file counting pass.

For raw conversation input, `data.prompt_build_num_proc=16` enables ordered
multiprocess task construction after tokenization. This is separate from
`data.build_dataset_num_proc`, which controls tokenization/filtering. The default
is one task-building process. Workers read the processed dataset directly, with
at most two batches of 64 samples per worker queued for transfer to the parent.
The final task list still resides in host RAM. Start with 8–16 workers rather than
using all CPU cores: transferring Python lists and allocating the final list can
limit scaling. Pre-tokenized JSONL retains its serial streaming path.

For example, `stage=materialize_tasks start` without `done` means the producer
is still constructing its task list, before any SGLang capture requests. These
logs are flushed immediately to stderr and are separate from the SGLang service
log. A cache path announces where cached tokenization is read/written; it does
not by itself mean that processing has completed. Progress logs do not change
the samples, cache keys, masks, or task ordering.

For the full run, choose a new attempt name and omit the three `training.*`
overrides above. The recipe limits training to 10,000 optimizer steps over up
to ten passes through the data, recomputing teacher features each pass.
Retained-service recovery uses the external
[DSpark recipe](../../examples/configs/qwen3-4b-dspark-disaggregated.yaml) and
the [recovery contract](../basic_usage/disaggregated_training.md#resume-and-freshness).

## DSpark capture and memory

The draft config is `configs/qwen3-4b-dspark.json`: five draft layers, block
size seven, and target layers `[1, 9, 17, 25, 33]`. DSpark also needs the
target's final hidden state for its training objectives. The SGLang capture
method is named `dflash`; `training.strategy` remains `dspark`. No full
per-token vocabulary logits are sent through Mooncake.

The server contributes a 32 GiB host-memory segment plus a 1 GiB transfer
buffer. Allow extra RAM for tensor copies, prompt preparation, and trainer
prefetch. Production pauses at 128 unacknowledged samples and resumes at 96.
For BF16, six hidden-size-2,560 vectors at 4,096 tokens occupy approximately
120 MiB/sample; 128 samples represent about 15 GiB of feature payload, leaving
headroom for in-flight captures. Keep queue sizes and segment capacity
consistent if changing sequence length or concurrency.

For a two-GPU smoke run, append both overrides:

```bash
deployment.trainer.nproc_per_node=1 \
'deployment.disaggregated.managed_local.trainer_cuda_visible_devices=["1"]'
```

## Muon for DFlash and DSpark

Both Qwen DFlash and DSpark support optional backbone Muon:

```bash
training.dspark_backbone_muon=true \
training.fsdp_sharding=NO_SHARD
```

The option retains its original DSpark name for script compatibility and
defaults to `false`. The managed 1-server DP7 recipe already sets `NO_SHARD`;
other recipes may need that override. Sharded training is not supported by this
Muon path.

The target-feature fusion projection and backbone Q/K/V/O and FFN matrices use
Muon with `adjust_lr_fn="match_rms_adamw"`, matching the existing Muon baseline.
Norms and DSpark's Markov/confidence heads remain on AdamW. Pure DFlash has no
Markov or confidence head. Frozen target parameters are excluded from both
optimizers. Both optimizers share gradient clipping and the learning-rate
schedule.

For pure DFlash, select the matching model and strategy:

```bash
model.draft_model_config=configs/qwen3-4b-dflash.json \
training.strategy=dflash \
training.dspark_backbone_muon=true
```

The DFlash JSON defaults to `block_size=16` (anchor plus 15 mask predictions).
For seven draft predictions, also set `model.draft_block_size=8`. DFlash uses
its own CE objective; DSpark CE/L1/confidence coefficients do not control it.

Checkpoints retain Muon momentum, AdamW moments, the scheduler and FP32 master
weights. Full resume must keep the same optimizer choice. Weights-only
initialization can start a fresh run with either choice; HF export is unchanged.

## Online/offline comparison

The recipe explicitly uses DSpark's original CE/L1/confidence objective,
matching the objective coefficients in the
[offline reference](../../examples/configs/qwen3-4b-dspark-deepspec-offline.yaml).
It does not establish numerical parity. In particular, its default global
batch is seven sequences, whereas the offline reference uses 512.

For a controlled comparison, match the draft initialization, token IDs,
response masks, capture layers, sequence length, batch order, global batch,
optimizer, and schedule. Begin with one training rank and the same small
dataset in both paths. Increasing accumulation requires count watermarks large
enough for a complete optimizer window:
`trainer_world_size * batch_size * accumulation_steps`. A larger queue cannot
compensate for a persistently slower teacher.


### DFlash 与 DSpark 的训练指标

两种策略都按日志窗口累计各个 microbatch 的指标，并跨 DP rank 汇总。
`train/acc` 是窗口内正确 token 总数除以有效 token 总数，不是最后一个
microbatch 的 accuracy，也不是各批次 accuracy 的简单平均。
TensorBoard 中 `train/top1_acc@0` 对应第一个 draft token；DFlash 排除已知
anchor，从第一个 mask 开始编号。`train/position_loss/01_ce` 是该位置的 CE。

`train/ce_loss` 和 `train/loss_weighted` 使用窗口内全局分子、分母汇总，
保留训练的位置衰减权重。诊断模式下的 `train/loss` 沿用 DSpark 的
rank 0 局部 microbatch loss 均值；比较聚合 loss 请用 `train/loss_weighted`。
`tracking.dspark_diagnostics=true` 也适用于 DFlash，可显示
`train/log_micro_batches` 和 `train/accuracy_denom`。
首步输出一次预览，常规日志窗口及最后不足一个窗口的部分遵循同一汇总路径。
这些指标汇总改动不改变训练目标。默认 CE 模式的 DFlash 不计算 target 概率分布，因此不输出
`accept_rate@i` 和 `tau_probabilistic`，也不会以 top-1 accuracy 冒充接受率。


### 纯 DFlash 使用 DSpark 损失

保持 DFlash 的模型配置与 `training.strategy=dflash`，增加：

```yaml
training:
  strategy: dflash
  loss_type: dspark
  dspark_loss_mode: original
  dspark_ce_loss_alpha: 0.1
  dspark_l1_loss_alpha: 0.9
```

该模式复用 DSpark 的 `0.1 * CE + 0.9 * L1`、位置衰减与跨 DP 有效权重归一化。
L1 是 draft/target 词表概率之差的绝对值之和。纯 DFlash 不增加 Markov 或
confidence head，`dspark_confidence_head_alpha` 不参与该模式。
`training.recompute_loss=true` 可启用与 DSpark 相同的损失重计算。
`loss_type=dflash` 仍为默认值，保持原有 hard-label CE；设置 `loss_type=dspark`
后也可以通过 `dspark_loss_mode=kl` 选择共享 KL 实现。

一个 block 仍为 anchor 加 `block_size-1` 个 mask；只有 mask 参与监督。
第一个 mask 预测 anchor+1 的 token，以 target 在 anchor 的最终层 hidden
投影出的分布为监督。有效位置采用 DSpark 的连续前缀规则。

Online capture 会额外请求 target 的最终层 hidden。请用新的运行目录启动，
避免复用旧 CE 任务缺少此特征的在途数据；离线训练需使用含最终层与辅助层 hidden
的 DSpark 格式缓存。该模式会记录 `ce_loss`、`l1_loss`、分位置 top-1、
`accept_rate@i` 和 `tau_probabilistic`，后两者是训练分布重叠统计，不是在线 MAL。

导出仍使用纯 DFlash JSON，推理模型和参数量不变。完整 resume 会检查 loss 模式
和系数，不能把旧 CE 训练状态当作相同目标直接续训；若要接着已有权重训练，
使用 warm start 并开启新实验。
