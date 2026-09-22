# Qwen3.8-Flash-Next：三层 aux 离线 DSpark

使用 DeepSpec 已完成合并的 v2 缓存，训练三层 dense GQA DSpark draft。
仅加载 target 的冻结 embedding 和 LM head，不实例化 Qwen4-Exp target 本体；
因此训练环境不需要 Transformers 的 Qwen4-Exp 建模实现。

## 特征约定

- Aux 为零基 decoder 层 `[45,46,47]` 的输出。每层四路 HC 在 FP32 中平均后转 BF16，拼接宽度为 `3 × 2560 = 7680`。
- 监督特征为最终 `hyper_connection_mixer` 输出，宽度 2560；不能替换为四路平均，也不额外施加 target norm。
- 缓存包含 prompt + response，最长 4096，训练监督默认 response-only。
- `input_ids`、aux 和最终 hidden 按相同 token 位置对齐。沿用现有 DSpark 标签偏移与 attention mask。
- 本实验不使用 ngram/PLE sidecar；缓存中可以保留它们。

启动时校验上述缓存元数据、target 标识及维度。元数据校验不能替代采集后 HF/SGLang 特征数值对照。
请使用合并完成、含 `manifest.json` 和 `samples.idx` 的根目录，不使用 `_workers/worker-*`。

## 生成配置

在 SpecForge 根目录、训练环境内运行：

```bash
PYTHONPATH=. python scripts/prepare_qwen38_dspark.py \
  --target-model-path /public/llm_models/Qwen/Qwen3.8-Flash-Next \
  --hidden-states-path /public/workspace/dspark/cache/deepspec/qwen38_flash_next \
  --output-dir outputs/qwen38-setup
```

生成 `draft_config.json` 和 `train.yaml`。脚本从实际 checkpoint 读取词表、token ID，
检查 embedding/head 的 safetensors 形状。优先用 tokenizer 中的 `<|fim_pad|>`，其次 `<|mask|>`；
若两者均不存在，需通过 `--mask-token-id` 指定词表中已有的占位 token，不扩充词表。
目标路径须与缓存中的 `target_model_name_or_path` 一致。不会覆盖已存在的配置。

Draft 使用 3 层、hidden 2560、FFN 9728；GQA 头配置对齐 target 的 full attention：24 Q / 2 KV、head dim 256、
block size 7、Markov rank 256。Draft 使用完整 RoPE、theta=1e6；不复制 target 的 hybrid attention、HC 或 PLE。
`model_type=qwen3` 描述的是 draft 的配置类，不代表 target 类型。

默认 4 GPU × micro batch 1 × 累积 128 = global batch 512。
令 `S = floor(缓存样本数 / 512)`，训练上限为 S optimizer steps（约一个 epoch），
学习率计划为 10S steps，warmup 4%，峰值 6e-4。日志每 10 步，checkpoint 每 S 步。
可用 `--world-size` 和 `--global-batch` 调整生成配置；可见 GPU 数须与 world size 一致。

## 运行

先在独立输出目录短跑，检查损失、梯度和峰值显存：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c outputs/qwen38-setup/train.yaml \
  training.max_steps=2 training.num_anchors=8 \
  run_id=qwen38-aux3-smoke output_dir=outputs/qwen38-aux3-smoke
```

正式训练恢复每样本最多 512 anchors：

```bash
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c outputs/qwen38-setup/train.yaml \
  2>&1 | tee logs/qwen38-aux3-train.log
```

TensorBoard 写入正式训练输出目录的 `runs/`。短跑降低了 anchor 数，不能据此保证正式训练显存充足。
本适配覆盖离线训练；Qwen3.8 target 与 draft 的在线推理集成需单独实现和验证。

MASK 替换消融见 [pre-anchor Engram 训练说明](qwen38-dspark-ngram-mask-fixed.md)。
