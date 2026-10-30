# Qwen3-4B：GQA backbone + Markov MHA

Backbone 沿用 `qwen3-4b-dspark.json` 的 5 层 GQA。仅将 vanilla Markov
head 改为 `W1 → MHA → W2`：Q 来自前驱 token 的 256 维 embedding，
K/V 来自已有的 target context `C = RMSNorm(FC(target features))`。

Markov head 使用 8 个 Q heads、8 个 KV heads、head dimension 128，
带可训练的 per-head Q/K RMSNorm，并使用 backbone 相同的 RoPE 配置。
输出投影回 256 维；无 residual、FFN 或额外 output norm。
W1、W2 和 confidence head 的定义不变。

复用 GQA 实现，Q/KV 头数相等时即为 MHA；分支、文件名及配置类型仍保留
`markov-gqa` / `gqa` 命名。Backbone 仍为 32 Q / 8 KV。

每个 block 的所有步骤只读取严格早于 anchor 的 target context。
slot `i` 使用 query position `anchor+i`，预测 token `anchor+i+1`。
K/V 不包含 anchor、MASK 或块内生成的 token。空历史时 Markov bias 为零。
head 有独立的 K/V 投影；训练中每次 forward 只投影一次上下文，供所有 anchors 复用。

相对 vanilla 新增 **5,767,424** 个参数（含 Q/K RMSNorm），BF16 约
**11.53 MB**。推理以 BF16 存储 8 个 KV heads 时，每个历史 token 增加
**4 KiB** cache；这里不代表训练的峰值显存。

## 训练

使用分支 `codex/dspark-v1.5-markov-gqa`。GPU 0 负责 target capture，
GPU 1–4 训练；有效 batch 为 `4 × 1 × 128 = 512`。
学习率 `3e-4`，CE/L1/confidence 系数为 `0.1/0.9/1.0`。
完整运行 2616 optimizer steps，学习率调度总步数为 26160，每 654 步保存。
是否恰好一个 epoch 取决于处理后的有效样本数。

先验证一次完整训练和保存：

```bash
SPECFORGE_PATH="$HOME/SpecForge" \
TARGET_MODEL_PATH="/data1/models/Qwen3-4B" \
DATA_PATH="$HOME/datasets/open-perfectblend-qwen3-4b-regen/data/train.jsonl" \
bash scripts/run_qwen3_4b_dspark_markov_gqa_online.sh smoke
```

将 `smoke` 换成 `train` 开始正式训练。脚本末尾接受配置覆盖，例如
`training.save_interval=654`。输出目录由 `OUTPUT_PATH` 和 `ATTEMPT` 控制。
默认使用 Flex Attention；可用 `training.attention_backend=sdpa` 做小规模排查。

模型配置为 `configs/qwen3-4b-dspark-markov-gqa.json`，部署配置为
`examples/configs/qwen3-4b-dspark-markov-gqa-online.yaml`。
新旧 head 的结构不同，不能把 vanilla/MLA checkpoint 当作本实验的完整训练恢复点。

## 导出

继续使用已有 HF 导出命令：

```bash
specforge export --to hf \
  --checkpoint /path/to/run-step2616 \
  --draft-config configs/qwen3-4b-dspark-markov-gqa.json \
  --output-dir /path/to/markov-gqa-hf
```

导出保留 GQA 配置、四个投影及 Q/K RMSNorm 权重；缺少必要 head 权重会报错。
本次只实现训练端及 Python 参考生成路径，vLLM 的 Markov GQA 支持需另行实现。
