# Qwen3.8 DSpark：pre-anchor Engram 替换 MASK

该消融保持三层 GQA draft、anchor embedding、aux、attention、Markov head 和训练目标不变，
只替换 block 内六个 MASK 位置的输入。对 anchor 位置 `a`：

`[E(anchor), z, z, z, z, z, z]`

`z = RMSNorm(P(g[a-1]))`

其中 `g[a-1]` 是不包含新 anchor 的 raw Engram，`P` 是无 bias 的
`2560 -> 2560` 投影。这里不使用 `E(MASK)`、残差、gate 或 `tanh`。投影采用正常
随机初始化并从第一步接收梯度；新增参数为 `2560 * 2560 = 6,553,600`。

六个位置共享同一输入表示，但仍使用各自的 position id，并在 block 内执行原有双向
attention。`a=0` 或无效 block 的 Engram 输入为零。该实验直接比较内容相关 Engram
表示与固定 MASK embedding，初始前向不再与 baseline 等价。

## 数据与训练

训练读取缓存中的 `extra_features.ngram_embedding`，要求 BF16、full-sequence、与
`input_ids` 同位置对齐，并验证 sidecar 与主缓存的 sample ID、序列长度和字节范围。
无需重新生成 hidden state。

默认配置使用四卡、micro-batch 4、累积 32、global batch 512、peak LR `3e-4`，
训练 2,604 optimizer step，每 200 step 保存：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3.8-flash-next-dspark-ngram-mask-fixed-offline.yaml \
  2>&1 | tee logs/qwen38-draft3-ngram-mask-fixed.log
```

正式训练前可覆盖 `training.max_steps=2` 做短跑。该结构与 gate 版本的 checkpoint
不兼容，应从头训练。推理时需由 host 在 target 计算并行期间提供 pre-anchor Engram；
部署调度尚未接入现有推理入口。
