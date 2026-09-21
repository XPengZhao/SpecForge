# Qwen3.8 DSpark：pre-anchor ngram 增强 MASK

沿用 v1.4 的三层 GQA draft 和 target 去重优化。对 anchor 位置 a，使用同一样本的
raw ngram 特征 g[a-1]，不包含新 anchor；a=0 或无效 block 时注入零。

输入为 `[E(anchor), E(MASK)+delta_1, ..., E(MASK)+delta_6]`，其中
`delta_j = tanh(gate_j) * RMSNorm(Linear(g[a-1]))`，不额外乘缩放系数 s。
共享投影无 bias，投影后的 RMSNorm 无可训练缩放参数，
六个标量 gate 从零初始化，tanh 将有效 gate 限制在 (-1, 1)。首位置 embedding、aux、attention mask、Markov head 和 loss 不变。
新增可训练参数 `2560 × 2560 + 6 = 6,553,606`。零 gate 的首步投影梯度为零；gate 更新后投影开始学习。
零 gate 在相同原有权重下保持 baseline 输出；新模块会影响随机数消耗，不能要求独立随机初始化的两个 run 逐位一致。

## 数据与训练

读取合并缓存中的 `extra_features.ngram_embedding`，要求 BF16、同位置 token 对齐、
`raw_ngram_lookup_concat_before_key_value_projection` 阶段。逐条检查 sidecar 与主索引的 sample ID、
原始长度和字节范围。截断及 padding 与主特征一致，无需重新 dump 或复制缓存。

只有 `dflash_config.ngram_mask=true` 的模型才读取 sidecar；baseline 不增加 ngram I/O。
目前只支持 local_colocated 离线 DSpark。推理时必须单独提供 pre-anchor ngram 特征，
host 查表、传输和 target 计算重叠的调度尚未接入。不能直接部署到现有 baseline 推理入口。

从 SpecForge 根目录运行。配置默认四卡、micro-batch 4、累积 32，global batch 512：

```bash
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3.8-flash-next-dspark-ngram-mask-offline.yaml \
  training.max_steps=2 \
  run_id=qwen38-ngram-mask-smoke \
  output_dir=outputs/qwen38-ngram-mask-smoke \
  2>&1 | tee logs/qwen38-ngram-mask-smoke.log
```

正式训练省略这三个 smoke 覆盖项，使用 YAML 默认的 2604 步预算和独立输出目录。
为比较消融，baseline 也应使用 batch 4 / accumulation 32。
新增参数改变模型结构，不要直接 full-resume baseline 的 optimizer checkpoint；本配置默认从头训练。
新增参数随 draft checkpoint 一起保存，同结构实验可以正常 resume。

本地验证覆盖 sidecar 读取、线程预取、截断/padding、a-1 边界、零 gate 等价性、gate/投影梯度及权重加载。
真实 CUDA/FSDP 短跑和吞吐仍需在服务器验证；增加 ngram 读取和投影会引入额外成本。

本版改变了归一化位置和 gate 语义。与旧版 pre-norm/free-gate 比较时，应使用新的 run/output 目录从头训练，不能将旧 checkpoint 当作同语义 resume。参数形状虽相同，前向行为不同。
