# Qwen3.8 DSpark：Engram 增强 Markov head

该消融保持 DSpark backbone、MASK 输入和原始 Markov `W1/W2` 不变。预测位置
`i` 时，将前一个已知 token 同位置的 raw Engram 特征投影到 rank-256 Markov
空间：

`m_i = W1(x_{i-1}) + P(RMSNorm(e_ngram(x_<=i-1)))`

`bias_i = W2(m_i)`

`P` 是无 bias 的 `2560 -> 256` 线性层，权重全零初始化，不使用 gate。训练
初始输出严格等于 vanilla Markov，且 `P` 从第一步即可接收梯度。新增参数
`2560 * 256 = 655,360`。

训练使用 full-sequence sidecar 中与前一个 token 同位置的 Engram：Draft 1 使用
anchor 位置，Draft 2 使用第一个 target token 位置，依次类推。绝不能使用待预测
token 同位置的 Engram，否则会泄漏标签。当前标准 loss 使用 teacher-forced 前缀；
OPD 生成前缀尚未接入在线 Engram lookup，因此本配置禁止二者同时启用。

推理时不能复用 teacher-forced sidecar。sampler 每产生一个 token 后，必须基于当前
生成前缀重新调用 Engram lookup，再修正下一个位置的 logits。模型提供了接收
`ngram_lookup(prefix_token_ids)` 的顺序采样接口，部署侧仍需连接 Qwen PLE/Engram
查表实现并测量逐 token lookup 延迟。

配置默认四卡、micro-batch 4、累积 32、global batch 512，并使用稳定性实验确定的
peak LR `3e-4`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m specforge.cli train \
  -c examples/configs/qwen3.8-flash-next-dspark-ngram-markov-offline.yaml \
  2>&1 | tee logs/qwen38-draft3-ngram-markov.log
```

只有 `markov_head_type=ngram` 的模型读取 ngram sidecar；baseline 不增加该 I/O。
新增模块改变 checkpoint 结构，应从头训练，不能完整恢复 vanilla baseline 的
optimizer checkpoint。
