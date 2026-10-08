# Qwen3.8 DSpark 固定前缀对照

这个对照固定输入 token ID、anchor 位置和 vLLM 实际生成的 anchor token。在线保存第一轮 draft block，再离线用同一导出模型回放。默认采集两条缓存序列，每条三个 anchor。

需要把 `scripts/install_dspark_parity_hook.py`、`scripts/dspark_parity_debug.py` 和 `scripts/compare_dspark_fixed_prefix.py` 同步到服务器的 SpecForge。离线环境需要已有的 PyTorch、Transformers 和 SpecForge 依赖。比较 reranker 时，SpecForge 必须使用支持该 checkpoint 的 reranker 分支。

## 1. 安装采集 hook

在 SpecForge 目录，用普通 Python 安装到运行服务所用的 vLLM 源码目录：

```bash
python scripts/install_dspark_parity_hook.py \
  --vllm-root /public/workspace/dspark/vllm-env/vllm
```

安装会备份原文件。未设置 `DSPARK_PARITY_DIR` 时，hook 不采集数据。

## 2. 启动诊断服务

在原来的 vLLM 启动命令前加：

```bash
export DSPARK_PARITY_DIR=/public/workspace/dspark/logs/dspark-parity-traces
```

并在原启动命令中加以下参数，然后重启服务：

```bash
--enforce-eager \
--no-enable-prefix-caching \
--no-enable-chunked-prefill \
--max-num-batched-tokens 8192
```

保持原 Target、draft 导出路径、TP=4、7 个 draft tokens、greedy 和关闭 adaptive verification。诊断期间单独运行这组请求。关闭 thinking 的服务可以继续保留原来的默认设置；本次直接发送缓存 token ID，不经过 chat template。

## 3. 采集六个 block

在 SpecForge 的训练环境运行：

```bash
PYTHONPATH=.:${PYTHONPATH:-} python scripts/compare_dspark_fixed_prefix.py collect \
  --cache /public/workspace/dspark/cache/deepspec/qwen38_gsm8k_eval256 \
  --dump-dir /public/workspace/dspark/logs/dspark-parity-traces \
  --output /public/workspace/dspark/logs/dspark-parity-baseline \
  --url http://127.0.0.1:8000 \
  --records 2 --anchors 3
```

输出目录必须是新目录。脚本只挑连续八个 token 都有 loss mask 的 anchor；发送 anchor 之前的完整前缀。在线 Target 新选出的 anchor 可能与缓存参考 token 不同，离线会使用在线实际 token。

## 4. 离线回放

采集结束后，可以停止诊断服务释放显存。在 SpecForge 环境运行以下命令。`--checkpoint` 必须与刚才服务实际加载的 draft 路径一致；下面使用你提供过的 baseline 路径。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/compare_dspark_fixed_prefix.py compare \
  --cache /public/workspace/dspark/cache/deepspec/qwen38_gsm8k_eval256 \
  --cases /public/workspace/dspark/logs/dspark-parity-baseline/cases.jsonl \
  --checkpoint /public/workspace/dspark/exports/qwen38-baseline-lr3e4-step7812 \
  --target /public/llm_models/Qwen/Qwen3.8-Flash-Next
```

只加载 draft 和 Target embedding/LM head，不加载完整 Target。输出 `comparison.json`：

- `aux_difference`：缓存 aux 与在线 aux 的相对 L2、最大绝对误差和余弦相似度。
- `hidden_with_cache_aux`：用缓存 aux 回放后的 draft hidden 与在线 hidden 的差异。
- `hidden_with_online_aux`：用在线 aux 回放后的差异。
- `cache_aux_token_matches` / `online_aux_token_matches`：各位置 draft token 是否与 vLLM 一致。

如果在线 aux 回放能复现 vLLM，而缓存 aux 回放不能，重点追查缓存特征。如果两份回放都无法复现，重点核对 draft 的权重加载、attention 和选词实现。六个 block 都吻合，则扩大样本，再检查在线生成前缀和离线参考前缀的差别。小幅浮点误差也可能改变相近 logits 的 argmax，不能只凭一个 token 不同判定实现错误。

这项对照使用独立请求的完整 prefill，尚未覆盖持续解码中的 KV 状态、并发调度和 CUDA graph。因此它是定位的第一步，吻合不等于整个在线链路已验证。

## 5. 恢复

```bash
python scripts/install_dspark_parity_hook.py \
  --vllm-root /public/workspace/dspark/vllm-env/vllm --restore
unset DSPARK_PARITY_DIR
```

再按原命令启动服务。若安装后手工改过该 vLLM 文件，恢复工具会拒绝覆盖，需检查备份后处理。

工具在本地完成静态及安装恢复检查；GPU 回放和服务器采集尚未实际执行。

## 已有差异 block：分数与接受长度

更新服务器上的 `scripts/compare_dspark_fixed_prefix.py` 即可，已有 trace 不需要重采。

先停止诊断服务，释放显存，运行 baseline 分数检查：

```bash
PARITY_DIR=/public/workspace/dspark/logs/dspark-parity-baseline-32
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/compare_dspark_fixed_prefix.py inspect-scores \
  --comparison "$PARITY_DIR/comparison.json" \
  --cases "$PARITY_DIR/cases.jsonl" \
  --checkpoint /public/workspace/dspark/exports/qwen38-baseline-lr3e4-step7812 \
  --target /public/llm_models/Qwen/Qwen3.8-Flash-Next \
  --output "$PARITY_DIR/scores.json"
```

默认只检查出现选词差异的 block，每条候选路径只看首次分歧位置，此前的 Markov 前驱相同。输出保存在线 hidden 经离线 LM head + Markov 重建的 top5 分数、top1/top2 分差，以及在线选词与离线候选的分差。它不包含原始 vLLM logits，也不包含缓存 aux 路径的原始 logits；重建 argmax 不同可提示继续检查 head 数值运算，单凭此项不能证明 bug。本子命令只支持当前 vanilla baseline，不支持 reranker。

之后启动不带 draft 的 Target 服务：沿用诊断服务原来的 Target、TP、dtype、eager 和缓存设置，去掉整个 `--speculative-config`，先 `unset DSPARK_PARITY_DIR`，增加 `--generation-config vllm`。服务模型名仍为 `qwen38-flash-next`，端口仍为 8000。

在训练环境中运行验证（不需要本地 GPU）：

```bash
PARITY_DIR=/public/workspace/dspark/logs/dspark-parity-baseline-32
PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/compare_dspark_fixed_prefix.py verify-target \
  --comparison "$PARITY_DIR/comparison.json" \
  --cases "$PARITY_DIR/cases.jsonl" \
  --url http://127.0.0.1:8000 \
  --all-blocks \
  --output "$PARITY_DIR/target-verification.json"
```

脚本发送相同前缀加实际 anchor，令 Target greedy 生成 7 个 token。比较三条 draft 路径与参考续写的连续匹配长度；这等价于相同 Target 条件分布下 greedy 验证的接受长度，不需要继续验证首次拒绝之后的候选。`accepted_draft_tokens` 不含 bonus，`acceptance_length_with_bonus` 加 1。

`cache_minus_vllm` 大于零表示缓存路径在这次 Target 回放中接受更多，负数表示更少。总平均在 `summary.mean_cache_minus_vllm` 中；指定 `--all-blocks` 才包含整个 96 block 样本。该平均不是原 GSM8K 在线 MAL：这里是参考前缀上的独立 Target prefill，持续解码的状态和验证 kernel 尚未复现。若要确认原在线轮次的实际接受长度，需要另行记录当时的 Target 验证输出。

新增子命令已完成语法和连续前缀计算的本地检查，GPU 分数重建和服务端验证需在服务器实际运行。
