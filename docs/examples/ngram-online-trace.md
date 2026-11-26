# 在线 n-gram 误差采集

从**实际训练子集**固定随机抽取 512 个不同 prompt，并发 32，在 vLLM 的真实 draft/verify 路径统计。多次 rollout 的相同 prompt 只参与一次抽样；保存源文件 SHA256、行号、prompt SHA256 和请求 ID。

## 一键运行（推荐）

在 SpecForge 目录执行，无需手动设置采集环境变量或另起服务：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ngram_diagnostics.py \
  --dataset regen \
  --checkpoint "$HOME/models/qwen3-4b/你的Engram导出目录" \
  --data-path "$HOME/datasets/实际训练子集/train.jsonl" \
  --target /data1/models/Qwen3-4B \
  --vllm-dir "$HOME/vllm"
```

默认：probabilistic draft、T=0.7、top-p=0.8、top-k=20、7 个 draft token、512 个不同 prompt、并发 32、端口 11443。需要带 on/off 采集改动的 vLLM 分支。评测和汇总默认使用 vLLM 的 `.venv/bin/python`，可用 `--python` 指定；该环境需要 aiohttp 和 numpy。

### 切换 sweep 数据集

`--dataset` 支持 `gsm8k`、`humaneval`、`mtbench`、`math500`、`aime`、`regen`。例如：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ngram_diagnostics.py \
  --dataset mtbench \
  --checkpoint "$HOME/models/qwen3-4b/你的Engram导出目录" \
  --target /data1/models/Qwen3-4B \
  --vllm-dir "$HOME/vllm" \
  --data-dir "$HOME/datasets/dspark-eval" \
  --temperature 0.7 --top-p 0.8 --top-k 20
```

| 数据集 | 默认题数 | `--data-dir` 下的路径 |
| --- | ---: | --- |
| gsm8k | 1319 | `gsm8k/test.jsonl`，few-shot 时还需 `gsm8k/train.jsonl` |
| humaneval | 164 | `humaneval.jsonl` |
| mtbench | 80（两轮共 160 个请求） | `mtbench.jsonl` |
| math500 | 500 | `math500.jsonl` |
| aime | 30 | `aime.jsonl` |
| regen | 512 个不同 prompt | `regen.jsonl` |

也可以用 `--data-path` 直接指定文件。GSM8K 在 `--num-shots 0` 时可直接读 test JSONL；few-shot 时该参数须指向含 train/test 的目录。`--data-path` 和 `--data-dir` 二选一。除 regen 必须提供数据路径外，其余数据集省略路径时沿用原评测器的数据下载/缓存。

benchmark 复用 sweep 的原始评测器，按原顺序取前 `--num-prompts N` 题，默认跑全量；只有 regen 使用 `--sample-seed` 固定随机抽样。GSM8K 默认 0-shot，可以设置 `--num-shots`；HumanEval 仍保存样本供 EvalPlus 后续评分，不在这里执行生成代码；MATH-500/AIME 仍进行 Math-Verify 评分；MT-Bench 仍用第一轮实际生成回答构造第二轮上下文，并输出 overall 接受率。

客户端只向实际请求增加 `request_id`，不重写 prompt。benchmark 的 `eval/trace_manifest.json` 和 `trace_requests.jsonl` 保存映射与请求内容；MT-Bench 两轮 ID 独立、prompt 家族相同，贡献 CSV 额外包含 `scope=turn` 的逐轮逐位置统计。两轮同一题在残差分析中始终分在同一个 A/B 组。

依赖与原评测器一致：HumanEval 需要 evalplus；MATH-500/AIME 需要 math-verify，在线加载还需 datasets；GSM8K 使用指定 vLLM 仓库内的 evaluator（requests、regex、tqdm）。

脚本沿用 sweep 的服务启动、健康检查和进程清理逻辑，自动开启 V2 runner、paired tracing，保留 CUDA Graph。评测结束先停止自己启动的服务，再汇总完整 trace。端口已占用时直接报错；失败和中断都会清理本次服务，保留日志。服务默认名称为 `qwen3-4b`。

输出在 `$HOME/outputs/ngram-diagnostics/ngram-随机后缀/`：`result.json` 汇总 MAL、接受率和贡献；`summary/contributions.csv` 是分组贡献；`eval/` 保存 prompt/响应；`trace/` 保存服务端记录；`serve.log`、`eval.log`、`summary.log` 保存各阶段日志，`run.json` 和 `serve.json` 保存命令及参数。可以通过 `--output-root` 改父目录，每次运行创建新目录。

采样参数可直接追加，例如 `--temperature 1 --top-p 1 --top-k -1`；其他选项见 `--help`。下面的步骤适合需要手动管理服务的情况。

## 启动服务

使用带采集改动的 vLLM `codex/dspark-v1.5-markov-engram` 分支（也支持 baseline DSpark）。在原来的 `vllm serve` 命令前设置：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NGRAM_TRACE_DIR="$HOME/outputs/ngram-trace/run01"
export VLLM_NGRAM_TRACE_PREFIX="ngram-run01-"
export VLLM_NGRAM_TRACE_TOPK=32
export VLLM_NGRAM_TRACE_MAX_EVENTS=2000000
export VLLM_NGRAM_TRACE_PAIRED=1
# 然后执行原来的 vllm serve 命令，保持 CUDA Graph 开启。
```

每次运行使用新的目录和 prefix。当前支持完整词表、固定 draft 长度、普通 rejection sampling 的 DSpark；不支持裁剪词表、draft_topk、自适应验证、grammar 和 watermark。旧 runner 会明确报错。

`PAIRED=1` 要求加载 Engram checkpoint。在原 speculative-config 中设置 `"draft_sample_method":"probabilistic"`，客户端设置正 temperature，才能观察概率采样的分布改善。只设置客户端 temperature 不会把 greedy draft 改成 probabilistic。baseline 或原来的单路残差采集使用 `VLLM_NGRAM_TRACE_PAIRED=0`（默认）。

## 发起评测

在 SpecForge 目录执行，服务应只接收本次评测请求，以免 acceptance 全局计数混入其他流量：

```bash
export EVAL_DATA_PATH="$HOME/datasets/actual-training-subset.jsonl"
export TRACE_PREFIX="ngram-run01-"
export EVAL_HOST="http://127.0.0.1"
export EVAL_PORT=8000
export EVAL_MODEL="qwen3-4b"
export EVAL_TEMPERATURE=0.7
export EVAL_TOP_P=0.8
export EVAL_TOP_K=20
bash scripts/run_ngram_trace_with_acceptance.sh ngram-run01
```

默认 sample seed=42、generation seed=42、512 个不同 prompt、并发 32、max_tokens=4096。可通过 `EVAL_*` 覆盖；完整参数见 `scripts/eval_ngram_trace_vllm.py --trace-prefix ngram-run01- --help`。客户端需要 aiohttp。

客户端输出 `outputs/eval/ngram-run01/`，包括 prompt 清单、响应、MAL、各位置接受率。服务端单独输出 `trace-*.jsonl`，记录 token ID 和概率信息。文件可能较大。

```bash
python scripts/summarize_ngram_trace.py \
  --eval-dir outputs/eval/ngram-run01 \
  --trace-dir "$HOME/outputs/ngram-trace/run01" \
  --output-dir outputs/eval/ngram-run01-summary
```

`results.csv` 按 n-gram 和 draft 位置统计出现次数、覆盖 prompt 数、条件接受率、首次拒绝次数、平均完整词表 L1。只汇总验证实际到达的位置；首次拒绝之后的轨迹保留在原始文件里，不混入接受率。位置从 0 开始。

## Engram on/off 贡献（主要结果）

开启 paired 后，每个位置旁路计算两份 logits：

```text
off = backbone_logits + MarkovBias(W1[token])
on  = backbone_logits + MarkovBias(W1[token] + Engram_correction)
```

生成一直使用 on；off 使用同一个 hidden 和 on 路径的历史 token，不额外采样。scalar gate、vector gate 都属于被移除的 Engram 增量。联合训练模型的 off 是同一 checkpoint 的原始 Markov 通路，不等同于另一个独立训练的 baseline。

`p` 使用验证器处理后的 target 分布（包括 temperature、top-p/top-k 等）；`q` 遵循当前 DSpark proposal 实现，只应用 temperature。每个位置计算完整词表重叠 `A = sum(min(p, q))`，而非 T=1 的原始 L1。greedy draft 的 q 是单点分布；temperature=0 时 A 退化为 top-1 是否相同。

`summary.json` 的 `contributions.overall` 给出整体结果；`contributions.csv` 按 overall、position、request、ngram 分组，分别输出 reached 和 after_rejection 两类。主要看 reached：

| 字段 | 含义 |
| --- | --- |
| `mean_a_on` / `mean_a_off` | 当前上下文上，两种分布的期望单步接受概率 |
| `mean_delta_a` | on 减 off 的平均净收益 |
| `g_plus` / `g_minus` | 正向收益与负向损失，均除以该组全部事件数 |
| `sum_positive_gain` / `sum_negative_loss` | 正负贡献总量，帮助定位主要贡献来源 |
| `oracle_a` | 每个位置理想选择 on/off 时的局部接受概率 |
| `oracle_gain_over_on` | 理想开关相对始终 on 的改善空间，等于 `g_minus` |
| `improved_events` / `harmed_events` | ΔA 大于 1e-7 / 小于 −1e-7 的事件数 |

`g_plus - g_minus = mean_delta_a`。整体按事件加权，长回答权重更大；request 行可查看收益是否集中在少数请求。ngram 行保留位置，并按 `--min-count` 和 `--min-prompts` 过滤，所有原始累计仍保存在 `contributions.sqlite`。2-gram/3-gram 分组都是**完整 Engram 增量**的贡献标签，不能据此分别归因到两张表，也不要将两个阶数的贡献相加。

oracle 只在当前 on 轨迹上选择开或关，不包括缩放/反转修正，更不是全局最优 MAL。贡献统计一轮即可完成，不需要下节的二次候选采集。旧 trace 没有 off logits，无法补算；混合新旧事件会报错。

### n-gram 的事件级方向汇总

主表 `summary/ngram_directions.csv` 按 **n-gram × cohort** 汇总所有 draft 位置和轮次。先看 `cohort=reached`。不要求跨 prompt 出现；同一回答中反复出现的组合也可分类，prompt 覆盖数只作参考。

一键脚本自动输出：

- `ngram_directions.csv`：主表，包含分类、事件正负比例、正负幅度、净收益和辅助 prompt 比例。
- `ngram_direction_summary.csv`：每类 n-gram 分组的数量，reached / after_rejection 分开。
- `ngram_prompt_contributions.csv`：按位置、轮次、prompt 拆分的原始累计明细。
- `ngram_position_turn_contributions.csv`：按位置、轮次汇总的辅助表，用于解释正负差异。

分类概览也写入 `result.json` 和 `summary.json` 的 `contributions.ngram_directions`。默认只要求 `--min-count 10` 个事件，使用 `--direction-epsilon 0.001`（**0.1 个百分点**）区分有意义的正负变化。所有事件都计入比例分母，包括中性事件。规则按下表顺序应用：

| 分类 | 判定 |
| --- | --- |
| 样本不足 | 事件数少于 min-count |
| 作用弱 | 平均绝对 ΔA ≤ epsilon |
| 正负混合 | 正、负事件各至少 2 个，且各占全部事件至少 20% |
| 多数有益 | 至少 70% 事件为正，且平均 ΔA > epsilon |
| 多数有害 | 至少 70% 事件为负，且平均 ΔA < −epsilon |
| 倾向不明确 | 其余情况，例如多数事件小幅改善，少数事件大幅伤害 |

“正负混合”优先于多数类，只描述事件表现，不预先归因于上下文。分类不是显著性结论。`positive_event_fraction` / `negative_event_fraction` 看发生比例；`g_plus` / `g_minus` 看幅度；`prompts`、prompt 正负比例和等权平均仅作辅助，且 MT-Bench 两轮同一题只计一个 prompt 家族。

`coverage_by_order` 还给出样本充分的组合覆盖了多少事件、正向贡献、负向损失；2-gram 和 3-gram 分别计算，避免重复计数。旧参数 `--direction-min-prompt-events`、`--direction-min-prompts` 为兼容旧命令仍接受，但不再控制方向分类。`--min-prompts` 只影响旧的贡献/残差表筛选，不影响方向主表。

已有 paired trace 可直接用新版 `summarize_ngram_trace.py` 汇总到新目录，无需重新推理。重新运行一键脚本也会自动使用新规则。方向指完整 Engram 增量的接受收益，不是逐 token 概率向量的方向；按 2-gram 分组的变化也可能来自不同 3-gram、哈希碰撞或上下文。

## 可选：判断剩余误差的方向是否稳定

第一轮的 `candidates.jsonl` 从 prompt 分组 A 中选择残差候选 token。要检查这些 token 的偏差在其他 prompt 上是否一致，用该文件启动第二轮服务：

```bash
export VLLM_NGRAM_TRACE_DIR="$HOME/outputs/ngram-trace/run02"
export VLLM_NGRAM_TRACE_PREFIX="ngram-run02-"
export VLLM_NGRAM_TRACE_CANDIDATES="/absolute/path/to/ngram-run01-summary/candidates.jsonl"
```

使用同一个数据文件、sample seed 和采样参数，客户端 prefix 改为 `ngram-run02-`，输出到新目录。汇总第二轮后，`results.jsonl` 包含 A/B 两组的固定候选平均残差及 `selected_consistency`：固定候选上平均残差的 L1，除以完整词表的平均 L1。该值衡量固定候选可解释的稳定偏差比例；第一轮稀疏 top-K 不用于计算它。样本少的 n-gram 可能只出现在一组。

## 采集与 CUDA Graph

单路采集在 draft 阶段增加固定缓冲区复制；paired 还增加一次原始 Markov 投影和 off logits 的复制，均随 draft CUDA Graph 捕获和重放。verify 之后，在 Graph 外做统计和文件写入；不修改 on logits、采样概率或随机数状态。请求重排使用持久 slot 映射，on/off 缓存各自用绝对位置检查是否过期。

原始分布残差使用 T=1 的完整 softmax；实际验证概率另存，包含本次 temperature 和采样处理。二者含义不同。

每份 logits 缓冲区约占 `max_num_seqs × draft长度 × vocab_size × head_dtype字节数`；32×7×151936 的 BF16 约 65 MiB，paired 两份约 130 MiB，FP32 再翻倍。缓冲区跟随实际 head dtype，避免诊断时降精度。CPU 同步、额外投影、完整 softmax 和写文件会降低速度，这次运行用于诊断，不用于吞吐比较。本地有 CPU 单测和可在 GPU 执行的 Graph 重放测试；真实 CUDA Graph 尚需服务器验证。
