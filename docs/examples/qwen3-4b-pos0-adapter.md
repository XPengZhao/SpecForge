# Qwen3-4B pos-0 adapter

在最后一层 backbone 与原有最终 RMSNorm 之间，仅对每个 block 的第一个预测位置增加：

```text
h → RMSNorm → gate/up → SiLU(gate) * up → down → + h → 原有最终 RMSNorm
```

`down` 零初始化，初始函数与原模型一致；第一次更新后其他新增参数开始收到梯度。
DSpark 的第一个预测位置是 block slot 0；纯 DFlash 是 slot 1（第一个 mask）。
其他位置不经过 adapter。原有 CE/L1、位置衰减与 Markov/confidence 路径保持不变，
不添加额外损失，也不截断传向 backbone 的梯度。

## 配置

DSpark 使用 `configs/qwen3-4b-dspark-pos0-adapter.json`，保留 7 个预测位置。
纯 DFlash 使用 `configs/qwen3-4b-dflash-pos0-adapter.json`，为 anchor + 7 masks。
替换原训练命令的配置路径即可，其余超参数沿用对照组：

```bash
model.draft_model_config=configs/qwen3-4b-dspark-pos0-adapter.json
```

开关在模型 JSON 的 `dflash_config` 中：

```json
"pos0_adapter_intermediate_size": 256
```

省略或设为 0 时关闭。2560 hidden、256 内部宽度共增加
`3 × 2560 × 256 + 2560 = 1,968,640` 个参数，其中最后一项是 RMSNorm 缩放。
启用 `training.dspark_backbone_muon=true` 时，三个投影矩阵使用 Muon，norm 使用 AdamW；
未启用 Muon 时均使用原有 AdamW。

该配置用于从头训练，完整 resume 使用同一模型配置。普通 baseline checkpoint
缺少 adapter 参数，当前严格加载不会自动补齐它们。

## 导出与推理

HF 导出使用训练时同一个 JSON，adapter 参数和配置随模型保存：

```bash
specforge export --to hf \
  --checkpoint "$CHECKPOINT" \
  --draft-config configs/qwen3-4b-dspark-pos0-adapter.json \
  --output-dir "$EXPORT_DIR"
```

本分支包含 SpecForge 模型与 HF 导出支持，尚未适配 vLLM。新权重需要配套推理代码，
不能用原 baseline 推理分支忽略 adapter 评测。
