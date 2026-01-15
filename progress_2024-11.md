# 2024 年 11 月创新点进度纪要

> 汇总时间：2024 年 11 月，依据当前仓库 `main` 分支代码和 `outputs/train_20251204_171726.log` 的实测结果。

## 1. 基线多任务能力
- `src/models/multitask_model.py:24-50` 已实现共享 ResNet18/34 编码器 + U-Net 解码器 + 分类头的分割/分类联合前向，`src/losses/combined.py:18-61` 通过固定权重把分割 BCE+Dice 与分类 BCE 相加。
- `src/engine/train.py:40-255` 提供 Hydra 训练循环，支持真实/ dummy 数据切换、分层采样及日志保存；`tests/test_smoke.py:1-7` 仅做了最小 forward 验证。
- 真实数据在 `seed=234` 实验中达到 `val_dice≈0.85`、`val_acc≈0.97`（`outputs/train_20251204_171726.log:54-101`），证明多任务骨干可用，但尚缺单任务对照和更细指标分析。

## 2. 创新点完成度
| 创新方向 | 现状 | 差距 |
| --- | --- | --- |
| 轻量化骨干 | `optional_modules/lightweight_backbones/*.py` 提供 MobileNet/EfficientNet 编码器样例，但 `configs/defaults.yaml:32-36` 仅允许 `resnet18/34`，训练脚本也未接入。 | 需把轻量编码器注册进 `get_backbone`、提供 Hydra 入口、补最小单元测试与速度/精度对比。 |
| 多任务结构 | 核心模型/损失已跑通，并在实测中显著提升 Dice、Acc。 | 缺少单任务 baseline、真实数据可视化/指标拆解和多任务协同分析。 |
| 注意力机制 | `optional_modules/attention_modules` 有 SE/CBAM block，但 `src/models/segmentation.py:18-72` 未暴露 attention 开关，训练/推理/配置也无对应参数。 | 需设计注入点（encoder/decoder/头部）、Hydra 开关与 ablation pipeline，记录性能波动。 |
| 动态多任务损失 | `optional_modules/dynamic_loss/*.py` 实现 GradNorm/DWA，主训练仍使用固定权重 `MultiTaskLoss`。 | 需要将动态策略封装为 loss/optimizer 插件，并通过日志追踪权重随 epoch 的变化，提供失衡样本 scenaria 测试。 |

## 3. 建议优先级
1. **打通轻量骨干**：在 `src/models/backbone.py` 建 registry，允许 `configs/model.backbone` 选择 `mobilenet_v2`/`efficientnet_b0`，并记录参数量、吞吐和验证指标差异。
2. **注意力模块化**：为 `UNetDecoder` 增加可选 attention block，Hydra 配置中添加 `model.attention={type: none|se|cbam}`，完成至少 baseline vs. CBAM 的消融（含日志/曲线）。
3. **动态 loss 策略**：把 GradNorm/DWA 封装到 `MultiTaskLoss` 或训练循环，记录各任务权重，构造前景稀少的 Dummy 数据以验证稳定性。
4. **多任务增益评估**：新增单任务分支配置，比较分类/分割单独训练 vs. 联合训练的指标，并扩展 `tests/test_smoke.py` 覆盖更多输入/任务组合。

落实以上步骤后，可形成“骨干替换 + 注意力 + 动态损失”的串联实验链，支撑论文中的创新点陈述。
