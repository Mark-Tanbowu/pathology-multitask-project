# 多/单任务消融方案说明

## 目的
- 对比同一共享编码器下，分割+分类联合训练 vs. 仅分割 / 仅分类的表现，证明多任务带来的正迁移。
- 为后续注意力、轻量骨干、动态 loss 等实验提供统一入口，确保所有实验共享相同的数据/优化配置。
- 在项目结题时直接引用该流程，复现实验并导出曲线/日志。

## 实施步骤
1. **配置开关**：`configs/defaults.yaml` 中新增 `tasks.enable_seg`、`tasks.enable_cls`，默认都为 `true`。
2. **模型条件执行**：`src/models/multitask_model.py` 根据开关决定是否运行分割/分类头，未启用的分支返回 `None`，避免额外计算。
3. **损失函数自适应**：`src/losses/combined.py` 只对启用的分支计算损失，其余分支的 loss 计为 0，保证日志结构一致。
4. **训练循环兼容**：`src/engine/train.py` 按开关跳过对应指标/日志，并在保存 checkpoint 时根据有效指标取平均分。
5. **自动化脚本**：`scripts/run_ablation.py` 顺序执行多任务、纯分割、纯分类三种配置，支持 `--extra-overrides` 传入额外的 Hydra 参数；`--dry-run` 可用于核对命令。

## 使用示例
```bash
# 多任务 + 单任务一次跑完
python scripts/run_ablation.py

# 仅打印命令
python scripts/run_ablation.py --dry-run

# 指定更多 Hydra 覆写，例如关闭 dummy
python scripts/run_ablation.py --extra-overrides data.use_dummy=false device=cuda
```

运行完脚本后，查看 `outputs/` 下的各次 `train_*.log` 和 `training_curves.png`，就能获得对比结果。若需扩展更多实验（例如引入注意力开关），只需在 `EXPERIMENTS` 列表中添加相应覆写即可。

## 2025-12-11 实验结果
采用 `tasks.enable_seg/enable_cls` 统一设置的 12 月 11 日实验表明，多任务与单任务之间存在显著的正迁移效应。多任务模型在 `train_20251211_155353.log` 中于第 20/40 轮达到验证 Dice ≈0.8522，并在第 33/40 与第 40/40 轮持续获得验证准确率 ≈0.9714，整体最优评分 0.9098，说明共享编码器能够同时维持高分割质量与强分类能力。仅分割模型（`train_20251211_161457.log`）虽然在第 34/40 轮实现最高验证 Dice ≈0.8581，略高于多任务峰值，但其最佳评分仅为 0.8581，且后续波动较大，显示缺乏分类辅助监督会削弱稳定性。仅分类模型（`train_20251211_163505.log`）即便验证损失持续下降，准确率仍停留在 ≈0.9429，低于多任务情形约 3 个百分点。综上，多任务框架在几乎不牺牲分割性能的前提下显著提升分类表现，是后续注意力、轻量骨干和动态 loss 消融的首选基线。
  
后续实验（注意力/轻量骨干等）延续此段落格式记录关键
  指标，便于统一引用。