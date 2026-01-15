losses 模块说明
==============

文件功能
--------
- `combined.py`：分割+分类的多任务组合损失（BCE+Dice）。
- `dice.py`：Dice 损失与评估函数。
- `losses.py`：损失聚合工具（基础入口）。
- `combined.zip`：打包文件（可清理，如确认无用）。

改进建议
--------
- 分类 BCE 支持 `pos_weight` / focal，缓解样本不平衡。
- 动态权重：对接 `optional_modules/dynamic_loss`（GradNorm、不确定性权重）。
- 分割可加入 Tversky/Focal-Dice，可配置 foreground 平衡系数。
