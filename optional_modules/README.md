可选模块总览
============

子目录
----
- `attention_modules/`：通道/空间注意力（SE、CBAM）。
- `dynamic_loss/`：动态损失权重（DWA、GradNorm）。
- `lightweight_backbones/`：轻量骨干（MobileNet/EfficientNet 编码器）。

建议用法
--------
- 通过 Hydra 配置开关接入注意力/动态权重，确保可与基线对比。
- 轻量骨干需在 `models/multitask_model.py` 配置化接入。
- 为每个模块补充冒烟测试与消融脚本记录。

待补充
------
- 统一接口（工厂函数）便于按配置创建模块。
- 文档化各模块的预期输入/输出形状与参数。
