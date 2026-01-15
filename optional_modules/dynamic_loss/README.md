动态损失模块
==========

文件
----
- dwa.py：Dynamic Weight Averaging 示例。
- gradnorm.py：GradNorm 动态权重实现。

改进建议
--------
- 与 `losses/combined.py` 打通接口，通过 Hydra 切换固定/动态权重。
- 为不同任务记录权重曲线与损失变化，加入单测/日志。
