engine 模块说明
==============

文件功能
--------
- `train.py`：Hydra 驱动的训练循环，支持 dummy/真实数据、分层采样、日志与 best.ckpt 保存。
- `infer.py`：推理脚本，读取模型输出分割 mask/分类结果。
- `validate.py`：独立验证/评测入口。
- `engine.zip`：打包文件（可清理，如确认无用）。
- `.hydra/`、`outputs/`：局部运行产生的配置与日志，通常已忽略。

改进建议
--------
- 验证指标按样本数加权，避免末尾小 batch 放大。
- 训练增加梯度裁剪、cls logits clamp、weight decay 与 LR 调度。
- 支持 AMP、`pin_memory`、`non_blocking` 以提升效率。
- 记录高 cls_loss 样本列表到独立日志，便于复现异常。
