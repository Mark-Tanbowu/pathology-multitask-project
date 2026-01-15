src 说明
========

目录结构
--------
- `engine/`：训练、验证、推理循环。
- `datasets/`：数据集、采样器、增强。
- `models/`：编码器、解码器、分类头、多任务封装。
- `losses/`：多任务/分割损失。
- `optional_modules/`：可选的注意力、轻量骨干、动态损失。
- `utils/`：工具函数、指标、可视化。
- `preprocessing/`：WSI 预处理/切块。
- `lightning/`：Lightning 入口占位（当前空）。

使用提示
--------
- 推荐从 `engine/train.py` / `engine/infer.py` 理解训练/推理管线。
- 模型结构由 `models/multitask_model.py` 驱动，可通过 Hydra 配置切换 backbone/分支开关。
- 数据管线入口在 `datasets/camelyon_dataset.py`（真实数据）与 `datasets/dummy_dataset.py`（冒烟）。

改进建议
--------
- 补充 AMP、梯度裁剪、分层采样随机化，减少 cls_loss 尖峰。
- 将可选模块（注意力、轻量骨干、动态损失）配置化并加入单测/冒烟。
- `lightning/` 目录可接入 LightningModule 以便分布式/混合精度。
