项目结构与复现指引（中文）
========================

概览
----
- 目标：病理切片多任务模型（分割+分类），共享编码器 + U-Net 解码器 + 分类头；可选轻量骨干、注意力、动态损失。
- 推荐阅读顺序：`README.md`（快速起步）→ 本文（模块地图）→ 各子目录 README 了解细节。

目录与作用
---------
- `configs/`：Hydra 配置，`defaults.yaml` 为主入口，`model_unet.yaml` 为模型可选配置。
- `scripts/`：训练/推理/消融入口，`run_train.py`、`run_infer.py`、`run_ablation.py`、`train.sh/ps1` 等。
- `src/`：核心代码  
  - `engine/` 训练、验证、推理循环；  
  - `datasets/` 数据集定义、采样、增强；  
  - `models/` 主干、解码器、分类头与多任务封装；  
  - `losses/` 多任务与 Dice 等损失；  
  - `optional_modules/` 轻量骨干、注意力、动态损失实验模块；  
  - `utils/` 通用工具（指标、随机种子、可视化）；  
  - `preprocessing/` WSI 切块脚本；  
  - `lightning/` 预留 Lightning 入口（当前空壳）。
- `tests/`：pytest 冒烟与 Dice 单测。
- `notebooks/`：研究笔记占位。
- `demo/`、`web_demo/`：简单示例与 Web Demo 入口。
- `dev_utils/`：临时调试脚本与检查工具。
- `outputs/`、`multirun/`、`src/engine/outputs/`：训练/推理日志与权重（已 .gitignore），无需改动。

启动脚本与复现
-------------
- 训练：`python -m src.engine.train` 或 `python scripts/run_train.py`（支持 Hydra 覆写，如 `data.use_dummy=false` 等）。
- 推理：`python -m src.engine.infer --image demo/demo_patch.png --mask_out demo/pred_mask.png` 或 `scripts/run_infer.py`。
- 消融：`scripts/run_ablation.py` 按配置批量运行（需在脚本内调整实验列表）。
- 评测/单测：`python -m pytest tests`；静态检查 `ruff check src tests`。

日志与输出来源
-------------
- 训练日志：默认写入 `outputs/train_*.log`，最佳权重 `outputs/best.pt`，摘要 `outputs/best_checkpoint.txt`。
- Hydra 快照：`.hydra/`、`multirun/` 记录运行配置；`src/engine/train.py` 中也会在 `log.save_dir` 生成日志。
- 额外输出：`src/engine/outputs/` 可能包含局部试验日志；保持不变即可。

冗余/待整理文件（当前保留，不删除）
-------------------------------
- 打包文件：`src/engine/engine.zip`、`src/losses/combined.zip`、`src/models/model.zip`、`src/datasets/datasets.zip`；若无用可后续清理。
- 临时/调试：`dev_utils/tmp_helpers/*`、`note.md`、`problem.md`、`zancun`。
- IDE/缓存：`.idea/`、`.mypy_cache/`、`.ruff_cache/`、`src/engine/.hydra/`、`outputs/`、`multirun/`、`.hydra/`。

改进方向（结合现状）
-----------------
- 训练稳健性：分层采样每 epoch 变 seed、验证指标按样本加权、梯度裁剪/cls logits clamp、加入 weight decay + 学习率调度、`pos_weight`/focal 应对不平衡。
- 模型增强：接入可选轻量骨干（MobileNet/EfficientNet）、SE/CBAM 注意力、动态损失加权（不确定性/GradNorm）、深监督或 ASPP。
- 数据与泛化：补充染色/颜色扰动，跨中心验证或 K 折；滑窗/分块推理脚本与可视化完善。
- 复现与记录：固定随机种子、保留 Hydra 配置、日志中保存高损失样本列表，所有脚本参数写入 README。

改动留痕
-------
- 所有新增 README/指引均在对应目录下，便于后续查阅；如有新增模块，请同步更新相应 README 与本文件的目录列表。
