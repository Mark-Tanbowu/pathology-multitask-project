# Pathology Multitask Project (Segmentation + Classification)

本仓库是一个面向 **数字病理** 的多任务学习模板：**共享编码器 + 分割头 + 分类头**。
针对 CAMELYON16/17 等病理数据集，支持从 Patch 级训练起步，逐步扩展到 WSI 工作流。

## 快速开始（CPU / 无本地 GPU）
下面流程按 **可最小化依赖的冒烟 → 准备真实数据 → 正式训练/推理** 来写，
默认以 Windows/CPU 兼容为主。

### 0) 环境与依赖
```bash
# 任选一种方式创建环境
# conda create -n patho python=3.10 -y && conda activate patho
# python -m venv .venv && .\.venv\Scripts\activate  (Windows)

pip install -r requirements.txt
```

### 1) 冒烟跑通（不需要数据）
默认配置里 `configs/defaults.yaml` 的 `data.use_dummy=false`，因此**直接跑会尝试读取真实数据**。
如果你只是确认训练链路是否可跑，请显式打开 dummy：
```bash
python -m src.engine.train data.use_dummy=true
```
这会在 `run/YYYYMMDD_HHMM/` 下生成日志、曲线和 `best.pt`（若 `log.save_ckpt=true`）。

### 2) 准备真实数据（Patch 级）
模型训练读取的**默认路径与命名规则**由 `configs/defaults.yaml` 决定，且 `PathologyDataset` 约定：
```
data/
  train/
    images_fixed/    # patch 图像（.bmp）
    masks_fixed/     # 对应掩膜（_anno.bmp）
    labels.csv       # CSV: name,label（name 不带扩展名）
  val/
    images_fixed/
    masks_fixed/
    labels.csv
```
**命名示例**：
```
images_fixed/001.bmp
masks_fixed/001_anno.bmp
labels.csv: 001,1
```

如需修改目录，使用 Hydra 覆写参数即可：
```bash
python -m src.engine.train \
  data.use_dummy=false \
  data.train_images=data/train/images_fixed \
  data.train_masks=data/train/masks_fixed \
  data.train_labels=data/train/labels.csv \
  data.val_images=data/val/images_fixed \
  data.val_masks=data/val/masks_fixed \
  data.val_labels=data/val/labels.csv
```

### 3) WSI 动态切（Slide Manifest，推荐）
该流程会生成 **train/val/test 各自的 slide manifest + coords**，训练/检测时直接动态读取 patch。

**按默认配置生成 train/val/test 的 slide manifest + coords：**
```bash
python -m prepare.manifest_builder_train
python -m prepare.manifest_builder_val
python -m prepare.manifest_builder_test
```

**如需手动覆盖参数（示例：train）：**
```bash
python -m prepare.manifest_builder_train ^
  --slides-dir <your_train_slides_dir> ^
  --masks-dir <your_train_masks_dir> ^
  --annotations-dir <your_annotations_dir> ^
  --coords-out-dir <your_coords_out_dir> ^
  --slide-manifest-path <your_slide_manifest_csv> ^
  --output-csv <your_patch_manifest_csv>
```

**启动训练（WSI 动态切）：**
```bash
python -m src.engine.train
```

**运行 test 检测（加载最佳权重）：**
```bash
python -m src.engine.test_slide_detect
```
若需要指定某个权重，再追加：
```bash
python -m src.engine.test_slide_detect test_detect.ckpt=run/20260214_0148/best.pt
```

说明：
- `manifest_slides_*.csv` + `coords/` 是训练用的核心输入。
- `patch_manifest_*.csv` 只是辅助输出，可忽略。

### 3b) （可选）从 WSI 生成 Patch Manifest（旧版）
旧入口 `python -m prepare.manifest_builder` 已移除，不再推荐使用。

若仅想导出 patch 级 CSV 便于排查，可在上述 train/val/test builder 中开启：
```bash
python -m prepare.manifest_builder_train --write-patch-manifest
```
说明：
- `x/y` 坐标是 level-0，便于 OpenSlide 直接读取。
- `--groups` 默认 `Tumor`，如标注名不同请调整。

### 4) 正式训练（真实数据 / Patch 级）
```bash
python -m src.engine.train
```
常用覆写示例（可按需组合）：
```bash
python -m src.engine.train \
  num_epochs=60 \
  batch_size=8 \
  model.backbone=resnet34 \
  loss.weighting=gradnorm
```

### 5) 推理与可视化
```bash
python -m src.engine.infer ^
  --image demo/demo_patch.png ^
  --ckpt run/20260116_0138/best.pt ^
  --mask_out demo/pred_mask.png ^
  --overlay_out demo/overlay.png
```
如果只想验证 CLI 是否可用：
```bash
python -m src.engine.infer --dry_run
```

> **提示**：Windows 无 GPU 可做小样本验证；正式训练建议使用 Colab/Kaggle/远端服务器。

### 关于 Dummy 数据与真实数据
- `data.use_dummy=true`：生成随机图像/掩膜/标签用于冒烟测试。
- `data.use_dummy=false`：读取真实数据目录（见上文结构与命名规则）。

### 可选增强模块
- `optional_modules/lightweight_backbones/`: MobileNet/EfficientNet 编码器示例，便于构建轻量模型。
- `optional_modules/attention_modules/`: SE/CBAM 注意力模块，示例说明如何在新文件中组合使用。
- `optional_modules/dynamic_loss/`: GradNorm/DWA 动态权重算法，可在自定义训练脚本中调用。
> 这些模块**不会改动 baseline 源码**，仅通过继承/组合方式演示接入。

### 输出与日志位置
- 训练输出目录由 Hydra 决定：默认 `run/YYYYMMDD_HHMM/`
- 每次运行会保存 `train_*.log`、`metrics_*.log`、`timing_*.log`，以及可选的 `best.pt`
- Hydra 会在 `.hydra/` 中记录 `config.yaml` / `overrides.yaml` 便于复现

## 目录结构
```text
📁 pathology-multitask-project/
├── 一、项目配置与自动化管理/
│   ├── README.md                        # 项目总体说明文档：介绍研究背景、使用方式、模型架构与结果展示
│   ├── requirements.txt                 # Python依赖包清单，列出运行项目所需库（供 pip install 使用）
│   ├── pyproject.toml                   # Python构建配置文件（兼容Poetry/pip），定义依赖与元信息
│   ├── .gitignore                       # Git忽略规则，排除模型权重、日志、缓存、临时数据等文件
│   ├── .pre-commit-config.yaml          # 预提交钩子配置，提交代码前自动执行格式化、Lint检查与测试
│   │
│   ├── .github/
│   │   └── workflows/
│   │       └── ci.yml                   # GitHub Actions持续集成配置文件，实现自动化测试、构建与代码质量审查
│   │
│   ├── .hydra/                          # Hydra运行时生成目录（自动创建）
│   │   ├── config.yaml                  # 当前运行完整参数快照（保存模型、训练、数据配置）
│   │   ├── hydra.yaml                   # Hydra自身配置文件，控制输出目录、日志记录等
│   │   └── overrides.yaml               # 命令行参数覆盖记录，便于实验复现
│   │
│   └── configs/                         # Hydra配置模板目录
│       ├── defaults.yaml                # 默认实验参数配置（模型、路径、优化器、学习率等）
│       └── model_unet.yaml              # U-Net模型结构配置，用于定义分割网络结构参数
│
├── 二、数据与预处理模块/
│   ├── data/                            # 数据目录
│   │   ├── raw/                         # 原始数据（未处理WSI切片及掩码）
│   │   │   ├── .gitkeep                 # 占位文件，保证空目录被Git追踪
│   │   │   ├── train/
│   │   │   │   ├── images/              # 原始训练图像
│   │   │   │   ├── masks/               # 对应像素级掩码
│   │   │   │   ├── images_fixed/        # 尺寸、颜色或格式修正后的训练图像
│   │   │   │   ├── masks_fixed/         # 处理后的掩码文件
│   │   │   │   └── labels.csv           # 训练集标签文件（图像级分类标签）
│   │   │   └── val/
│   │   │       ├── images/              # 验证集图像
│   │   │       ├── masks/               # 验证集掩码
│   │   │       ├── images_fixed/
│   │   │       ├── masks_fixed/
│   │   │       └── labels.csv
│   │   └── processed/                   # 预处理后数据（如裁剪、归一化）
│   │       └── .gitkeep
│   │
│   └── src/preprocessing/
│       └── wsi_tiling.py                # Whole Slide Image 切片脚本，将大图切为可训练的tile块
│
├── 三、核心模型与算法模块/
│   ├── src/datasets/                    # 数据加载与增强模块
│   │   ├── __init__.py
│   │   ├── camelyon_dataset.py          # CAMELYON数据集类，定义图像、掩码与标签的读取逻辑
│   │   └── transforms.py                # 数据增强函数（旋转、翻转、归一化等）
│   │
│   ├── src/models/                      # 模型结构定义
│   │   ├── __init__.py
│   │   ├── backbone.py                  # 主干网络（ResNet、MobileNet、EfficientNet等）
│   │   ├── segmentation.py              # 分割分支（U-Net结构）
│   │   ├── classification.py            # 分类分支（整图分类输出）
│   │   └── multitask_model.py           # 多任务联合模型，整合共享编码器与双任务输出
│   │
│   ├── src/losses/                      # 损失函数模块
│   │   ├── __init__.py
│   │   ├── dice.py                      # Dice损失（用于分割任务，衡量区域重叠度）
│   │   ├── combined.py                  # 联合损失函数（综合分割与分类任务的加权）
│   │   └── losses.py                    # 其他损失封装（交叉熵、BCE、Focal Loss等）
│   │
│   ├── src/engine/                      # 训练与验证引擎
│   │   ├── __init__.py
│   │   ├── train.py                     # 模型训练主循环（前向传播、反向更新、日志记录）
│   │   ├── validate.py                  # 验证过程（计算IoU、F1等指标）
│   │   ├── infer.py                     # 模型推理接口，支持单图或批量预测
│   │   └── train.log                    # 训练引擎日志文件（记录训练进度与性能）
│   │
│   ├── src/lightning/                   # 预留的PyTorch Lightning封装（可用于未来模块化训练）
│   │   └── __init__.py
│   │
│   └── src/utils/                       # 通用工具模块
│       ├── __init__.py
│       ├── metrics.py                   # 模型评估指标（IoU、Dice、Accuracy、Precision、Recall等）
│       ├── visualizer.py                # 可视化工具（预测结果叠加显示、训练曲线绘制）
│       ├── misc.py                      # 杂项工具函数（日志记录、路径管理、配置加载）
│       └── dice.py                      # 独立Dice指标计算工具，用于快速验证模型输出
│
├── 四、训练脚本与实验控制/
│   ├── scripts/run_train.py             # 模型训练主脚本（整合Hydra配置与训练引擎）
│   ├── scripts/run_infer.py             # 推理脚本（加载模型并进行预测）
│   ├── scripts/train.ps1                # Windows PowerShell版本训练启动脚本
│   └── scripts/train.sh                 # Linux/Mac Shell版本训练启动脚本
│
├── 五、训练结果与输出管理/
│   ├── outputs/checkpoints/             # 模型权重保存目录（如 best.pt、last.pt）
│   ├── outputs/best.pt                  # 当前最优模型权重文件
│   ├── outputs/loss_visualization.png   # 训练损失变化曲线图
│   └── outputs/train.log                # 训练日志文件（记录每轮epoch的loss与指标）
│
├── 六、测试与验证模块/
│   ├── tests/test_dice.py               # 验证Dice损失与指标计算的正确性
│   └── tests/test_smoke.py              # 冒烟测试：确保核心模块可运行、不崩溃
│
├── 七、Web演示与可视化接口/
│   ├── web_demo/app.py                  # Flask/Gradio前端演示接口，支持图像上传与模型预测展示
│   └── web_demo/.gitignore              # 忽略Web上传缓存与临时文件
│
└── 八、文档与辅助资料/
    ├── demo/test.png                    # 模型预测示例图片
    ├── notebooks/README.md              # Jupyter笔记本说明文件，记录实验步骤与结果
    ├── notebooks/.gitkeep               # 占位文件
    ├── note.md                          # 研究笔记（模型设计与实验思路）
    ├── problem.md                       # 过程问题记录与调试总结
    └── train.log                        # 全局训练日志（模型在不同阶段的性能汇总）


```

## 免责声明
本模板用于科研教学参考；在真实临床场景前，必须进行充分验证与伦理审查。
