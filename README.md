# Pathology Multitask Project (Segmentation + Classification)

本仓库是一个面向 **数字病理** 的多任务学习模板：**共享编码器 + 分割头 + 分类头**。
针对 CAMELYON16/17 等病理数据集，支持从 Patch 级训练起步，逐步扩展到 WSI 工作流。

## 快速开始（CPU / 无本地 GPU）
```bash
# 1) 创建并激活环境（可选：conda 或 venv）
# conda create -n patho python=3.10 -y && conda activate patho

# 2) 安装依赖
pip install -r requirements.txt

# 3) 准备最小数据结构（示例）
# data/train/images/*.png, data/train/masks/*_mask.png, data/train/labels.csv
# data/val/images/*.png,   data/val/masks/*_mask.png,   data/val/labels.csv

# 4) 训练（Hydra 配置）
python -m src.engine.train  # 默认使用 Dummy 数据快速冒烟

# 5) 推理
python -m src.engine.infer --image demo/demo_patch.png --mask_out demo/pred_mask.png --overlay_out demo/overlay.png
```

> **提示**：如果你在 Windows 且没有 GPU，可在本地做小样本验证；正式训练建议使用 Colab/Kaggle/远端服务器。

### 关于 Dummy 数据与真实数据
- 默认配置 (`configs/defaults.yaml`) 会启用 `data.use_dummy=true`，生成随机图像/掩膜/标签完成端到端跑通。
- 当你准备好 CAMELYON16/17 等数据时，将 `use_dummy` 设为 `false` 并填好对应路径即可。

### 可选增强模块
- `optional_modules/lightweight_backbones/`: MobileNet/EfficientNet 编码器示例，便于构建轻量模型。
- `optional_modules/attention_modules/`: SE/CBAM 注意力模块，示例说明如何在新文件中组合使用。
- `optional_modules/dynamic_loss/`: GradNorm/DWA 动态权重算法，可在自定义训练脚本中调用。
> 这些模块**不会改动 baseline 源码**，仅通过继承/组合方式演示接入。

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
