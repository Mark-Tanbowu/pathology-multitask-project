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
python -m src.engine.train

# 5) 推理
python -m src.engine.infer --image demo/demo_patch.png --mask_out demo/pred_mask.png --overlay_out demo/overlay.png
```

> **提示**：如果你在 Windows 且没有 GPU，可在本地做小样本验证；正式训练建议使用 Colab/Kaggle/远端服务器。

## 目录结构
```text
📂 pathology-multitask-project/
├── README.md
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── defaults.yaml
│   └── model_unet.yaml              # 示例：替换更强的分割头
├── data/
│   ├── raw/                         # DVC/LFS 管理的大文件（.gitignore）
│   └── processed/
├── notebooks/
│   └── README.md
├── scripts/
│   ├── run_train.py
│   └── run_infer.py
├── src/
│   ├── preprocessing/
│   │   └── wsi_tiling.py
│   ├── datasets/
│   │   ├── camelyon_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── segmentation.py
│   │   ├── classification.py
│   │   └── multitask_model.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dice.py
│   │   └── combined.py
│   ├── engine/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── infer.py
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── misc.py
├── tests/
│   └── test_smoke.py
└── web_demo/
    └── app.py                       # Streamlit 演示（可选）
```

## 关键设计
- **Trunk-based Development**：main 受保护，短分支 + PR 审核。
- **数据与权重**：建议使用 DVC/云端远端；仓库仅保存指针与小样本。
- **配置管理**：Hydra YAML；`configs/defaults.yaml` 可一键切参。
- **可替换组件**：`models/` 下可自由替换骨干与分割/分类头。
- **评估与可视化**：`utils/` 提供 Dice、AUC、叠加可视化等工具。
- **CI**：GitHub Actions 进行 lint/测试/快速推理检查。

## 免责声明
本模板用于科研教学参考；在真实临床场景前，必须进行充分验证与伦理审查。
