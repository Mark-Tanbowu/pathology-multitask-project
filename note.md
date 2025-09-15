一、深度学习核心
torch：PyTorch，本项目的核心深度学习框架，用来定义网络、训练、GPU/CPU 运算。
torchvision：PyTorch 的视觉工具包，提供图像数据集、常见模型（ResNet 等）、图像变换。

二、模型扩展与特化
timm：PyTorch Image Models，大量预训练模型库（EfficientNet、MobileNet、ViT 等），你计划用的轻量化骨干就来自这里。
segmentation-models-pytorch：常用分割模型封装（U-Net、DeepLabV3+ 等），可以快速调用而不用自己从零写。

三、科学计算与数据处理
numpy：数值计算库，张量/矩阵运算基础。
pandas：表格数据处理，常用来读写 CSV（比如 labels.csv）。
scikit-learn：机器学习工具，尤其是评估指标（AUC、F1-score 等），在医学图像分类里很常用。
scikit-image：图像处理工具包，提供滤波、边缘检测、形态学操作等。
Pillow：图像读取/保存库，PyTorch 常用的图像 I/O 后端。
opencv-python：计算机视觉库，用于图像预处理、几何变换、绘制结果。

四、训练配置与增强
albumentations：强大的图像增强库，特别适合医学图像（旋转、翻转、颜色扰动、对比度、模糊等）。
hydra-core：配置管理框架，可以用 yaml 灵活切换模型/超参数。
omegaconf：Hydra 底层依赖，用来操作配置字典。

五、训练框架与日志
pytorch-lightning：高层训练框架，帮你减少 train/validate 循环的样板代码；有助于写更整洁的实验代码。
tensorboard：训练可视化工具，记录 loss/accuracy 曲线、模型图等。
mlflow：实验管理工具，保存参数、指标、模型，便于复现实验。
rich：终端美化工具，可以打印彩色日志、进度条。

六、开发/测试辅助（dev 组）
pytest：单元测试框架，保证代码正确性。
ruff：快速 Python 代码检查工具，替代 flake8。
pre-commit：Git 提交钩子管理，确保提交前自动做代码格式检查、lint。

📌 总结类比：

torch/torchvision/timm/smp = 模型和深度学习引擎。
numpy/pandas/sklearn/skimage/Pillow/opencv = 科学计算和图像处理工具。
albumentations/hydra/omegaconf = 数据增强 + 配置管理。
lightning/tensorboard/mlflow/rich = 训练框架与实验日志。
pytest/ruff/pre-commit = 开发质量保证。