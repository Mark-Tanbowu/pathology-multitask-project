models 模块说明
==============

文件功能
--------
- `backbone.py`：ResNet18/34 编码器封装，返回多尺度特征与 skip。
- `segmentation.py`：U-Net 解码器实现。
- `classification.py`：全局池化 + 全连接分类头（可选 dropout）。
- `multitask_model.py`：共享编码器的分割+分类双头模型。

改进建议
--------
- 接入轻量骨干（MobileNet/EfficientNet）与注意力（SE/CBAM）为配置化选项。
- 分类头支持 logits clamp/temperature、dropout/label smoothing。
- 分割头可加入深监督、ASPP/PPM、多类别输出。
- 统一权重初始化，暴露参数宽度/通道配置为 Hydra 可调。
