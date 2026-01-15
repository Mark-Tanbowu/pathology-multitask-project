轻量骨干模块
=============

文件
----
- mobilenet_encoder.py：MobileNetV2 编码器封装。
- efficientnet_encoder.py：EfficientNet-Lite 编码器封装。

改进建议
--------
- 在模型工厂中配置化接入，支持预训练权重与通道宽度调节。
- 对比参数量/吞吐/显存与 ResNet18/34，记录消融结果。
