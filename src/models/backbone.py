from typing import Literal

import torch
import torch.nn as nn


def get_backbone(name: Literal["resnet18", "resnet34"] = "resnet18") -> nn.Sequential:
    """返回去掉池化/FC的特征提取骨干。"""
    if name == "resnet34":
        backbone = torch.hub.load("pytorch/vision:v0.14.0", "resnet34", pretrained=True)
        ch = 512
    else:
        backbone = torch.hub.load("pytorch/vision:v0.14.0", "resnet18", pretrained=True)
        ch = 512
    encoder = nn.Sequential(*list(backbone.children())[:-2])
    encoder.out_channels = ch  # 标注输出通道
    return encoder
