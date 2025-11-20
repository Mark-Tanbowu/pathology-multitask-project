"""多任务病理模型的 ResNet 编码器封装。 

- 目标：在不修改 torchvision 官方实现的前提下，额外返回 U-Net 解码器所需的多尺度特征；
- 痛点：原生 ResNet 只输出最高层特征，分割解码器缺少 skip connection 输入；
- 意义：通过轻量包装，既能复用 ImageNet 预训练权重，也能在离线环境下默认不下载权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

ResNetName = Literal["resnet18", "resnet34"]


@dataclass
class EncoderOutput:
    """编码器前向输出的统一容器。

    features: 最深层特征图，供分类头与解码器入口使用；
    skips:     自顶向下的中间特征列表，供 U-Net 解码器做特征拼接。"""

    features: torch.Tensor
    skips: List[torch.Tensor]

    @property
    def channels(self) -> int:
        return self.features.shape[1]


class ResNetEncoder(nn.Module):
    """返回 skip connection 的 ResNet 编码器。

    参数：
        name: 选择 ResNet18/34，兼顾轻量与表现；
        pretrained: 是否加载 ImageNet 预训练，默认为 False 以便完全离线运行。
    """

    def __init__(self, name: ResNetName = "resnet18", pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet34_Weights.DEFAULT if name == "resnet34" else ResNet18_Weights.DEFAULT

        if name == "resnet34":
            backbone = resnet34(weights=weights)
            self.feature_dims = [64, 64, 128, 256, 512]
        else:
            backbone = resnet18(weights=weights)
            self.feature_dims = [64, 64, 128, 256, 512]

        # Stem：保持原始输入的低层纹理，供最后一级 skip 拼接，减少信息损失
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool

        # Encoder blocks：逐层提取语义，channel 维度递增、空间分辨率递减
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        # 逐层前向：保留每一级输出，兼顾分辨率与语义丰富度
        x0 = self.stem(x)  # [B, 64, H/2, W/2] 低层纹理，利于精细分割
        x1 = self.pool(x0)  # [B, 64, H/4, W/4]
        x2 = self.layer1(x1)  # [B, 64, H/4, W/4]
        x3 = self.layer2(x2)  # [B, 128, H/8, W/8]
        x4 = self.layer3(x3)  # [B, 256, H/16, W/16]
        x5 = self.layer4(x4)  # [B, 512, H/32, W/32] 高层语义，供分类与解码器入口

        # U-Net 需要从深到浅的 skip 列表，这里按解码顺序组织
        skips: List[torch.Tensor] = [x4, x3, x2, x0]
        return EncoderOutput(features=x5, skips=skips)

    @property
    def out_channels(self) -> int:
        return self.feature_dims[-1]


def get_backbone(name: ResNetName = "resnet18", pretrained: bool = False) -> ResNetEncoder:
    """Factory function to align with legacy API."""

    return ResNetEncoder(name=name, pretrained=pretrained)
