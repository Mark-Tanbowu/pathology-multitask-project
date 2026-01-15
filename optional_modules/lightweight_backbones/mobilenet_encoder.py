"""MobileNetV2 编码器（轻量骨干）。

- 输出多尺度特征与 skip，便于直接接入 U-Net 解码器；
- 预设通道/分辨率顺序与 ResNet 封装保持一致：features 最深层，skips 从深到浅。

调整点：
- 增加 width_mult 与 last_channel 缩放，默认 0.4，将最终通道压到 ~512（与 ResNet18 一致）；
- 仍保留标准切分顺序，方便与解码器适配。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import _make_divisible


class MobileNetEncoder(nn.Module):
    """将 torchvision MobileNetV2 切分为 5 级特征，返回主干输出与 4 级 skip。"""

    def __init__(self, pretrained: bool = False, width_mult: float = 0.4):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        # 在部分 torchvision 版本中 mobilenet_v2 不支持 last_channel 形参，这里只传 width_mult
        backbone = mobilenet_v2(weights=weights, width_mult=width_mult).features

        # 划分阶段，确保获得 4 个 skip 与最深层输出
        self.stem = backbone[0]  # conv-bn-relu6, stride=2, c=32
        self.block1 = backbone[1]  # bottleneck, stride=1, c=16（保持尺寸）
        self.block2 = backbone[2:4]  # stride=2 + stride=1, 输出 c=24, H/4
        self.block3 = backbone[4:7]  # stride=2 + stride=1*2, 输出 c=32, H/8
        self.block4 = backbone[7:14]  # stride=2 + 后续，输出 c=96, H/16
        self.block5 = backbone[14:]  # stride=2 + 最终 1x1 conv，输出 c≈1280, H/32

        # 将最终特征压缩到 512 通道，便于与 resnet18 对齐、减轻解码器开销
        self.target_out = 512
        self.proj = nn.Sequential(
            nn.Conv2d(self.block5[-1].out_channels, self.target_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.target_out),
            nn.ReLU(inplace=True),
        )

        # 与 skip 顺序对齐的通道列表（features 在末尾），按 width_mult 动态计算
        stem_ch = _make_divisible(32 * width_mult, 8)
        c24 = _make_divisible(24 * width_mult, 8)
        c32 = _make_divisible(32 * width_mult, 8)
        c96 = _make_divisible(96 * width_mult, 8)
        self.feature_dims = [stem_ch, c24, c32, c96, self.target_out]
        self.width_mult = width_mult

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)        # [B,32,H/2,W/2]
        x1 = self.block1(x0)     # [B,16,H/2,W/2]
        x2 = self.block2(x1)     # [B,24,H/4,W/4]
        x3 = self.block3(x2)     # [B,32,H/8,W/8]
        x4 = self.block4(x3)     # [B,96,H/16,W/16]
        features = self.block5(x4)  # [B,~1280,H/32,W/32]
        features = self.proj(features)  # 压缩到 512 通道

        # 从深到浅的 skip 列表
        skips = [x4, x3, x2, x0]
        return features, skips

    @property
    def out_channels(self) -> int:
        return self.feature_dims[-1]
