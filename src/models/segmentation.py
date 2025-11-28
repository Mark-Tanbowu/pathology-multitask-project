"""U-Net 风格的分割解码器模块。

- 适配 ResNet 编码器输出的多尺度特征；
- 采用逐级上采样 + skip connection 还原空间信息；
- 头部输出单通道 logits，可直接接 `BCEWithLogitsLoss` 或 Dice；
- 目的：在保持结构最小化的同时确保端到端可运行，便于后续替换为更复杂的解码器。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 经典的 Conv-BN-ReLU ×2 结构，兼顾稳定性与表达能力
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # 解码阶段：上采样后的特征与对应 skip 拼接，再做卷积融合
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """最小化实现的 U-Net 解码器，直接消费 ResNet 编码器输出。

    设计要点：
        - 通过四级上采样逐步恢复分辨率，保持计算开销可控；
        - channel 设计与 ResNet18/34 默认输出对齐，无需额外适配层；
        - 末端使用 1×1 卷积生成单通道 logits，便于后续 Sigmoid/阈值化。
    """

    def __init__(self, encoder_channels: List[int]):
        super().__init__()
        enc_ch = encoder_channels
        self.up1 = UpBlock(enc_ch[-1], enc_ch[-2], 256)
        self.up2 = UpBlock(256, enc_ch[-3], 128)
        self.up3 = UpBlock(128, enc_ch[-4], 64)
        self.up4 = ConvBlock(64 + enc_ch[-5], 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        # 解码顺序与 encoder skip 顺序相反：最深特征先与最深 skip 结合
        x = self.up1(features, skips[0])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[2])
        # 最浅层 skip 通道数较少，直接使用 ConvBlock 融合
        x = self.up4(torch.cat([F.interpolate(x, size=skips[3].shape[2:], mode="bilinear", align_corners=False), skips[3]], dim=1))
        return self.head(x)
