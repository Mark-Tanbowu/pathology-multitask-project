"""Convolutional Block Attention Module (CBAM) as an optional add-on."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3))
        max_val, _ = torch.max(x, dim=(2, 3))
        attn = self.mlp(avg) + self.mlp(max_val)
        return torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_val], dim=1)
        return torch.sigmoid(self.conv(x))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


def integrate_cbam_with_decoder(conv_block_cls):
    """Illustrative helper showing how to wrap a decoder block with CBAM.

    Usage in a separate model file (without editing baseline):
        from optional_modules.attention_modules.cbam_block import CBAMBlock, integrate_cbam_with_decoder
        from src.models.segmentation import ConvBlock

        AttentionConv = integrate_cbam_with_decoder(ConvBlock)
    """

    class AttentionConv(conv_block_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cbam = CBAMBlock(self.block[-1].num_features if hasattr(self, "block") else args[1])

        def forward(self, x):
            x = super().forward(x)
            return self.cbam(x)

    return AttentionConv
