"""Squeeze-and-Excitation (SE) block for optional attention augmentation.

Example usage without touching baseline files:
    from optional_modules.attention_modules.se_block import SEBlock
    from src.models.segmentation import ConvBlock

    class SegBlockWithSE(ConvBlock):
        def __init__(self, in_ch, out_ch):
            super().__init__(in_ch, out_ch)
            self.se = SEBlock(out_ch)
        def forward(self, x):
            x = super().forward(x)
            return self.se(x) * x
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y
