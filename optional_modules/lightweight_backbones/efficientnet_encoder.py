"""EfficientNet encoder as an optional drop-in replacement for the baseline encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = efficientnet_b0(weights="DEFAULT" if pretrained else None)
        self.stages = nn.ModuleList(
            [
                backbone.features[:2],
                backbone.features[2:4],
                backbone.features[4:6],
                backbone.features[6:],
            ]
        )
        self.feature_dims = [24, 40, 112, 1280]

    def forward(self, x: torch.Tensor):
        skips = []
        for stage in self.stages[:-1]:
            x = stage(x)
            skips.append(x)
        x = self.stages[-1](x)
        skips = skips[::-1]
        return x, skips

    @property
    def out_channels(self) -> int:
        return self.feature_dims[-1]
