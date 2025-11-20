"""MobileNet-based encoder as an optional lightweight alternative.

Usage example (without modifying baseline files):

    from optional_modules.lightweight_backbones.mobilenet_encoder import MobileNetEncoder
    from src.models.multitask_model import MultiTaskModel

    class LightweightMultitaskModel(MultiTaskModel):
        def __init__(self, num_classes: int = 1):
            super().__init__(num_classes=num_classes)
            self.encoder = MobileNetEncoder()
            self.decoder = UNetDecoder(self.encoder.feature_dims)
            self.cls_head = ClassificationHead(self.encoder.out_channels, num_classes=num_classes)

You can place the subclass above in a new script (e.g., `models/multitask_model_light.py`) and
import it in a training script without touching the baseline source code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = mobilenet_v2(weights="DEFAULT" if pretrained else None).features
        self.stages = nn.ModuleList([
            backbone[:4],  # 24 ch
            backbone[4:7],  # 32 ch
            backbone[7:14],  # 96 ch
            backbone[14:],  # 320 ch
        ])
        self.feature_dims = [24, 32, 96, 320]

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
