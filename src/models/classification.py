"""多任务病理模型的分类头。

- 复用编码器最高层特征，避免额外计算；
- 通过全局平均池化将特征压缩为 1×1，强调整体形态信息；
- 末端线性层兼容二分类/多分类，保持接口统一。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, dropout: float = 0.0):
        super().__init__()
        # 结构：全局池化 → 展平 → （可选 Dropout）→ 全连接
        layers = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)
