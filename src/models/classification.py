import torch.nn as nn


class SimpleClsHead(nn.Module):
    """简单分类头：全局池化 + 全连接。"""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, feats):
        return self.head(feats)
