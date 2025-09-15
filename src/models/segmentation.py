import torch.nn as nn
import torch


class SimpleSegHead(nn.Module):
    """简单分割头：一层卷积 + 上采样（可选）。"""

    def __init__(self, in_channels: int, upsample_to_input: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )
        self.upsample = upsample_to_input

    def forward(self, feats, input_size=None):
        x = self.conv(feats)
        if self.upsample and input_size is not None:
            x = torch.nn.functional.interpolate(
                x, size=input_size, mode="bilinear", align_corners=False
            )
        return x
