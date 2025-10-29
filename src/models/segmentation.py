"""
把特征图变回为图像，并且预测每个像素属于目标的概率
"""


import torch
import torch.nn as nn


class SimpleSegHead(nn.Module):
    """简单分割头：一层卷积 + 上采样（可选）。"""
#卷积特征＋预测mask+根据需要上采样回输入尺寸
    def __init__(self, in_channels: int, upsample_to_input: bool = True):
    #in_channels：来自backbone部分的通道数 upsample_to_input：是否上采样会输入原图像大小
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),#3×3卷积融合语义信息
            nn.ReLU(inplace=True),#激活非线性
            nn.Conv2d(256, 1, kernel_size=1),#1×1降卷积通道到1 输出mask logits
        )#输出1通道 代表二分类情况，如果多分类的话改为num_classes
        self.upsample = upsample_to_input
#记录是否需要上采样
#前向传播函数
    def forward(self, feats, input_size=None):
        x = self.conv(feats)
        if self.upsample and input_size is not None:
            x = torch.nn.functional.interpolate(
                x, size=input_size, mode="bilinear", align_corners=False
            )
        return x

