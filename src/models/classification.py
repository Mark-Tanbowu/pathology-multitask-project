"""
分类头的作用 把编码器的卷积特征图转化为类别logits 全局池化-展平-全连接
输入的时来自backbone的feats，形状(N,C,H,W)    输出分类logits 形状(N,num_classes)
"""


import torch.nn as nn


class SimpleClsHead(nn.Module):
    """简单分类头：全局池化 + 全连接。"""

    def __init__(self, in_channels: int, num_classes: int = 1):
        #in_channels 输入通道数 一定要和backbone部分的输出通道一致。 num_classes：分类数默认为1，便于二分类
        super().__init__()#初始化父类，注册子模块以及缓冲区
        self.head = nn.Sequential(#用nn.Sequential串起三步前向算子
            nn.AdaptiveAvgPool2d((1, 1)),#自适应全局平均池化 把所有H,W转化为1,1 例如(N,C,H,W)转化为(N,C,1,1)
            nn.Flatten(),#展平成二维(N,C)为全连接层做准备
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, feats):#前向传播递给上述内容展开 期望 feats 的形状为 (N, in_channels, H, W)；返回 (N, num_classes)。
        return self.head(feats)
