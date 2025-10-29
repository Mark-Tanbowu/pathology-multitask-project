"""提供了一个可复用的卷积特征提取网络，用于提取图像的高层语义特征.
该部分包含了 1.U-Net的编码器编码器部分  2.分类分支的特征输入层  3.多任务模型的共享特征抽取部分"""
#导入必要库
from typing import Literal
#用于限制函数参数只能是resnet34 or 18
import torch#提供张量计算
import torch.nn as nn#提供神经网络层与结构定义
from torchvision.models import resnet18 , ResNet18_Weights
from torchvision.models import resnet34 , ResNet34_Weights


#定义了一个用于加载指定的预训练ResNet的函数 利用了Literal进行标记
def get_backbone(name: Literal["resnet18", "resnet34"] = "resnet18") -> nn.Sequential:
    """返回去掉池化/FC的特征提取骨干。"""
    if name == "resnet34":
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        ch = 512
    else:
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        ch = 512
        #选用模型部分 34中等 18轻量
    encoder = nn.Sequential(*list(backbone.children())[:-2])
    #将backbon的所有子模块也就是children取出并且去掉了后两层，取前面的卷积层用nn.Sequential封装成一个新的容器进行调用
    #这样处理过后方便直接输出二维特征图
    encoder.out_channels = ch  # 标注输出通道 动态添加一个属性
    return encoder#返回封装好的新容器
