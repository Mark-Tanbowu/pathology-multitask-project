import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """二值 Dice 损失。"""#自定义一个损失函数

    def __init__(self, eps: float = 1e-6):#构建一个很小的epslion 用于分母
        super().__init__()
        self.eps = eps#加入函数内部成员

    def forward(self, logits, targets):#定义一个前向传播函数 输入的内容弄个分别是预测值和真实值
        probs = torch.sigmoid(logits)#进行sigmoid进行映射激活 给出预测结果
        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.eps
        dice = num / den#计算Dice
        return 1 - dice.mean()#我们希望损失值越小越好 但是Dice是越大越好