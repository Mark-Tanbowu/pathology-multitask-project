import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = (probs + targets).sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


class MultiTaskLoss(nn.Module):#多任务损失类自定义
    def __init__(
        self, seg_weight: float = 1.0, cls_weight: float = 1.0, dice_weight: float = 0.5
    ):#构建三个权重可调节参数
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight

    def forward(self, outputs, batch):#前向计算逻辑
        loss = 0.0
        #计算分割部分损失
        if outputs.get("seg") is not None and batch.get("mask") is not None:
            #计算二值交叉熵损失BCE
            bce = F.binary_cross_entropy_with_logits(outputs["seg"], batch["mask"])
            #调节dice loss计算Dice Loss
            dice = dice_loss(outputs["seg"], batch["mask"])
            #组合计算loss
            loss = loss + self.seg_weight * (
                self.dice_weight * dice + (1 - self.dice_weight) * bce
            )
        #计算分类部分损失
        if outputs.get("cls") is not None and batch.get("label") is not None:
            ce = F.cross_entropy(outputs["cls"], batch["label"])
            loss = loss + self.cls_weight * ce
        return loss#返回总loss
