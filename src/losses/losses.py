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


class MultiTaskLoss(nn.Module):
    def __init__(
        self, seg_weight: float = 1.0, cls_weight: float = 1.0, dice_weight: float = 0.5
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight

    def forward(self, outputs, batch):
        loss = 0.0
        if outputs.get("seg") is not None and batch.get("mask") is not None:
            bce = F.binary_cross_entropy_with_logits(outputs["seg"], batch["mask"])
            dice = dice_loss(outputs["seg"], batch["mask"])
            loss = loss + self.seg_weight * (
                self.dice_weight * dice + (1 - self.dice_weight) * bce
            )
        if outputs.get("cls") is not None and batch.get("label") is not None:
            ce = F.cross_entropy(outputs["cls"], batch["label"])
            loss = loss + self.cls_weight * ce
        return loss
