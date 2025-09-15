import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """二值 Dice 损失。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.eps
        dice = num / den
        return 1 - dice.mean()
