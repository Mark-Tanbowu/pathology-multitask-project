"""Dice loss implementations."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss implemented as a PyTorch module."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W)
        targets: (B, 1, H, W)
        """
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(1, 2, 3)) + self.eps
        dice = 1 - num / den
        return dice.mean()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(1, 2, 3)) + eps
    dice = 1 - num / den
    return dice.mean()


def dice_coefficient(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return (2 * inter + eps) / union
