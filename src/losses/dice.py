"""Dice loss implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
