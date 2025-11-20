"""Utility functions for losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combined import MultiTaskLoss
from .dice import soft_dice_loss

__all__ = ["MultiTaskLoss", "soft_dice_loss", "binary_cross_entropy"]


def binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)
