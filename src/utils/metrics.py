"""Evaluation metrics for segmentation and classification."""

from __future__ import annotations

import torch


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return (2 * inter + eps) / union


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3)) + eps
    return inter / union


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim == 2 and logits.shape[1] == 1:
        preds = (torch.sigmoid(logits) > 0.5).long()
    else:
        preds = torch.argmax(logits, dim=1)
    correct = (preds.view_as(labels) == labels).float().mean()
    return correct.item()
