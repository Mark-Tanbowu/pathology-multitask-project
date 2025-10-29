"""
metrics.py - 评价指标模块
提供 Dice、IoU、Accuracy 等常用医学图像与分类指标。
"""
# ============================================================
# metrics.py （2025-10-29 更新）
# 修复 Dice 恒定问题：移除 .item() 保持 Tensor 类型
# ============================================================

import torch

def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算 Dice 系数（支持批次）
    preds: 模型预测概率（0~1）
    targets: ground truth 掩码（0/1）
    """
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice.mean()  # ⚙️ 不再 .item()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    计算 IoU（Jaccard 指数）
    """
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3)) + eps
    iou = inter / union
    return iou.mean().item()


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    从 logits 计算分类准确率
    支持二分类与多分类
    """
    if logits.ndim == 2 and logits.shape[1] == 1:
        preds = (torch.sigmoid(logits) > 0.5).long()
    else:
        preds = torch.argmax(logits, dim=1)

    correct = (preds.view_as(labels) == labels).float().mean()
    return correct.item()
