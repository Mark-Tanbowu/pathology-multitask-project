"""多任务联合损失：分割 + 分类。

设计目的：
    - 通过固定权重的线性组合，将分割与分类信号同时回传，避免单任务主导训练；
    - seg_loss 由 BCE 与软 Dice 组合，兼顾像素级对齐与区域重叠；
    - cls_loss 支持二分类 BCE 与多分类 CE，保持与模型输出一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import soft_dice_loss


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        dice_weight: float = 0.5,
        cls_type: str = "bce",
    ) -> None:
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.cls_type = cls_type

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # 分割损失：BCE 捕捉像素级准确度，Dice 缓解前景-背景不平衡痛点
        seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets)
        seg_dice = soft_dice_loss(seg_logits, seg_targets)
        seg_loss = self.dice_weight * seg_dice + (1 - self.dice_weight) * seg_bce

        # 分类损失：根据类别数自动选择 BCE/CE，确保接口自适应
        if self.cls_type == "ce" and cls_logits.shape[1] > 1:
            cls_loss = F.cross_entropy(cls_logits, cls_targets.long())
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits.squeeze(1), cls_targets.float())

        # 联合损失：配置文件指定权重，后续可与 dynamic_loss 模块进行对比实验
        total = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        parts = {
            "seg_loss": seg_loss.item(),
            "seg_bce": seg_bce.item(),
            "seg_dice": seg_dice.item(),
            "cls_loss": cls_loss.item(),
        }
        return total, parts
