from typing import Literal

import torch.nn as nn

from .dice import DiceLoss


class MultiTaskLoss(nn.Module):
    """多任务损失包装：seg + cls，简单加权。"""

    def __init__(
        self,
        seg_type: Literal["bce_with_logits", "dice"] = "bce_with_logits",
        cls_type: Literal["bce_with_logits"] = "bce_with_logits",
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
    ):
        super().__init__()
        if seg_type == "dice":
            self.seg_loss = DiceLoss()
        else:
            self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.w_seg = seg_weight
        self.w_cls = cls_weight

    def forward(self, seg_logits, seg_targets, cls_logits, cls_targets):
        seg = self.seg_loss(seg_logits, seg_targets)
        cls = self.cls_loss(cls_logits.squeeze(1), cls_targets)
        return self.w_seg * seg + self.w_cls * cls, {
            "seg_loss": seg.item(),
            "cls_loss": cls.item(),
        }
