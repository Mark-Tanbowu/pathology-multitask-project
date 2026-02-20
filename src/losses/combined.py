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
        enable_seg: bool = True,
        enable_cls: bool = True,
        weighting: str | None = None,
        use_uncertainty: bool = False,
        uncertainty_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.cls_type = cls_type
        if not enable_seg and not enable_cls:
            raise ValueError("MultiTaskLoss requires at least one enabled task.")
        self.enable_seg = enable_seg
        self.enable_cls = enable_cls
        weighting = (weighting or "").strip().lower()
        if not weighting:
            weighting = "uncertainty" if use_uncertainty else "fixed"
        if weighting not in {"fixed", "uncertainty", "gradnorm"}:
            raise ValueError(f"Unsupported weighting strategy: {weighting}")
        self.weighting = weighting
        self.use_uncertainty = weighting == "uncertainty"
        if self.use_uncertainty:
            if enable_seg:
                self.log_sigma_seg = nn.Parameter(torch.tensor(float(uncertainty_init)))
            else:
                self.register_parameter("log_sigma_seg", None)
            if enable_cls:
                self.log_sigma_cls = nn.Parameter(torch.tensor(float(uncertainty_init)))
            else:
                self.register_parameter("log_sigma_cls", None)

    def compute_task_losses(
        self,
        seg_logits: torch.Tensor | None,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor | None,
        cls_targets: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[str], dict]:
        """计算各任务原始损失，不做动态加权。

        返回：
            - task_losses：按顺序排列的 loss 张量（seg、cls）
            - task_names：与 task_losses 对齐的任务名
            - parts：用于日志的标量与中间项
        """
        if seg_logits is None and cls_logits is None:
            raise ValueError("At least one of seg_logits or cls_logits must be provided.")

        parts = {
            "seg_loss": 0.0,
            "seg_bce": 0.0,
            "seg_dice": 0.0,
            "cls_loss": 0.0,
        }
        task_losses: list[torch.Tensor] = []
        task_names: list[str] = []
        if self.enable_seg:
            if seg_logits is None:
                raise ValueError("Segmentation branch enabled but seg_logits is None.")
            # 分割损失：BCE 捕捉像素级准确度，Dice 缓解前景-背景不平衡痛点
            seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets)
            seg_dice = soft_dice_loss(seg_logits, seg_targets)
            seg_loss = self.dice_weight * seg_dice + (1 - self.dice_weight) * seg_bce
            parts["seg_loss"] = seg_loss.item()
            parts["seg_bce"] = seg_bce.item()
            parts["seg_dice"] = seg_dice.item()
            task_losses.append(seg_loss)
            task_names.append("seg")

        if self.enable_cls:
            if cls_logits is None:
                raise ValueError("Classification branch enabled but cls_logits is None.")
            # 分类损失：根据类别数自动选择 BCE/CE，确保接口自适应
            if self.cls_type == "ce" and cls_logits.shape[1] > 1:
                cls_loss = F.cross_entropy(cls_logits, cls_targets.long())
            else:
                cls_logits_flat = cls_logits.view(cls_logits.shape[0], -1)
                if cls_logits_flat.shape[1] == 1:
                    cls_logits_flat = cls_logits_flat.squeeze(1)
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits_flat, cls_targets.float())
            parts["cls_loss"] = cls_loss.item()
            task_losses.append(cls_loss)
            task_names.append("cls")

        return task_losses, task_names, parts

    def forward(
        self,
        seg_logits: torch.Tensor | None,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor | None,
        cls_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        task_losses, task_names, parts = self.compute_task_losses(
            seg_logits=seg_logits,
            seg_targets=seg_targets,
            cls_logits=cls_logits,
            cls_targets=cls_targets,
        )
        device = task_losses[0].device
        total = torch.zeros(1, device=device)

        for task_name, task_loss in zip(task_names, task_losses):
            if task_name == "seg":
                base_weight = self.seg_weight
                if self.use_uncertainty:
                    # 使用 log(sigma^2) 作为可训练参数，确保数值稳定且 sigma 始终为正
                    log_sigma = self.log_sigma_seg
                    weight = torch.exp(-log_sigma)
                    total = total + base_weight * (weight * task_loss + log_sigma)
                    parts["seg_weight_dyn"] = weight.item()
                    parts["log_sigma_seg"] = log_sigma.item()
                else:
                    total = total + base_weight * task_loss
            elif task_name == "cls":
                base_weight = self.cls_weight
                if self.use_uncertainty:
                    log_sigma = self.log_sigma_cls
                    weight = torch.exp(-log_sigma)
                    total = total + base_weight * (weight * task_loss + log_sigma)
                    parts["cls_weight_dyn"] = weight.item()
                    parts["log_sigma_cls"] = log_sigma.item()
                else:
                    total = total + base_weight * task_loss

        return total, parts
