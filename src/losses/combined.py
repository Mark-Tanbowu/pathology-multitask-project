# ============================================================
# combined.py - 多任务损失函数模块
# 更新日期：2025-10-29
# 作者：花花项目组
#
# 改动说明（2025-10-29）：
# ✳️ 1. 新增 Soft Dice Loss（梯度更平滑）
# ✳️ 2. 分割任务采用 Weighted BCE + Soft Dice 混合策略
# ✳️ 3. 引入 pos_weight 缓解前景稀疏问题
# ✳️ 4. 调整权重：seg_weight=5.0, dice_weight=0.8
# ✳️ 5. 修复多设备兼容问题（pos_weight 自动迁移）
# ✳️ 6. 优化日志输出与类型安全
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ✳️ 改动 1（2025-10-29）：改良 Dice 损失函数
# 目的：使梯度更平滑，在早期训练阶段不至于失效。
# ============================================================
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice Loss：对概率平方项进行平滑，避免梯度为零。
    logits: 模型输出（未 sigmoid）
    targets: 0/1 掩码
    """
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(1, 2, 3)) + eps
    dice = 1 - (num / den)
    return dice.mean()


# ============================================================
# ✳️ 改动 2（2025-10-29）：多任务损失函数（改进版）
# 支持：
# - 分割：Soft Dice + Weighted BCE 混合
# - 分类：BCE（二分类）或 CE（多分类）
# - 自动日志输出
# ============================================================
class MultiTaskLoss(nn.Module):
    """
    多任务损失函数（改进版）
    支持：
    - 分割：Soft Dice + Weighted BCE 混合
    - 分类：BCE（二分类）或 CE（多分类）
    - 自动日志输出
    """

    def __init__(
        self,
        seg_type: str = "bce_with_logits",
        cls_type: str = "bce_with_logits",
        seg_weight: float = 5.0,          # ✳️ 改动 3（2025-10-29）：提高分割权重（原3.0）——提升分割主导性
        cls_weight: float = 1.0,
        dice_weight: float = 0.8,         # ✳️ 改动 4（2025-10-29）：提高Dice比重（原0.5）——避免BCE压制Dice
        multi_class: bool = False,
        pos_weight: float = 5.0           # ✳️ 改动 5（2025-10-29）：新增 Weighted BCE——平衡前景稀疏
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.multi_class = multi_class

        # ⚙️ 改动 6（2025-10-29）：注册 buffer 确保 pos_weight 自动迁移设备
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
    ):
        total_loss = torch.tensor(0.0, device=seg_logits.device)
        log_dict = {}

        # ============================================================
        # 1️⃣ 分割损失：Soft Dice + Weighted BCE（2025-10-29）
        # ============================================================
        if seg_logits is not None and seg_targets is not None:
            # ✳️ 添加 Weighted BCE —— 增强前景梯度
            pos_weight = self.pos_weight.to(seg_logits.device)
            bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets, pos_weight=pos_weight)

            # ✳️ 使用 Soft Dice Loss —— 防止 Dice 梯度崩塌
            dice = soft_dice_loss(seg_logits, seg_targets)

            seg_loss = self.dice_weight * dice + (1 - self.dice_weight) * bce
            total_loss = total_loss + self.seg_weight * seg_loss

            log_dict.update({
                "bce_loss": round(bce.item(), 5),
                "dice_loss": round(dice.item(), 5),
                "seg_loss": round(seg_loss.item(), 5)
            })

        # ============================================================
        # 2️⃣ 分类损失：BCE / CE（保持原逻辑）
        # ============================================================
        if cls_logits is not None and cls_targets is not None:
            if self.multi_class:
                cls_loss = F.cross_entropy(cls_logits, cls_targets)
            else:
                # ⚠️ squeeze(1) 避免维度不匹配，确保 logits 与 targets 对齐
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits.squeeze(1), cls_targets.float()
                )

            total_loss = total_loss + self.cls_weight * cls_loss
            log_dict["cls_loss"] = round(cls_loss.item(), 5)

        # ============================================================
        # 3️⃣ 汇总日志
        # ============================================================
        log_dict["total_loss"] = round(total_loss.item(), 5)
        return total_loss, log_dict
