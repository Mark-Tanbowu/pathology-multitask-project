"""Hydra 驱动的多任务训练循环（分割 + 分类）。

- 配置统一由 configs/defaults.yaml 读取，便于实验复现；
- 支持 dummy 数据与真实数据集两种路径，方便快速冒烟测试；
- 日志/可视化/权重保存逻辑集中于此，训练脚本更简洁。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets import DummyPathologyDataset, PathologyDataset
from src.datasets.transforms import Compose, HorizontalFlip
from src.losses.combined import MultiTaskLoss
from src.models.multitask_model import MultiTaskModel
from src.utils.metrics import accuracy_from_logits, dice_coefficient
from src.utils.misc import ensure_dir, get_device, set_seed
from src.utils.visualizer import LossVisualizer

LOGGER = logging.getLogger(__name__)


def build_datasets(cfg: DictConfig) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """根据配置构建训练/验证集，支持 dummy 与真实数据两条路径。

    - use_dummy=True：走合成数据，保证“零数据依赖”也能跑通训练管线；
    - use_dummy=False：读取真实路径，接口与 CAMELYON 系列兼容，便于日后替换。"""
    if cfg.data.get("use_dummy", True):
        train_ds = DummyPathologyDataset(
            length=cfg.data.dummy.train_samples,
            image_size=cfg.data.dummy.image_size,
            num_classes=cfg.model.num_classes,
            seed=cfg.seed,
        )
        val_ds = DummyPathologyDataset(
            length=cfg.data.dummy.val_samples,
            image_size=cfg.data.dummy.image_size,
            num_classes=cfg.model.num_classes,
            seed=cfg.seed + 1,
        )
    else:
        # 真实数据路径：可在此扩展更多数据增强/预处理
        transform = Compose([HorizontalFlip(p=0.5)])
        train_ds = PathologyDataset(
            cfg.data.train_images,
            cfg.data.train_masks,
            cfg.data.train_labels,
            transform=transform,
            debug_log=False,
        )
        val_ds = PathologyDataset(
            cfg.data.val_images,
            cfg.data.val_masks,
            cfg.data.val_labels,
            transform=None,
            debug_log=False,
        )
    return train_ds, val_ds


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    ensure_dir(cfg.log.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg.log.save_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    LOGGER.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))

    train_ds, val_ds = build_datasets(cfg)  # 数据集切换由配置控制，便于做 ablation
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # 核心模型：共享 ResNet 编码器 + U-Net 解码器 + 分类头
    model = MultiTaskModel(
        backbone_name=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        seg_upsample_to_input=cfg.model.seg_upsample_to_input,
        encoder_pretrained=cfg.model.get("pretrained", False),
    ).to(device)

    # 固定权重多任务损失：与 optional_modules/dynamic_loss 对比时可作基线
    criterion = MultiTaskLoss(
        seg_weight=cfg.loss.seg_weight,
        cls_weight=cfg.loss.cls_weight,
        dice_weight=cfg.loss.dice_weight,
        cls_type=cfg.loss.cls,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    visualizer = LossVisualizer(save_dir=cfg.log.save_dir)

    best_score = -1.0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0  # 累积训练损失
        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            seg_logits, cls_logits = model(images)
            loss, _ = criterion(seg_logits, masks, cls_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / max(len(train_ds), 1)

        model.eval()
        val_loss, dice_sum, acc_sum, batches = 0.0, 0.0, 0.0, 0  # 验证集指标累计
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                seg_logits, cls_logits = model(images)
                loss, _ = criterion(seg_logits, masks, cls_logits, labels)
                val_loss += loss.item() * images.size(0)
                dice = dice_coefficient(torch.sigmoid(seg_logits), masks).mean().item()
                acc = accuracy_from_logits(cls_logits, labels)
                dice_sum += dice
                acc_sum += acc
                batches += 1
        val_loss /= max(len(val_ds), 1)
        val_dice = dice_sum / max(batches, 1)
        val_acc = acc_sum / max(batches, 1)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | val_acc=%.4f",
            epoch,
            cfg.num_epochs,
            train_loss,
            val_loss,
            val_dice,
            val_acc,
        )
        visualizer.update(epoch, {"total": train_loss}, {"total": val_loss, "dice": val_dice, "acc": val_acc})

        # 简单的综合评分：Dice 与 Acc 平均，便于挑选最佳权重
        score = (val_dice + val_acc) / 2
        if score > best_score and cfg.log.save_ckpt:
            best_score = score
            ckpt_path = os.path.join(cfg.log.save_dir, "best.pt")
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg)}, ckpt_path)
            LOGGER.info("Saved best checkpoint to %s", ckpt_path)

    LOGGER.info("Training finished. Best score=%.4f", best_score)


if __name__ == "__main__":
    main()
