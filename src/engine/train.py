"""Hydra 驱动的多任务训练循环（分割 + 分类）。

- 配置统一由 configs/defaults.yaml 读取，便于实验复现；
- 支持 dummy 数据与真实数据集两种路径，方便快速冒烟测试；
- 日志/可视化/权重保存逻辑集中于此，训练脚本更简洁。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple
import time

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from src.datasets import DummyPathologyDataset, PathologyDataset, StratifiedBatchSampler
from src.datasets.transforms import BaseAug
from src.losses import GradNorm, MultiTaskLoss
from src.models.multitask_model import MultiTaskModel
from src.utils.metrics import accuracy_from_logits, dice_coefficient
from src.utils.misc import ensure_dir, get_device, set_seed
from src.utils.visualizer import LossVisualizer

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def compute_binary_roc(probs: torch.Tensor, labels: torch.Tensor) -> tuple[list, list, list, float]:
    """计算二分类 ROC 曲线与 AUC。"""
    probs_np = probs.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


def save_roc_curve(fpr: list, tpr: list, roc_auc: float, out_path: str) -> None:
    """保存 ROC 曲线图，便于后续查看。"""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def resolve_weighting(cfg_loss: DictConfig) -> str:
    """解析多任务权重策略，兼容旧的 use_uncertainty 配置。"""
    weighting = str(cfg_loss.get("weighting", "")).strip().lower()
    if not weighting:
        weighting = "uncertainty" if bool(cfg_loss.get("use_uncertainty", False)) else "fixed"
    if weighting not in {"fixed", "uncertainty", "gradnorm"}:
        raise ValueError(f"Unknown loss weighting strategy: {weighting}")
    return weighting


def get_shared_parameters(model: torch.nn.Module) -> torch.Tensor:
    """返回共享参数张量，供 GradNorm 计算梯度范数使用。"""
    if hasattr(model, "encoder"):
        return next(model.encoder.parameters())
    return next(model.parameters())


def build_datasets(cfg: DictConfig) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """根据配置构建训练/验证集，支持 dummy 与真实数据两条路径。

    - use_dummy=True：走合成数据，保证“零数据依赖”也能跑通训练管线；
    - use_dummy=False：读取真实路径，接口与 CAMELYON 系列兼容，便于日后替换。"""
    normalize_mode = cfg.data.get("normalize", "none")
    normalize_mean = None
    normalize_std = None
    if normalize_mode == "imagenet":
        normalize_mean = cfg.data.get("imagenet_mean", [0.485, 0.456, 0.406])
        normalize_std = cfg.data.get("imagenet_std", [0.229, 0.224, 0.225])

    if cfg.data.get("use_dummy", True):
        train_ds = DummyPathologyDataset(
            length=cfg.data.dummy.train_samples,
            image_size=cfg.data.dummy.image_size,
            num_classes=cfg.model.num_classes,
            seed=cfg.seed,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        val_ds = DummyPathologyDataset(
            length=cfg.data.dummy.val_samples,
            image_size=cfg.data.dummy.image_size,
            num_classes=cfg.model.num_classes,
            seed=cfg.seed + 1,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    else:
        # 真实数据路径：采用统一 BaseAug（几何 + 轻度 elastic）增强，可由配置控制
        aug_cfg = cfg.get("aug")
        if aug_cfg is None:
            transform = BaseAug()
        else:
            affine_kwargs = OmegaConf.to_container(aug_cfg.get("affine", {}), resolve=True)
            elastic_kwargs = OmegaConf.to_container(aug_cfg.get("elastic", {}), resolve=True)
            transform = BaseAug(
                flip_p=float(aug_cfg.get("flip_p", 0.5)),
                affine_kwargs=affine_kwargs,
                elastic_kwargs=elastic_kwargs,
            )
        train_images = resolve_path(cfg.data.train_images)
        train_masks = resolve_path(cfg.data.train_masks)
        train_labels = resolve_path(cfg.data.train_labels)
        val_images = resolve_path(cfg.data.val_images)
        val_masks = resolve_path(cfg.data.val_masks)
        val_labels = resolve_path(cfg.data.val_labels)
        train_ds = PathologyDataset(
            train_images,
            train_masks,
            train_labels,
            transform=transform,
            debug_log=False,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        val_ds = PathologyDataset(
            val_images,
            val_masks,
            val_labels,
            transform=None,
            debug_log=False,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    return train_ds, val_ds


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    enable_seg = cfg.tasks.get("enable_seg", True)
    enable_cls = cfg.tasks.get("enable_cls", True)
    if not enable_seg and not enable_cls:
        raise ValueError("At least one task must be enabled via cfg.tasks.")

    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(cfg.log.get("save_dir", os.getcwd()))
    timing_dir = resolve_path(cfg.log.get("save_timing_dir", run_dir))
    params_dir = resolve_path(cfg.log.get("params_dir", run_dir))
    roc_dir = resolve_path(cfg.log.get("roc_dir", run_dir))
    ensure_dir(run_dir)
    ensure_dir(timing_dir)
    ensure_dir(params_dir)
    ensure_dir(roc_dir)
    log_path = os.path.join(run_dir, f"train_{file_stamp}.log")
    timing_log_path = os.path.join(timing_dir, f"timing_{file_stamp}.log")
    params_log_path = os.path.join(params_dir, f"metrics_{file_stamp}.log")

    # Hydra 在 main 之前已经注册了 logger handler，因此这里强制覆写以保证文件日志生效
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    timing_logger = logging.getLogger("timing")
    timing_logger.setLevel(logging.INFO)
    timing_logger.handlers.clear()
    timing_logger.addHandler(logging.FileHandler(timing_log_path))
    timing_logger.propagate = False
    timing_logger.info(
        "stage timings per batch: dataload, to_device, encoder, decoder, cls_head, forward_total, loss, backward, optimizer"
    )
    # 指标日志：独立文件、只落盘不打印到控制台，便于后续集中分析。
    params_logger = logging.getLogger("metrics_params")
    params_logger.setLevel(logging.INFO)
    params_logger.handlers.clear()
    params_logger.addHandler(logging.FileHandler(params_log_path))
    params_logger.propagate = False
    LOGGER.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))

    weighting = resolve_weighting(cfg.loss)
    if weighting == "gradnorm" and (enable_seg + enable_cls) < 2:
        LOGGER.warning("GradNorm requires at least two tasks; fallback to fixed weighting.")
        weighting = "fixed"
    task_names = []
    if enable_seg:
        task_names.append("seg")
    if enable_cls:
        task_names.append("cls")
    LOGGER.info("Loss weighting strategy: %s", weighting)

    train_ds, val_ds = build_datasets(cfg)  # 数据集切换由配置控制，便于做 ablation
    if cfg.loader.get("stratified_train", False):
        # 训练同样使用分层采样，保证每个 batch 的正负样本比例稳定
        if not hasattr(train_ds, "label_list"):
            raise ValueError("Training dataset must expose label_list for stratified sampling.")
        train_sampler = StratifiedBatchSampler(
            labels=getattr(train_ds, "label_list"),
            batch_size=cfg.batch_size,
            positive_ratio=cfg.loader.get("train_positive_ratio", 0.5),
            drop_last=cfg.loader.get("train_drop_last", False),
            seed=cfg.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=cfg.num_workers)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
    if cfg.loader.get("stratified_val", False):
        if not hasattr(val_ds, "label_list"):
            raise ValueError("Validation dataset must expose label_list for stratified sampling.")
        val_sampler = StratifiedBatchSampler(
            labels=getattr(val_ds, "label_list"),
            batch_size=cfg.batch_size,
            positive_ratio=cfg.loader.get("val_positive_ratio", 0.33),
            drop_last=False,
            seed=cfg.seed,
        )
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=cfg.num_workers)
    else:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

    # 核心模型：共享 ResNet 编码器 + U-Net 解码器 + 分类头
    model = MultiTaskModel(
        backbone_name=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        seg_upsample_to_input=cfg.model.seg_upsample_to_input,
        encoder_pretrained=cfg.model.get("pretrained", False),
        enable_seg=enable_seg,
        enable_cls=enable_cls,
        mobilenet_width_mult=cfg.model.get("mobilenet_width_mult", 0.4),
        use_light_decoder=cfg.model.get("use_light_decoder"),
        attention=cfg.model.get("attention", "none"),
        attention_location=cfg.model.get("attention_location", "decoder"),
        encoder_attention=cfg.model.get("encoder_attention"),
        decoder_attention=cfg.model.get("decoder_attention"),
        attention_reduction=cfg.model.get("attention_reduction", 16),
        decoder_attention_layers=cfg.model.get("decoder_attention_layers"),
    ).to(device)

    # 固定权重多任务损失：与 optional_modules/dynamic_loss 对比时可作基线
    criterion = MultiTaskLoss(
        seg_weight=cfg.loss.seg_weight,
        cls_weight=cfg.loss.cls_weight,
        dice_weight=cfg.loss.dice_weight,
        cls_type=cfg.loss.cls,
        enable_seg=enable_seg,
        enable_cls=enable_cls,
        weighting=weighting,
        use_uncertainty=weighting == "uncertainty",
        uncertainty_init=float(cfg.loss.get("uncertainty_init", 0.0)),
    )
    gradnorm = None
    shared_parameters = None
    optimizer_params = list(model.parameters())
    if weighting == "uncertainty":
        optimizer_params += list(criterion.parameters())
    elif weighting == "gradnorm":
        gradnorm = GradNorm(
            num_tasks=len(task_names),
            alpha=float(cfg.loss.get("gradnorm_alpha", 1.5)),
        ).to(device)
        shared_parameters = get_shared_parameters(model)
        optimizer_params += list(gradnorm.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=cfg.lr)
    scheduler = None  # 预留：若未来引入调度器，此处替换并保存状态
    visualizer = LossVisualizer(save_dir=run_dir)

    best_score = -1.0
    history = {"train": [], "val": []}
    last_step_end = time.perf_counter()
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0  # 累积训练损失
        running_seg_loss = 0.0
        running_cls_loss = 0.0
        for step, (images, masks, labels, _) in enumerate(train_loader, start=1):
            t_after_load = time.perf_counter()
            dataload_time = t_after_load - last_step_end

            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            t_after_device = time.perf_counter()

            optimizer.zero_grad()
            timing_parts: dict[str, float] = {}
            t_forward_start = time.perf_counter()
            seg_logits, cls_logits = model(images, timings=timing_parts)
            t_forward_end = time.perf_counter()

            if weighting == "gradnorm":
                # GradNorm: match task gradient norms to balance training; backward happens inside.
                task_losses, _, parts = criterion.compute_task_losses(
                    seg_logits=seg_logits,
                    seg_targets=masks,
                    cls_logits=cls_logits,
                    cls_targets=labels,
                )
                t_after_loss = time.perf_counter()
                loss = gradnorm.update_and_weight(task_losses, shared_parameters)
                t_after_backward = time.perf_counter()
                for name, value in zip(task_names, gradnorm.weights.detach().tolist()):
                    parts[f"{name}_weight_dyn"] = float(value)
            else:
                loss, parts = criterion(seg_logits, masks, cls_logits, labels)
                t_after_loss = time.perf_counter()
                loss.backward()
                t_after_backward = time.perf_counter()

            optimizer.step()
            t_after_opt = time.perf_counter()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            if enable_seg:
                running_seg_loss += parts["seg_loss"] * batch_size
            if enable_cls:
                running_cls_loss += parts["cls_loss"] * batch_size
            timing_logger.info(
                "epoch=%d step=%d dataload=%.6f to_device=%.6f encoder=%.6f decoder=%.6f cls_head=%.6f forward_total=%.6f loss=%.6f backward=%.6f optimizer=%.6f",
                epoch,
                step,
                dataload_time,
                t_after_device - t_after_load,
                timing_parts.get("encoder", 0.0),
                timing_parts.get("decoder", 0.0),
                timing_parts.get("cls_head", 0.0),
                t_forward_end - t_forward_start,
                t_after_loss - t_forward_end,
                t_after_backward - t_after_loss,
                t_after_opt - t_after_backward,
            )
            last_step_end = t_after_opt
        dataset_len = max(len(train_ds), 1)
        train_loss = running_loss / dataset_len
        train_seg_loss = running_seg_loss / dataset_len if enable_seg else 0.0
        train_cls_loss = running_cls_loss / dataset_len if enable_cls else 0.0

        model.eval()
        val_loss, dice_sum, acc_sum, batches = 0.0, 0.0, 0.0, 0  # 验证集指标累计
        val_seg_loss_sum, val_cls_loss_sum = 0.0, 0.0
        # ROC 统计：验证阶段收集分类概率与标签。
        roc_probs: list[torch.Tensor] = []
        roc_labels: list[torch.Tensor] = []
        roc_supported = True
        with torch.no_grad():
            for v_step, (images, masks, labels, names) in enumerate(val_loader, start=1):
                t_after_load = time.perf_counter()
                dataload_time = t_after_load - last_step_end

                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                t_after_device = time.perf_counter()

                timing_parts: dict[str, float] = {}
                t_forward_start = time.perf_counter()
                seg_logits, cls_logits = model(images, timings=timing_parts)
                t_forward_end = time.perf_counter()

                if weighting == "gradnorm" and gradnorm is not None:
                    # GradNorm eval: reuse learned weights for a weighted total loss.
                    task_losses, _, parts = criterion.compute_task_losses(
                        seg_logits=seg_logits,
                        seg_targets=masks,
                        cls_logits=cls_logits,
                        cls_targets=labels,
                    )
                    loss = gradnorm.weight_losses(task_losses)
                    for name, value in zip(task_names, gradnorm.weights.detach().tolist()):
                        parts[f"{name}_weight_dyn"] = float(value)
                else:
                    loss, parts = criterion(seg_logits, masks, cls_logits, labels)
                if enable_cls and parts["cls_loss"] > 10.0:
                    LOGGER.warning(
                        (
                            "High validation cls_loss=%.4f at epoch %d batch %d. "
                            "Label range=(%.2f, %.2f), logits range=(%.2f, %.2f), samples=%s"
                        ),
                        parts["cls_loss"],
                        epoch,
                        batches + 1,
                        labels.min().item(),
                        labels.max().item(),
                        cls_logits.min().item(),
                        cls_logits.max().item(),
                        ",".join(map(str, names)),
                    )
                t_after_loss = time.perf_counter()

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                if enable_seg:
                    val_seg_loss_sum += parts["seg_loss"] * batch_size
                    dice = dice_coefficient(torch.sigmoid(seg_logits), masks).mean().item()
                    dice_sum += dice
                if enable_cls:
                    val_cls_loss_sum += parts["cls_loss"] * batch_size
                    acc = accuracy_from_logits(cls_logits, labels)
                    acc_sum += acc
                    # 收集 ROC 输入（仅支持二分类：1 个 logit 或 2 类 softmax）。
                    if cls_logits is not None:
                        if cls_logits.ndim == 2 and cls_logits.shape[1] == 1:
                            probs = torch.sigmoid(cls_logits).view(-1)
                        elif cls_logits.ndim == 2 and cls_logits.shape[1] == 2:
                            probs = torch.softmax(cls_logits, dim=1)[:, 1]
                        else:
                            roc_supported = False
                            probs = None
                        if probs is not None:
                            roc_probs.append(probs.detach().cpu())
                            roc_labels.append(labels.detach().cpu())
                batches += 1
                timing_logger.info(
                    "val epoch=%d step=%d dataload=%.6f to_device=%.6f encoder=%.6f decoder=%.6f cls_head=%.6f forward_total=%.6f loss=%.6f",
                    epoch,
                    v_step,
                    dataload_time,
                    t_after_device - t_after_load,
                    timing_parts.get("encoder", 0.0),
                    timing_parts.get("decoder", 0.0),
                    timing_parts.get("cls_head", 0.0),
                    t_forward_end - t_forward_start,
                    t_after_loss - t_forward_end,
                )
                last_step_end = t_after_loss
        val_len = max(len(val_ds), 1)
        val_loss /= val_len
        val_seg_loss = val_seg_loss_sum / val_len if enable_seg else 0.0
        val_cls_loss = val_cls_loss_sum / val_len if enable_cls else 0.0
        val_dice = dice_sum / max(batches, 1) if enable_seg else 0.0
        val_acc = acc_sum / max(batches, 1) if enable_cls else 0.0
        # ROC：计算曲线并保存图，同时把参数写入指标日志（不输出到控制台）。
        roc_payload = {}
        if enable_cls and roc_supported and roc_probs:
            all_probs = torch.cat(roc_probs, dim=0)
            all_labels = torch.cat(roc_labels, dim=0)
            fpr, tpr, thresholds, roc_auc = compute_binary_roc(all_probs, all_labels)
            roc_plot_path = os.path.join(roc_dir, f"roc_{file_stamp}_epoch{epoch:03d}.png")
            save_roc_curve(fpr, tpr, roc_auc, roc_plot_path)
            roc_payload = {
                "roc_auc": roc_auc,
                "roc_fpr": [round(x, 6) for x in fpr],
                "roc_tpr": [round(x, 6) for x in tpr],
                "roc_thresholds": [round(x, 6) for x in thresholds],
                "roc_plot": roc_plot_path,
            }
        elif enable_cls and not roc_supported:
            roc_payload = {"roc_note": "ROC 仅支持二分类输出（1 个 logit 或 2 类 softmax）。"}
        seg_train_log = f"{train_seg_loss:.4f}" if enable_seg else "NA"
        cls_train_log = f"{train_cls_loss:.4f}" if enable_cls else "NA"
        seg_val_log = f"{val_seg_loss:.4f}" if enable_seg else "NA"
        cls_val_log = f"{val_cls_loss:.4f}" if enable_cls else "NA"
        dice_log = f"| val_dice={val_dice:.4f} " if enable_seg else ""
        acc_log = f"| val_acc={val_acc:.4f}" if enable_cls else ""
        LOGGER.info(
            (
                "Epoch %d/%d | train_loss=%.4f (seg=%s, cls=%s) "
                "| val_loss=%.4f (seg=%s, cls=%s) %s%s"
            ),
            epoch,
            cfg.num_epochs,
            train_loss,
            seg_train_log,
            cls_train_log,
            val_loss,
            seg_val_log,
            cls_val_log,
            dice_log,
            acc_log,
        )
        visualizer.update(
            epoch,
            {"total": train_loss},
            {"total": val_loss, "dice": val_dice, "acc": val_acc},
        )
        log_sigma_seg = None
        log_sigma_cls = None
        weight_seg_dyn = None
        weight_cls_dyn = None
        if weighting == "uncertainty":
            if getattr(criterion, "log_sigma_seg", None) is not None:
                log_sigma_seg = float(criterion.log_sigma_seg.item())
                weight_seg_dyn = float(torch.exp(-criterion.log_sigma_seg).item())
            if getattr(criterion, "log_sigma_cls", None) is not None:
                log_sigma_cls = float(criterion.log_sigma_cls.item())
                weight_cls_dyn = float(torch.exp(-criterion.log_sigma_cls).item())
        elif weighting == "gradnorm" and gradnorm is not None:
            weight_map = dict(zip(task_names, gradnorm.weights.detach().tolist()))
            weight_seg_dyn = weight_map.get("seg")
            weight_cls_dyn = weight_map.get("cls")
        # 独立指标日志：每行 JSON，便于脚本解析，且不输出到控制台。
        params_logger.info(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_seg_loss": train_seg_loss if enable_seg else None,
                    "train_cls_loss": train_cls_loss if enable_cls else None,
                    "val_loss": val_loss,
                    "val_seg_loss": val_seg_loss if enable_seg else None,
                    "val_cls_loss": val_cls_loss if enable_cls else None,
                    "val_dice": val_dice if enable_seg else None,
                    "val_acc": val_acc if enable_cls else None,
                    "loss_weighting": weighting,
                    "log_sigma_seg": log_sigma_seg,
                    "log_sigma_cls": log_sigma_cls,
                    "weight_seg_dyn": weight_seg_dyn,
                    "weight_cls_dyn": weight_cls_dyn,
                    **roc_payload,
                },
                ensure_ascii=True,
            )
        )
        history["train"].append(
            {
                "epoch": epoch,
                "total_loss": train_loss,
                "seg_loss": train_seg_loss if enable_seg else None,
                "cls_loss": train_cls_loss if enable_cls else None,
            }
        )
        history["val"].append(
            {
                "epoch": epoch,
                "total_loss": val_loss,
                "seg_loss": val_seg_loss if enable_seg else None,
                "cls_loss": val_cls_loss if enable_cls else None,
                "dice": val_dice if enable_seg else None,
                "acc": val_acc if enable_cls else None,
            }
        )

        # 简单的综合评分：Dice 与 Acc 平均，便于挑选最佳权重
        metrics_for_score = []
        if enable_seg:
            metrics_for_score.append(val_dice)
        if enable_cls:
            metrics_for_score.append(val_acc)
        score = sum(metrics_for_score) / len(metrics_for_score)
        if score > best_score and cfg.log.save_ckpt:
            best_score = score
            ckpt_path = os.path.join(run_dir, "best.pt")
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "gradnorm_state_dict": gradnorm.state_dict() if gradnorm is not None else None,
                "epoch": epoch,
                "best_score": score,
                "loss_weighting": weighting,
                "metrics": {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice if enable_seg else None,
                    "val_acc": val_acc if enable_cls else None,
                },
                "history": history,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "random_seed": cfg.seed,
            }
            torch.save(checkpoint_payload, ckpt_path)
            summary_lines = [
                "Best checkpoint summary",
                f"saved_at={datetime.now().isoformat()}",
                f"epoch={epoch}",
                f"score={score:.4f}",
                f"val_dice={(f'{val_dice:.4f}' if enable_seg else 'NA')}",
                f"val_acc={(f'{val_acc:.4f}' if enable_cls else 'NA')}",
                f"train_loss={train_loss:.4f}",
                f"val_loss={val_loss:.4f}",
                "payload_keys=[model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_score, metrics, history, config, random_seed]",
                f"ckpt_path={ckpt_path}",
            ]
            summary_path = Path(run_dir) / "best_checkpoint.txt"
            summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
            LOGGER.info("Saved best checkpoint to %s", ckpt_path)
            LOGGER.info("Updated best checkpoint summary at %s", summary_path)

    LOGGER.info("Training finished. Best score=%.4f", best_score)


if __name__ == "__main__":
    main()
