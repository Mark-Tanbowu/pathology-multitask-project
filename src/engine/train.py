"""Hydra 驱动的多任务训练循环（分割 + 分类）。

- 配置统一由 configs/defaults.yaml 读取，便于实验复现；
- 支持 dummy 数据与真实数据集两种路径，方便快速冒烟测试；
- 日志/可视化/权重保存逻辑集中于此，训练脚本更简洁。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets import (
    DummyPathologyDataset,
    PathologyDataset,
    SlideCoordsDataset,
    StratifiedBatchSampler,
    WsiPatchDataset,
)
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


def running_mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) for a 1-D float list; empty list maps to (0, 0)."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def safe_zscore(value: float, mean: float, std: float, std_eps: float) -> float:
    """Stable z-score; return 0 when std is too small."""
    if std <= float(std_eps):
        return 0.0
    return float((value - mean) / std)


def topk_mean_score(values: list[float], k: int) -> float:
    """Return mean of top-k values; k is clipped to [1, len(values)]."""
    if not values:
        return 0.0
    k_use = min(max(int(k), 1), len(values))
    topk = sorted(values, reverse=True)[:k_use]
    return float(sum(topk) / float(k_use))


def binary_metrics_from_probs(
    labels: torch.Tensor,
    probs: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Compute binary classification metrics at a fixed threshold."""
    y_true = labels.float().view(-1)
    y_pred = (probs.float().view(-1) > float(threshold)).float()
    tp = float(((y_pred == 1.0) & (y_true == 1.0)).sum().item())
    tn = float(((y_pred == 0.0) & (y_true == 0.0)).sum().item())
    fp = float(((y_pred == 1.0) & (y_true == 0.0)).sum().item())
    fn = float(((y_pred == 0.0) & (y_true == 1.0)).sum().item())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    balanced_acc = 0.5 * (recall + specificity)
    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def infer_slide_id(sample_name: str) -> str:
    """从样本名中提取 slide_id，兼容 `slide_x_y` 与普通文件名。"""
    stem = Path(sample_name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
        return parts[0]
    return stem


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_geometry_multiplier_by_slide(
    dataset: SlideCoordsDataset,
    geometry_field: str,
    gamma: float,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """构建 slide 级几何重采样倍率（默认更关注小病灶，倍率作用于正样本权重）。"""
    pos_values: list[float] = []
    per_slide_raw: dict[str, float] = {}
    for row in dataset.rows:
        slide_id = str(row.get("slide_id", ""))
        slide_label = row.get("slide_label")
        if not slide_id or int(slide_label or 0) <= 0:
            continue
        geom_val = _safe_float(row.get(geometry_field), default=0.0)
        if geom_val <= 0.0:
            continue
        pos_values.append(float(geom_val))
        per_slide_raw[slide_id] = float(geom_val)

    if not pos_values:
        return {}, {"ref_value": 0.0, "used_slides": 0.0}

    ref_value = float(np.median(np.asarray(pos_values, dtype=np.float64)))
    gamma_use = max(0.0, float(gamma))
    clip_lo = max(0.0, float(clip_min))
    clip_hi = max(float(clip_max), clip_lo + 1e-6)
    eps_use = max(float(eps), 1.0e-12)

    per_slide_multiplier: dict[str, float] = {}
    for slide_id, geom_val in per_slide_raw.items():
        mult = float((ref_value / max(geom_val, eps_use)) ** gamma_use)
        mult = float(max(clip_lo, min(clip_hi, mult)))
        per_slide_multiplier[slide_id] = mult

    return per_slide_multiplier, {"ref_value": ref_value, "used_slides": float(len(per_slide_raw))}


def build_slide_binary_weights(
    dataset: SlideCoordsDataset,
    geometry_cfg: dict[str, object] | None = None,
) -> tuple[torch.Tensor, int, int, dict[str, float]]:
    """为 SlideCoordsDataset 构建二分类重采样权重。

    - 基础策略：正负总权重各占 0.5；
    - 可选几何增强：按 slide 级几何统计调整“正样本权重倍率”。
    """
    all_labels: list[np.ndarray] = []
    row_refs: list[dict[str, Any]] = []
    pos_count = 0
    neg_count = 0
    for row in dataset.rows:
        coords = np.load(row["coords_path"])
        labels = (coords[:, 2] > 0).astype(np.int64)
        all_labels.append(labels)
        row_refs.append(row)
        pos_count += int(labels.sum())
        neg_count += int(labels.size - labels.sum())

    diag: dict[str, float] = {"geom_enabled": 0.0, "geom_ref_value": 0.0, "geom_used_slides": 0.0}
    if pos_count == 0 or neg_count == 0:
        weights = np.ones(sum(len(x) for x in all_labels), dtype=np.float32)
        return torch.from_numpy(weights), pos_count, neg_count, diag

    pos_w = 0.5 / float(pos_count)
    neg_w = 0.5 / float(neg_count)
    slide_geom_mult: dict[str, float] = {}
    if geometry_cfg and bool(geometry_cfg.get("enabled", False)):
        geom_field = str(geometry_cfg.get("field", "tumor_area_ratio"))
        geom_gamma = float(geometry_cfg.get("gamma", 0.5))
        geom_eps = float(geometry_cfg.get("eps", 1.0e-8))
        geom_clip_min = float(geometry_cfg.get("clip_min", 0.5))
        geom_clip_max = float(geometry_cfg.get("clip_max", 2.0))
        slide_geom_mult, geom_diag = _build_geometry_multiplier_by_slide(
            dataset=dataset,
            geometry_field=geom_field,
            gamma=geom_gamma,
            eps=geom_eps,
            clip_min=geom_clip_min,
            clip_max=geom_clip_max,
        )
        diag["geom_enabled"] = 1.0 if slide_geom_mult else 0.0
        diag["geom_ref_value"] = float(geom_diag["ref_value"])
        diag["geom_used_slides"] = float(geom_diag["used_slides"])

    weight_chunks: list[np.ndarray] = []
    for labels, row in zip(all_labels, row_refs):
        slide_id = str(row.get("slide_id", ""))
        pos_mult = float(slide_geom_mult.get(slide_id, 1.0))
        chunk = np.where(labels > 0, pos_w * pos_mult, neg_w).astype(np.float32)
        weight_chunks.append(chunk)
    weights = np.concatenate(weight_chunks, axis=0)
    return torch.from_numpy(weights), pos_count, neg_count, diag


def build_slide_label_map(dataset: torch.utils.data.Dataset) -> tuple[dict[str, int], int]:
    """从数据集元信息构建 slide_id -> slide_label 映射。"""
    rows = getattr(dataset, "rows", None)
    if not rows:
        return {}, 0

    label_map: dict[str, int] = {}
    conflict_count = 0
    for row in rows:
        slide_id = str(row.get("slide_id", ""))
        slide_label = row.get("slide_label")
        if not slide_id or slide_label is None:
            continue
        try:
            label_int = int(slide_label)
        except (TypeError, ValueError):
            continue
        prev = label_map.get(slide_id)
        if prev is not None and prev != label_int:
            conflict_count += 1
            continue
        label_map[slide_id] = label_int
    return label_map, conflict_count


def build_slide_float_map(dataset: torch.utils.data.Dataset, field: str) -> dict[str, float]:
    """从数据集元信息构建 slide_id -> float 字段映射（用于几何统计字段读取）。"""
    rows = getattr(dataset, "rows", None)
    if not rows:
        return {}
    out: dict[str, float] = {}
    for row in rows:
        slide_id = str(row.get("slide_id", ""))
        if not slide_id:
            continue
        value = _safe_float(row.get(field), default=0.0)
        out[slide_id] = value
    return out


def geometry_positive_bin(value: float, bins: list[float]) -> str:
    """按阳性病灶面积占比分桶，返回稳定分桶名。"""
    if value <= 0.0:
        return "pos_unknown"
    for edge in bins:
        if value < edge:
            return f"pos_lt_{edge:.6f}"
    return f"pos_ge_{bins[-1]:.6f}" if bins else "pos_all"


def compute_geometry_stability_score(
    slide_ids: list[str],
    slide_probs: torch.Tensor,
    slide_labels: torch.Tensor,
    threshold: float,
    geometry_map: dict[str, float],
    pos_bins: list[float],
    min_bin_slides: int,
) -> tuple[float | None, dict[str, float]]:
    """计算几何稳定分数：
    - normal bin: specificity；
    - positive bins: recall；
    最终取可用分桶的宏平均。
    """
    normal_probs: list[float] = []
    normal_labels: list[float] = []
    pos_bin_probs: dict[str, list[float]] = {}
    pos_bin_labels: dict[str, list[float]] = {}
    for sid, prob, label in zip(
        slide_ids,
        slide_probs.detach().cpu().tolist(),
        slide_labels.detach().cpu().tolist(),
    ):
        if int(label) <= 0:
            normal_probs.append(float(prob))
            normal_labels.append(float(label))
            continue
        gval = float(geometry_map.get(sid, 0.0))
        bin_name = geometry_positive_bin(gval, pos_bins)
        pos_bin_probs.setdefault(bin_name, []).append(float(prob))
        pos_bin_labels.setdefault(bin_name, []).append(float(label))

    metrics: dict[str, float] = {}
    macro_parts: list[float] = []
    min_count = max(int(min_bin_slides), 1)
    if len(normal_probs) >= min_count:
        n_probs = torch.tensor(normal_probs, dtype=torch.float32)
        n_labels = torch.tensor(normal_labels, dtype=torch.float32)
        n_metrics = binary_metrics_from_probs(n_labels, n_probs, threshold)
        metrics["normal_specificity"] = float(n_metrics["specificity"])
        macro_parts.append(float(n_metrics["specificity"]))

    for bin_name, probs in sorted(pos_bin_probs.items()):
        if len(probs) < min_count:
            continue
        b_probs = torch.tensor(probs, dtype=torch.float32)
        b_labels = torch.tensor(pos_bin_labels[bin_name], dtype=torch.float32)
        b_metrics = binary_metrics_from_probs(b_labels, b_probs, threshold)
        metrics[f"{bin_name}_recall"] = float(b_metrics["recall"])
        macro_parts.append(float(b_metrics["recall"]))

    if not macro_parts:
        return None, metrics
    return float(sum(macro_parts) / len(macro_parts)), metrics


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
    elif cfg.data.get("use_slide_manifest", False):
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
        train_slide_manifest_cfg = cfg.data.get("train_slide_manifest")
        if not train_slide_manifest_cfg:
            raise ValueError(
                "配置缺失：已启用 use_slide_manifest=true，但未设置 data.train_slide_manifest。"
            )
        val_slide_manifest_cfg = cfg.data.get("val_slide_manifest")
        if not val_slide_manifest_cfg:
            raise ValueError(
                "配置缺失：已启用 use_slide_manifest=true，但未设置 data.val_slide_manifest。"
            )
        train_prepare_cfg = cfg.get("prepare_train")
        if train_prepare_cfg is None:
            raise ValueError(
                "配置缺失：已启用 use_slide_manifest=true，但未设置 prepare_train。"
            )
        val_prepare_cfg = cfg.get("prepare_val")
        if val_prepare_cfg is None:
            raise ValueError(
                "配置缺失：已启用 use_slide_manifest=true，但未设置 prepare_val。"
            )
        train_masks_dir_cfg = train_prepare_cfg.get("masks_dir")
        if not train_masks_dir_cfg:
            raise ValueError("配置缺失：prepare_train.masks_dir 不能为空。")
        val_masks_dir_cfg = val_prepare_cfg.get("masks_dir")
        if not val_masks_dir_cfg:
            raise ValueError("配置缺失：prepare_val.masks_dir 不能为空。")
        slide_manifest = resolve_path(str(train_slide_manifest_cfg))
        val_slide_manifest = resolve_path(str(val_slide_manifest_cfg))
        train_masks_dir = resolve_path(str(train_masks_dir_cfg))
        val_masks_dir = resolve_path(str(val_masks_dir_cfg))
        train_mask_suffix = str(train_prepare_cfg.get("mask_suffix", "_mask.tif"))
        val_mask_suffix = str(val_prepare_cfg.get("mask_suffix", "_mask.tif"))
        coords_cache_slides = int(cfg.data.get("coords_cache_slides", 2))
        cache_masks = bool(cfg.data.get("cache_masks", False))
        train_mask_level = train_prepare_cfg.get("mask_level", cfg.data.get("mask_level"))
        val_mask_level = val_prepare_cfg.get("mask_level", cfg.data.get("mask_level"))
        train_mask_max_size = train_prepare_cfg.get("mask_max_size", cfg.data.get("mask_max_size"))
        val_mask_max_size = val_prepare_cfg.get("mask_max_size", cfg.data.get("mask_max_size"))
        train_ds = SlideCoordsDataset(
            slide_manifest,
            train_masks_dir,
            mask_suffix=train_mask_suffix,
            coords_cache_slides=coords_cache_slides,
            cache_masks=cache_masks,
            mask_level=int(train_mask_level) if train_mask_level is not None else None,
            mask_max_size=int(train_mask_max_size) if train_mask_max_size is not None else None,
            transform=transform,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        val_ds = SlideCoordsDataset(
            val_slide_manifest,
            val_masks_dir,
            mask_suffix=val_mask_suffix,
            coords_cache_slides=coords_cache_slides,
            cache_masks=cache_masks,
            mask_level=int(val_mask_level) if val_mask_level is not None else None,
            mask_max_size=int(val_mask_max_size) if val_mask_max_size is not None else None,
            transform=None,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    elif cfg.data.get("use_manifest", False):
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
        train_manifest = resolve_path(cfg.data.get("train_manifest", cfg.prepare.manifest_path))
        val_manifest = resolve_path(cfg.data.get("val_manifest", cfg.prepare.manifest_path))
        masks_dir = resolve_path(cfg.prepare.masks_dir)
        mask_suffix = str(cfg.prepare.get("mask_suffix", "_mask.tif"))
        train_ds = WsiPatchDataset(
            train_manifest,
            masks_dir,
            mask_suffix=mask_suffix,
            transform=transform,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        val_ds = WsiPatchDataset(
            val_manifest,
            masks_dir,
            mask_suffix=mask_suffix,
            transform=None,
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
    wsi_scores_log_path = os.path.join(params_dir, f"wsi_scores_{file_stamp}.log")
    wsi_k_metrics_log_path = os.path.join(params_dir, f"wsi_k_metrics_{file_stamp}.log")
    wsi_geom_metrics_log_path = os.path.join(params_dir, f"wsi_geom_metrics_{file_stamp}.log")

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
    wsi_logger = logging.getLogger("metrics_wsi_scores")
    wsi_logger.setLevel(logging.INFO)
    wsi_logger.handlers.clear()
    wsi_logger.addHandler(logging.FileHandler(wsi_scores_log_path))
    wsi_logger.propagate = False
    wsi_k_logger = logging.getLogger("metrics_wsi_k")
    wsi_k_logger.setLevel(logging.INFO)
    wsi_k_logger.handlers.clear()
    wsi_k_logger.addHandler(logging.FileHandler(wsi_k_metrics_log_path))
    wsi_k_logger.propagate = False
    wsi_geom_logger = logging.getLogger("metrics_wsi_geom")
    wsi_geom_logger.setLevel(logging.INFO)
    wsi_geom_logger.handlers.clear()
    wsi_geom_logger.addHandler(logging.FileHandler(wsi_geom_metrics_log_path))
    wsi_geom_logger.propagate = False
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
    train_log_interval = int(cfg.log.get("train_log_interval", 20))
    val_log_interval = int(cfg.log.get("val_log_interval", 20))
    if cfg.data.get("use_slide_manifest", False) and cfg.loader.get("stratified_train", False):
        LOGGER.warning("SlideCoordsDataset does not support stratified sampling; disabling.")
        cfg.loader.stratified_train = False
    if cfg.data.get("use_slide_manifest", False) and cfg.loader.get("stratified_val", False):
        LOGGER.warning("SlideCoordsDataset does not support stratified sampling; disabling.")
        cfg.loader.stratified_val = False

    slide_weighted_sampling = bool(
        cfg.data.get("use_slide_manifest", False) and cfg.loader.get("slide_weighted_sampling", False)
    )
    if slide_weighted_sampling:
        geom_cfg_raw = cfg.loader.get("train_geometry_sampling", {})
        if isinstance(geom_cfg_raw, DictConfig):
            geom_cfg_raw = OmegaConf.to_container(geom_cfg_raw, resolve=True)
        geom_cfg = geom_cfg_raw if isinstance(geom_cfg_raw, dict) else {}
        geom_weight_cfg = {
            "enabled": bool(geom_cfg.get("enabled", False)),
            "field": str(geom_cfg.get("field", "tumor_area_ratio")),
            "gamma": float(geom_cfg.get("gamma", 0.5)),
            "eps": float(geom_cfg.get("eps", 1.0e-8)),
            "clip_min": float(geom_cfg.get("clip_min", 0.5)),
            "clip_max": float(geom_cfg.get("clip_max", 2.0)),
        }
        sample_weights, pos_count, neg_count, weight_diag = build_slide_binary_weights(
            train_ds, geometry_cfg=geom_weight_cfg
        )
        requested_samples = int(cfg.loader.get("slide_samples_per_epoch", len(train_ds)))
        if requested_samples <= 0:
            requested_samples = len(train_ds)
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=requested_samples,
            replacement=bool(cfg.loader.get("slide_weighted_replacement", True)),
            generator=generator,
        )
        LOGGER.info(
            (
                "Using weighted sampling for SlideCoordsDataset: "
                "samples_per_epoch=%d pool=%d pos=%d neg=%d"
            ),
            requested_samples,
            len(train_ds),
            pos_count,
            neg_count,
        )
        if float(weight_diag.get("geom_enabled", 0.0)) > 0.0:
            LOGGER.info(
                (
                    "Geometry-aware train sampling enabled: field=%s gamma=%.3f "
                    "clip=[%.3f, %.3f] ref=%.6f slides=%d"
                ),
                geom_weight_cfg["field"],
                geom_weight_cfg["gamma"],
                geom_weight_cfg["clip_min"],
                geom_weight_cfg["clip_max"],
                float(weight_diag.get("geom_ref_value", 0.0)),
                int(weight_diag.get("geom_used_slides", 0.0)),
            )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers
        )
    elif cfg.loader.get("stratified_train", False):
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
    val_slide_weighted_sampling = bool(
        cfg.data.get("use_slide_manifest", False) and cfg.loader.get("val_slide_weighted_sampling", False)
    )
    if val_slide_weighted_sampling:
        val_weights, val_pos_count, val_neg_count, _ = build_slide_binary_weights(val_ds)
        val_requested_samples = int(cfg.loader.get("val_slide_samples_per_epoch", len(val_ds)))
        if val_requested_samples <= 0:
            val_requested_samples = len(val_ds)
        val_generator = torch.Generator()
        val_generator.manual_seed(cfg.seed + 1)
        val_sampler = WeightedRandomSampler(
            weights=val_weights,
            num_samples=val_requested_samples,
            replacement=bool(cfg.loader.get("val_slide_weighted_replacement", True)),
            generator=val_generator,
        )
        LOGGER.info(
            (
                "Using weighted sampling for SlideCoordsDataset (val): "
                "samples_per_epoch=%d pool=%d pos=%d neg=%d"
            ),
            val_requested_samples,
            len(val_ds),
            val_pos_count,
            val_neg_count,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, sampler=val_sampler, num_workers=cfg.num_workers
        )
    elif cfg.loader.get("stratified_val", False):
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
    wsi_slide_label_map, wsi_manifest_label_conflicts = build_slide_label_map(val_ds)
    if enable_cls and wsi_slide_label_map:
        LOGGER.info(
            "Loaded %d slide-level labels from validation dataset metadata.",
            len(wsi_slide_label_map),
        )
    if wsi_manifest_label_conflicts > 0:
        LOGGER.warning(
            "Found %d conflicting slide-level labels in validation metadata; kept first occurrence.",
            wsi_manifest_label_conflicts,
        )
    has_wsi_slide_label_map = bool(wsi_slide_label_map)
    wsi_acc_threshold = float(cfg.eval.get("wsi_acc_threshold", 0.5))
    wsi_topk = int(cfg.eval.get("wsi_topk", 15))
    wsi_k_values_cfg = cfg.eval.get("wsi_k_values", [wsi_topk])
    if isinstance(wsi_k_values_cfg, (int, float, str)):
        wsi_k_values_cfg = [wsi_k_values_cfg]
    wsi_k_values = sorted({max(1, int(k)) for k in wsi_k_values_cfg})
    geom_val_cfg_raw = cfg.eval.get("geometry_validation", {})
    if isinstance(geom_val_cfg_raw, DictConfig):
        geom_val_cfg_raw = OmegaConf.to_container(geom_val_cfg_raw, resolve=True)
    geom_val_cfg = geom_val_cfg_raw if isinstance(geom_val_cfg_raw, dict) else {}
    geom_val_enabled = bool(geom_val_cfg.get("enabled", False))
    geom_val_field = str(geom_val_cfg.get("field", "tumor_area_ratio"))
    geom_val_bins_raw = geom_val_cfg.get("positive_bins", [0.0005, 0.002, 0.01])
    if isinstance(geom_val_bins_raw, (int, float, str)):
        geom_val_bins_raw = [geom_val_bins_raw]
    geom_val_positive_bins = sorted(
        {
            float(x)
            for x in geom_val_bins_raw
            if x not in (None, "") and float(x) > 0.0
        }
    )
    geom_val_min_bin_slides = int(geom_val_cfg.get("min_bin_slides", 3))
    val_slide_geometry_map = (
        build_slide_float_map(val_ds, geom_val_field)
        if (geom_val_enabled and cfg.data.get("use_slide_manifest", False))
        else {}
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

    best_cfg = cfg.eval.get("best_checkpoint", {})
    z_dice_weight = float(best_cfg.get("z_dice_weight", 0.4))
    z_dice_weight = max(0.0, min(1.0, z_dice_weight))
    z_wsi_auc_weight = 1.0 - z_dice_weight
    auc_gate_delta = float(best_cfg.get("auc_gate_delta", 0.002))
    geom_gate_delta = float(best_cfg.get("geom_gate_delta", 0.02))
    geom_tie_eps = float(best_cfg.get("geom_tie_eps", 1.0e-4))
    score_tie_eps = float(best_cfg.get("score_tie_eps", 1.0e-4))
    std_eps = float(best_cfg.get("std_eps", 1.0e-8))
    loss_tie_eps = float(best_cfg.get("loss_tie_eps", 1.0e-8))

    best_score = float("-inf")
    best_val_loss = float("inf")
    best_wsi_auc: float | None = None
    best_wsi_geom_stability: float | None = None
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
            if train_log_interval > 0 and (step % train_log_interval == 0 or step == 1):
                LOGGER.info(
                    "Train epoch=%d step=%d/%d loss=%.4f seg=%.4f cls=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    loss.item(),
                    parts.get("seg_loss", 0.0),
                    parts.get("cls_loss", 0.0),
                )
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
        # WSI 级聚合：使用 top-k mean 概率作为分类分数。
        wsi_probs_by_slide: dict[str, list[float]] = {}
        wsi_patch_count: dict[str, int] = {}
        wsi_max_sample: dict[str, str] = {}
        wsi_max_prob: dict[str, float] = {}
        wsi_labels: dict[str, int] = {}
        wsi_patch_label_conflict_count = 0
        wsi_missing_slide_label_count = 0
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
                            for sample_name, sample_prob, sample_label in zip(
                                names,
                                probs.detach().cpu().tolist(),
                                labels.detach().cpu().view(-1).tolist(),
                            ):
                                slide_id = infer_slide_id(str(sample_name))
                                label_int = (
                                    wsi_slide_label_map.get(slide_id)
                                    if has_wsi_slide_label_map
                                    else None
                                )
                                if label_int is None:
                                    # 兜底路径：若缺少 slide_label，则退化为 patch label（仅用于兼容旧数据）。
                                    if has_wsi_slide_label_map:
                                        wsi_missing_slide_label_count += 1
                                    label_int = int(sample_label)
                                    prev_label = wsi_labels.get(slide_id)
                                    if prev_label is not None and prev_label != label_int:
                                        wsi_patch_label_conflict_count += 1
                                        continue
                                wsi_labels[slide_id] = label_int
                                prob_val = float(sample_prob)
                                wsi_probs_by_slide.setdefault(slide_id, []).append(prob_val)
                                wsi_patch_count[slide_id] = wsi_patch_count.get(slide_id, 0) + 1
                                prev = wsi_max_prob.get(slide_id)
                                if prev is None or sample_prob > prev:
                                    wsi_max_prob[slide_id] = prob_val
                                    wsi_max_sample[slide_id] = str(sample_name)
                if val_log_interval > 0 and (v_step % val_log_interval == 0 or v_step == 1):
                    LOGGER.info(
                        "Val epoch=%d step=%d/%d loss=%.4f seg=%.4f cls=%.4f",
                        epoch,
                        v_step,
                        len(val_loader),
                        loss.item(),
                        parts.get("seg_loss", 0.0),
                        parts.get("cls_loss", 0.0),
                    )
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
        val_wsi_acc = None
        val_wsi_auc = None
        val_wsi_count = 0
        val_wsi_geom_stability = None
        val_wsi_geom_details: dict[str, float] = {}
        val_wsi_details: list[dict[str, object]] = []
        if enable_cls and wsi_probs_by_slide:
            slide_ids = sorted(wsi_probs_by_slide.keys())
            slide_probs = torch.tensor(
                [topk_mean_score(wsi_probs_by_slide[sid], wsi_topk) for sid in slide_ids],
                dtype=torch.float32,
            )
            slide_labels = torch.tensor([wsi_labels[sid] for sid in slide_ids], dtype=torch.float32)
            val_wsi_count = len(slide_ids)
            val_wsi_acc = float(
                ((slide_probs > wsi_acc_threshold).float() == slide_labels).float().mean().item()
            )
            if torch.unique(slide_labels).numel() >= 2:
                _, _, _, val_wsi_auc = compute_binary_roc(slide_probs, slide_labels)
            if geom_val_enabled and val_slide_geometry_map:
                val_wsi_geom_stability, val_wsi_geom_details = compute_geometry_stability_score(
                    slide_ids=slide_ids,
                    slide_probs=slide_probs,
                    slide_labels=slide_labels,
                    threshold=wsi_acc_threshold,
                    geometry_map=val_slide_geometry_map,
                    pos_bins=geom_val_positive_bins,
                    min_bin_slides=geom_val_min_bin_slides,
                )
                geom_payload = {
                    "epoch": epoch,
                    "field": geom_val_field,
                    "threshold": wsi_acc_threshold,
                    "positive_bins": geom_val_positive_bins,
                    "min_bin_slides": geom_val_min_bin_slides,
                    "score": val_wsi_geom_stability,
                    **val_wsi_geom_details,
                }
                wsi_geom_logger.info(json.dumps(geom_payload, ensure_ascii=True))
                LOGGER.info(
                    "WSI geometry stability epoch=%d score=%s field=%s",
                    epoch,
                    (
                        f"{float(val_wsi_geom_stability):.4f}"
                        if val_wsi_geom_stability is not None
                        else "NA"
                    ),
                    geom_val_field,
                )

            for k in wsi_k_values:
                k_probs = torch.tensor(
                    [topk_mean_score(wsi_probs_by_slide[sid], k) for sid in slide_ids],
                    dtype=torch.float32,
                )
                k_metrics = binary_metrics_from_probs(slide_labels, k_probs, wsi_acc_threshold)
                k_payload = {
                    "epoch": epoch,
                    "k": int(k),
                    "threshold": wsi_acc_threshold,
                    "num_slides": val_wsi_count,
                    "acc": float(k_metrics["acc"]),
                    "balanced_acc": float(k_metrics["balanced_acc"]),
                    "f1": float(k_metrics["f1"]),
                    "recall": float(k_metrics["recall"]),
                    "specificity": float(k_metrics["specificity"]),
                    "tp": int(k_metrics["tp"]),
                    "tn": int(k_metrics["tn"]),
                    "fp": int(k_metrics["fp"]),
                    "fn": int(k_metrics["fn"]),
                }
                wsi_k_logger.info(json.dumps(k_payload, ensure_ascii=True))
                LOGGER.info(
                    (
                        "WSI-K epoch=%d k=%d thr=%.3f slides=%d "
                        "acc=%.4f bacc=%.4f f1=%.4f recall=%.4f spec=%.4f"
                    ),
                    epoch,
                    int(k),
                    wsi_acc_threshold,
                    val_wsi_count,
                    float(k_metrics["acc"]),
                    float(k_metrics["balanced_acc"]),
                    float(k_metrics["f1"]),
                    float(k_metrics["recall"]),
                    float(k_metrics["specificity"]),
                )

            for sid in slide_ids:
                probs_sid = wsi_probs_by_slide[sid]
                slide_score = float(topk_mean_score(probs_sid, wsi_topk))
                slide_label = int(wsi_labels[sid])
                pred_label = 1 if slide_score > wsi_acc_threshold else 0
                patch_count = int(wsi_patch_count.get(sid, 0))
                mean_prob = float(sum(probs_sid) / max(len(probs_sid), 1))
                max_prob = float(wsi_max_prob.get(sid, 0.0))
                detail = {
                    "epoch": epoch,
                    "slide_id": sid,
                    "slide_label": slide_label,
                    "slide_score_topk": slide_score,
                    "slide_score_max": max_prob,
                    "slide_score_mean": mean_prob,
                    "pred_label": pred_label,
                    "correct": int(pred_label == slide_label),
                    "num_patches_seen": patch_count,
                    "max_score_sample": wsi_max_sample.get(sid),
                    "wsi_score_method": "topk_mean",
                    "wsi_topk": wsi_topk,
                    "slide_prob_max_raw": max_prob,
                    "wsi_acc_threshold": wsi_acc_threshold,
                }
                val_wsi_details.append(detail)
                wsi_logger.info(json.dumps(detail, ensure_ascii=True))
                LOGGER.info(
                    (
                        "WSI epoch=%d slide=%s label=%d score_topk=%.6f score_max=%.6f "
                        "score_mean=%.6f pred=%d correct=%d patches=%d max_sample=%s"
                    ),
                    epoch,
                    sid,
                    slide_label,
                    slide_score,
                    max_prob,
                    mean_prob,
                    pred_label,
                    int(pred_label == slide_label),
                    patch_count,
                    wsi_max_sample.get(sid),
                )
            LOGGER.info(
                "WSI aggregation details: method=topk_mean k=%d threshold=%.3f slides=%d",
                wsi_topk,
                wsi_acc_threshold,
                val_wsi_count,
            )
        if has_wsi_slide_label_map and wsi_missing_slide_label_count > 0:
            LOGGER.warning(
                "Missing slide-level label for %d samples during WSI aggregation; used patch labels as fallback.",
                wsi_missing_slide_label_count,
            )
        if wsi_patch_label_conflict_count > 0:
            LOGGER.warning(
                "Found %d fallback patch-label conflicts during WSI aggregation; conflicting samples were skipped.",
                wsi_patch_label_conflict_count,
            )
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
        if val_wsi_acc is not None:
            acc_log += f" | val_wsi_acc={val_wsi_acc:.4f}(k={wsi_topk})"
        if val_wsi_auc is not None:
            acc_log += f" | val_wsi_auc={val_wsi_auc:.4f}"
        if val_wsi_geom_stability is not None:
            acc_log += f" | val_wsi_geom_stability={val_wsi_geom_stability:.4f}"
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
                    "val_wsi_acc": val_wsi_acc,
                    "val_wsi_auc": val_wsi_auc,
                    "val_wsi_count": val_wsi_count if enable_cls else None,
                    "val_wsi_geom_stability": val_wsi_geom_stability if enable_cls else None,
                    "val_wsi_geom_field": geom_val_field if enable_cls else None,
                    "val_wsi_threshold": wsi_acc_threshold if enable_cls else None,
                    "val_wsi_score_method": "topk_mean" if enable_cls else None,
                    "val_wsi_topk": wsi_topk if enable_cls else None,
                    **val_wsi_geom_details,
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
                "wsi_acc": val_wsi_acc,
                "wsi_auc": val_wsi_auc,
                "wsi_geom_stability": val_wsi_geom_stability,
            }
        )

        # Best checkpoint 规则：
        # 1) 优先采用 z-score 融合（val_dice 与 val_wsi_auc）。
        # 2) AUC 守门：当前 val_wsi_auc 不能比 best_wsi_auc 低超过设定阈值。
        # 3) 几何稳定性守门：当前稳定性分数不能比 best 低超过设定阈值。
        # 4) score 相近时，先比较几何稳定性，再比较 val_loss。
        score_mode = "fallback_mean"
        z_dice = None
        z_wsi_auc = None
        score = float("-inf")
        if enable_seg and enable_cls and val_wsi_auc is not None:
            dice_hist = [float(x["dice"]) for x in history["val"] if x.get("dice") is not None]
            auc_hist = [float(x["wsi_auc"]) for x in history["val"] if x.get("wsi_auc") is not None]
            dice_mean, dice_std = running_mean_std(dice_hist)
            auc_mean, auc_std = running_mean_std(auc_hist)
            z_dice = safe_zscore(float(val_dice), dice_mean, dice_std, std_eps)
            z_wsi_auc = safe_zscore(float(val_wsi_auc), auc_mean, auc_std, std_eps)
            score = z_dice_weight * z_dice + z_wsi_auc_weight * z_wsi_auc
            score_mode = "zscore_dice_wsi_auc"
        else:
            metrics_for_score = []
            if enable_seg:
                metrics_for_score.append(val_dice)
            if enable_cls:
                metrics_for_score.append(val_acc)
            if metrics_for_score:
                score = sum(metrics_for_score) / len(metrics_for_score)

        auc_gate_pass = True
        if enable_cls and val_wsi_auc is not None and best_wsi_auc is not None:
            auc_gate_pass = float(val_wsi_auc) >= float(best_wsi_auc) - auc_gate_delta
        if cfg.log.save_ckpt and not auc_gate_pass and best_wsi_auc is not None and val_wsi_auc is not None:
            LOGGER.info(
                (
                    "Skip best checkpoint update at epoch %d due to AUC gate: "
                    "val_wsi_auc=%.4f < best_wsi_auc(%.4f) - delta(%.4f)"
                ),
                epoch,
                float(val_wsi_auc),
                float(best_wsi_auc),
                float(auc_gate_delta),
            )

        geom_gate_pass = True
        if (
            geom_val_enabled
            and val_wsi_geom_stability is not None
            and best_wsi_geom_stability is not None
        ):
            geom_gate_pass = float(val_wsi_geom_stability) >= float(best_wsi_geom_stability) - float(
                geom_gate_delta
            )
        if (
            cfg.log.save_ckpt
            and not geom_gate_pass
            and best_wsi_geom_stability is not None
            and val_wsi_geom_stability is not None
        ):
            LOGGER.info(
                (
                    "Skip best checkpoint update at epoch %d due to geometry gate: "
                    "val_wsi_geom_stability=%.4f < best_wsi_geom_stability(%.4f) - delta(%.4f)"
                ),
                epoch,
                float(val_wsi_geom_stability),
                float(best_wsi_geom_stability),
                float(geom_gate_delta),
            )

        score_improved = bool(score > best_score + score_tie_eps)
        score_close = bool(abs(score - best_score) <= score_tie_eps)
        loss_improved = val_loss < best_val_loss - loss_tie_eps
        geom_tie_ready = bool(
            val_wsi_geom_stability is not None and best_wsi_geom_stability is not None
        )
        geom_improved = bool(
            geom_tie_ready
            and float(val_wsi_geom_stability) > float(best_wsi_geom_stability) + float(geom_tie_eps)
        )
        geom_close = bool(
            geom_tie_ready
            and abs(float(val_wsi_geom_stability) - float(best_wsi_geom_stability))
            <= float(geom_tie_eps)
        )
        tie_break_pass = bool(score_improved)
        if score_close and not score_improved:
            if geom_tie_ready:
                tie_break_pass = geom_improved or (geom_close and loss_improved)
            else:
                tie_break_pass = loss_improved
        should_save_ckpt = (
            cfg.log.save_ckpt
            and auc_gate_pass
            and geom_gate_pass
            and tie_break_pass
        )
        if should_save_ckpt:
            best_score = score
            best_val_loss = val_loss
            if val_wsi_auc is not None:
                best_wsi_auc = float(val_wsi_auc)
            if val_wsi_geom_stability is not None:
                best_wsi_geom_stability = float(val_wsi_geom_stability)
            ckpt_path = os.path.join(run_dir, "best.pt")
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "gradnorm_state_dict": gradnorm.state_dict() if gradnorm is not None else None,
                "epoch": epoch,
                "best_score": score,
                "best_score_mode": score_mode,
                "best_val_loss": best_val_loss,
                "best_wsi_auc": best_wsi_auc,
                "best_wsi_geom_stability": best_wsi_geom_stability,
                "best_selector": {
                    "z_dice_weight": z_dice_weight,
                    "z_wsi_auc_weight": z_wsi_auc_weight,
                    "auc_gate_delta": auc_gate_delta,
                    "geom_gate_delta": geom_gate_delta,
                    "geom_tie_eps": geom_tie_eps,
                    "score_tie_eps": score_tie_eps,
                    "std_eps": std_eps,
                    "loss_tie_eps": loss_tie_eps,
                },
                "score_components": {
                    "z_dice": z_dice,
                    "z_wsi_auc": z_wsi_auc,
                    "val_dice": val_dice if enable_seg else None,
                    "val_wsi_auc": val_wsi_auc,
                    "val_wsi_geom_stability": val_wsi_geom_stability,
                },
                "loss_weighting": weighting,
                "metrics": {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice if enable_seg else None,
                    "val_acc": val_acc if enable_cls else None,
                    "val_wsi_auc": val_wsi_auc,
                    "val_wsi_geom_stability": val_wsi_geom_stability,
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
                f"score={score:.6f}",
                f"score_mode={score_mode}",
                f"z_dice_weight={z_dice_weight:.4f}",
                f"z_wsi_auc_weight={z_wsi_auc_weight:.4f}",
                f"auc_gate_delta={auc_gate_delta:.6f}",
                f"geom_gate_delta={geom_gate_delta:.6f}",
                f"geom_tie_eps={geom_tie_eps:.6g}",
                f"score_tie_eps={score_tie_eps:.6g}",
                f"std_eps={std_eps:.6g}",
                f"loss_tie_eps={loss_tie_eps:.6g}",
                f"z_dice={(f'{z_dice:.6f}' if z_dice is not None else 'NA')}",
                f"z_wsi_auc={(f'{z_wsi_auc:.6f}' if z_wsi_auc is not None else 'NA')}",
                f"val_dice={(f'{val_dice:.4f}' if enable_seg else 'NA')}",
                f"val_acc={(f'{val_acc:.4f}' if enable_cls else 'NA')}",
                f"val_wsi_auc={(f'{val_wsi_auc:.4f}' if val_wsi_auc is not None else 'NA')}",
                (
                    "val_wsi_geom_stability="
                    f"{(f'{val_wsi_geom_stability:.4f}' if val_wsi_geom_stability is not None else 'NA')}"
                ),
                f"train_loss={train_loss:.4f}",
                f"val_loss={val_loss:.4f}",
                "payload_keys=[model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_score, metrics, history, config, random_seed]",
                f"ckpt_path={ckpt_path}",
            ]
            summary_path = Path(run_dir) / "best_checkpoint.txt"
            summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
            LOGGER.info("Saved best checkpoint to %s", ckpt_path)
            LOGGER.info("Updated best checkpoint summary at %s", summary_path)

    best_wsi_auc_log = f"{best_wsi_auc:.4f}" if best_wsi_auc is not None else "NA"
    best_wsi_geom_log = (
        f"{best_wsi_geom_stability:.4f}" if best_wsi_geom_stability is not None else "NA"
    )
    LOGGER.info(
        (
            "Training finished. Best score=%.6f best_val_loss=%.4f "
            "best_wsi_auc=%s best_wsi_geom_stability=%s"
        ),
        best_score,
        best_val_loss,
        best_wsi_auc_log,
        best_wsi_geom_log,
    )


if __name__ == "__main__":
    main()
