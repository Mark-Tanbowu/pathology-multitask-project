"""基于 slide manifest 的 test 检测脚本（与 train/val 数据格式对齐）。

功能：
1. 读取 `manifest_slides_test.csv + coords/*.npy`；
2. 加载最佳权重（或显式指定 ckpt）；
3. 输出 patch 级与 WSI 级检测日志，便于复现实验结果。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from src.datasets import DummyPathologyDataset, SlideCoordsDataset
from src.models.multitask_model import MultiTaskModel
from src.utils.metrics import accuracy_from_logits, dice_coefficient
from src.utils.misc import ensure_dir, get_device, set_seed

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def find_latest_run_ckpt(run_root: Path) -> str | None:
    """从 run 目录中寻找最新的 best.pt。"""
    if not run_root.exists():
        return None
    run_dirs = [d for d in run_root.iterdir() if d.is_dir()]
    for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
        ckpt_path = run_dir / "best.pt"
        if ckpt_path.exists():
            return str(ckpt_path)
    return None


def infer_slide_id(sample_name: str) -> str:
    """从 `slide_x_y` 命名中提取 slide_id。"""
    stem = Path(sample_name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
        return parts[0]
    return stem


def topk_mean_score(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    k_use = min(max(int(k), 1), len(values))
    topk = sorted(values, reverse=True)[:k_use]
    return float(sum(topk) / float(k_use))


def build_slide_label_map(dataset: torch.utils.data.Dataset) -> tuple[dict[str, int], int]:
    """从 slide manifest 中读取 slide_id -> slide_label 映射。"""
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


def compute_binary_roc(
    probs: torch.Tensor, labels: torch.Tensor
) -> tuple[list[float], list[float], list[float], float] | None:
    """计算 ROC；若标签仅单类则返回 None。"""
    probs_np = probs.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    if np.unique(labels_np).size < 2:
        return None
    fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


def binary_metrics_from_probs(
    labels: torch.Tensor,
    probs: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """按固定阈值计算二分类指标。"""
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


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_manifest_entry_path(path_str: str, manifest_path: Path) -> Path:
    """解析 manifest 中的路径字段，优先项目根目录，再回退到 manifest 所在目录。"""
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    candidate_root = (PROJECT_ROOT / raw).resolve()
    if candidate_root.exists():
        return candidate_root
    return (manifest_path.parent / raw).resolve()


def validate_test_manifest_paths(cfg: DictConfig, test_ds: torch.utils.data.Dataset) -> None:
    """启动前校验 test manifest 与 test 检测脚本使用路径是否一致。"""
    if bool(cfg.data.get("use_dummy", False)):
        LOGGER.info("[TEST-DETECT] use_dummy=true，跳过 test manifest 路径校验。")
        return

    test_manifest_cfg = cfg.data.get("test_slide_manifest")
    if not test_manifest_cfg:
        raise ValueError("配置缺失：data.test_slide_manifest 不能为空。")
    prepare_test_cfg = cfg.get("prepare_test")
    if prepare_test_cfg is None:
        raise ValueError("配置缺失：prepare_test 不能为空。")

    manifest_runtime = Path(resolve_path(str(test_manifest_cfg))).resolve()
    manifest_prepare_cfg = prepare_test_cfg.get("slide_manifest_path")
    manifest_prepare = (
        Path(resolve_path(str(manifest_prepare_cfg))).resolve() if manifest_prepare_cfg else None
    )
    coords_root_cfg = prepare_test_cfg.get("coords_out_dir")
    coords_root = Path(resolve_path(str(coords_root_cfg))).resolve() if coords_root_cfg else None
    masks_dir_cfg = prepare_test_cfg.get("masks_dir")
    if not masks_dir_cfg:
        raise ValueError("配置缺失：prepare_test.masks_dir 不能为空。")
    masks_dir = Path(resolve_path(str(masks_dir_cfg))).resolve()
    mask_suffix = str(prepare_test_cfg.get("mask_suffix", "_mask.tif"))

    if manifest_prepare is not None and manifest_prepare != manifest_runtime:
        LOGGER.warning(
            (
                "[TEST-DETECT] 路径不一致：data.test_slide_manifest=%s, "
                "prepare_test.slide_manifest_path=%s"
            ),
            str(manifest_runtime),
            str(manifest_prepare),
        )
    else:
        LOGGER.info("[TEST-DETECT] test manifest 路径一致：%s", str(manifest_runtime))

    if not manifest_runtime.exists():
        raise FileNotFoundError(f"Test manifest not found: {manifest_runtime}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Test masks_dir not found: {masks_dir}")
    if coords_root is not None and not coords_root.exists():
        raise FileNotFoundError(f"Test coords_out_dir not found: {coords_root}")

    rows = getattr(test_ds, "rows", None)
    if not rows:
        LOGGER.warning("[TEST-DETECT] test dataset rows 为空，跳过行级路径检查。")
        return

    missing_coords: list[str] = []
    missing_masks: list[str] = []
    outside_coords_root: list[str] = []
    for row in rows:
        slide_id = str(row.get("slide_id", "")).strip()
        coords_path_raw = str(row.get("coords_path", "")).strip()
        if slide_id:
            mask_path = masks_dir / f"{slide_id}{mask_suffix}"
            if not mask_path.exists():
                missing_masks.append(str(mask_path))
        if not coords_path_raw:
            missing_coords.append(f"{slide_id}:<empty>")
            continue
        coords_path = resolve_manifest_entry_path(coords_path_raw, manifest_runtime)
        if not coords_path.exists():
            missing_coords.append(str(coords_path))
            continue
        if coords_root is not None and not _path_is_under(coords_path, coords_root):
            outside_coords_root.append(str(coords_path))

    LOGGER.info(
        (
            "[TEST-DETECT] 路径检查完成：slides=%d missing_coords=%d "
            "missing_masks=%d outside_coords_root=%d coords_root=%s masks_dir=%s"
        ),
        len(rows),
        len(missing_coords),
        len(missing_masks),
        len(outside_coords_root),
        str(coords_root) if coords_root is not None else "NA",
        str(masks_dir),
    )
    if outside_coords_root:
        LOGGER.warning(
            "[TEST-DETECT] 发现 coords_path 不在 prepare_test.coords_out_dir 下，示例：%s",
            outside_coords_root[:3],
        )

    if missing_coords or missing_masks:
        raise FileNotFoundError(
            (
                "Test 数据路径检查失败："
                f"missing_coords={len(missing_coords)} sample={missing_coords[:3]}, "
                f"missing_masks={len(missing_masks)} sample={missing_masks[:3]}"
            )
        )


def build_test_dataset(cfg: DictConfig) -> torch.utils.data.Dataset:
    """构建 test 数据集，保持与 train/val 相同的 SlideCoordsDataset 格式。"""
    normalize_mode = cfg.data.get("normalize", "none")
    normalize_mean = None
    normalize_std = None
    if normalize_mode == "imagenet":
        normalize_mean = cfg.data.get("imagenet_mean", [0.485, 0.456, 0.406])
        normalize_std = cfg.data.get("imagenet_std", [0.229, 0.224, 0.225])

    if cfg.data.get("use_dummy", False):
        return DummyPathologyDataset(
            length=cfg.data.dummy.get("test_samples", 4),
            image_size=cfg.data.dummy.image_size,
            num_classes=cfg.model.num_classes,
            seed=cfg.seed + 2,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    test_slide_manifest_cfg = cfg.data.get("test_slide_manifest")
    if not test_slide_manifest_cfg:
        raise ValueError("配置缺失：data.test_slide_manifest 不能为空。")

    prepare_test_cfg = cfg.get("prepare_test")
    if prepare_test_cfg is None:
        raise ValueError("配置缺失：prepare_test 不能为空。")
    test_masks_dir_cfg = prepare_test_cfg.get("masks_dir")
    if not test_masks_dir_cfg:
        raise ValueError("配置缺失：prepare_test.masks_dir 不能为空。")

    test_slide_manifest = resolve_path(str(test_slide_manifest_cfg))
    test_masks_dir = resolve_path(str(test_masks_dir_cfg))
    test_mask_suffix = str(prepare_test_cfg.get("mask_suffix", "_mask.tif"))
    coords_cache_slides = int(cfg.data.get("coords_cache_slides", 2))
    cache_masks = bool(cfg.data.get("cache_masks", False))
    test_mask_level = prepare_test_cfg.get("mask_level", cfg.data.get("mask_level"))
    test_mask_max_size = prepare_test_cfg.get("mask_max_size", cfg.data.get("mask_max_size"))

    return SlideCoordsDataset(
        test_slide_manifest,
        test_masks_dir,
        mask_suffix=test_mask_suffix,
        coords_cache_slides=coords_cache_slides,
        cache_masks=cache_masks,
        mask_level=int(test_mask_level) if test_mask_level is not None else None,
        mask_max_size=int(test_mask_max_size) if test_mask_max_size is not None else None,
        transform=None,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    enable_seg = bool(cfg.tasks.get("enable_seg", True))
    enable_cls = bool(cfg.tasks.get("enable_cls", True))
    if not enable_seg and not enable_cls:
        raise ValueError("At least one task must be enabled via cfg.tasks.")

    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = str(cfg.test_detect.get("log_prefix", "wsi_test_detect")).strip()
    if not log_prefix:
        log_prefix = "wsi_test_detect"
    save_dir = resolve_path(str(cfg.test_detect.get("save_dir", cfg.log.params_dir)))
    ensure_dir(save_dir)
    summary_log_path = os.path.join(save_dir, f"{log_prefix}_metrics_{file_stamp}.log")
    slide_scores_log_path = os.path.join(save_dir, f"{log_prefix}_slide_scores_{file_stamp}.log")
    patch_scores_log_path = os.path.join(save_dir, f"{log_prefix}_patch_scores_{file_stamp}.log")
    runtime_log_path = os.path.join(save_dir, f"{log_prefix}_runtime_{file_stamp}.log")
    save_runtime_log = bool(cfg.test_detect.get("save_runtime_log", True))
    save_slide_scores = bool(cfg.test_detect.get("save_slide_scores", True))
    save_patch_scores = bool(cfg.test_detect.get("save_patch_scores", False))

    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    if save_runtime_log:
        log_handlers.append(logging.FileHandler(runtime_log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=log_handlers,
        force=True,
    )
    LOGGER.info("[TEST-DETECT] Starting with config:\n%s", OmegaConf.to_yaml(cfg))
    if save_runtime_log:
        LOGGER.info("[TEST-DETECT] runtime_log=%s", runtime_log_path)

    test_ds = build_test_dataset(cfg)
    validate_test_manifest_paths(cfg, test_ds)
    test_batch_size = int(cfg.test_detect.get("batch_size", cfg.batch_size))
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    total_patches = int(len(test_ds))
    total_batches = int(len(test_loader))
    progress_log_interval_sec = max(
        0.1, float(cfg.test_detect.get("progress_log_interval_sec", 3.0))
    )
    LOGGER.info(
        (
            "[TEST-DETECT] DataLoader ready: patches=%d batches=%d batch_size=%d "
            "num_workers=%d progress_interval=%.1fs"
        ),
        total_patches,
        total_batches,
        test_batch_size,
        int(cfg.num_workers),
        progress_log_interval_sec,
    )

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

    ckpt_path = cfg.test_detect.get("ckpt")
    if ckpt_path is None:
        ckpt_path = cfg.eval.get("ckpt")
    if ckpt_path is None:
        ckpt_path = find_latest_run_ckpt(PROJECT_ROOT / "run")
    if ckpt_path is None:
        raise FileNotFoundError("找不到可用的 best.pt，请显式传入 test_detect.ckpt。")
    ckpt_path = resolve_path(str(ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    strict_load = bool(cfg.test_detect.get("strict_load", True))
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    try:
        load_result = model.load_state_dict(state_dict, strict=strict_load)
    except RuntimeError as exc:
        LOGGER.exception(
            (
                "[TEST-DETECT] 加载 checkpoint 失败: ckpt=%s strict_load=%s "
                "backbone=%s encoder_attention=%s decoder_attention=%s"
            ),
            ckpt_path,
            strict_load,
            cfg.model.backbone,
            str(cfg.model.get("encoder_attention")),
            str(cfg.model.get("decoder_attention")),
        )
        raise RuntimeError(
            "加载 checkpoint 失败：当前模型配置与权重结构不一致。"
            "请对齐 model.* 注意力/骨干参数，或设置 test_detect.strict_load=false。"
        ) from exc
    if (not strict_load) and (
        getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None)
    ):
        LOGGER.warning(
            "[TEST-DETECT] strict_load=false: missing_keys=%d unexpected_keys=%d",
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )
        LOGGER.warning(
            "[TEST-DETECT] missing_keys(sample)=%s unexpected_keys(sample)=%s",
            list(getattr(load_result, "missing_keys", []))[:8],
            list(getattr(load_result, "unexpected_keys", []))[:8],
        )
    LOGGER.info("[TEST-DETECT] Loaded checkpoint: %s", ckpt_path)

    wsi_topk = int(cfg.test_detect.get("wsi_topk", cfg.eval.get("wsi_topk", 15)))
    wsi_threshold = float(
        cfg.test_detect.get("wsi_acc_threshold", cfg.eval.get("wsi_acc_threshold", 0.5))
    )

    patch_batches = 0
    patch_count = 0
    patch_dice_sum = 0.0
    patch_acc_sum = 0.0
    patch_probs: list[torch.Tensor] = []
    patch_labels: list[torch.Tensor] = []
    roc_supported = True

    wsi_probs_by_slide: dict[str, list[float]] = {}
    wsi_patch_count: dict[str, int] = {}
    wsi_max_prob: dict[str, float] = {}
    wsi_max_sample: dict[str, str] = {}
    wsi_labels: dict[str, int] = {}
    wsi_patch_label_conflict_count = 0

    slide_label_map, slide_manifest_label_conflicts = build_slide_label_map(test_ds)
    if slide_manifest_label_conflicts > 0:
        LOGGER.warning(
            "[TEST-DETECT] slide manifest label conflicts=%d，冲突项已跳过。",
            slide_manifest_label_conflicts,
        )

    patch_log_f = None
    slide_log_f = None
    try:
        if save_patch_scores:
            patch_log_f = open(patch_scores_log_path, "w", encoding="utf-8")
        if save_slide_scores:
            slide_log_f = open(slide_scores_log_path, "w", encoding="utf-8")

        model.eval()
        infer_start_t = time.perf_counter()
        last_progress_t = infer_start_t
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader, start=1):
                sample_names: list[str] = []
                try:
                    images, masks, labels, sample_names = batch_data
                    images = images.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    seg_logits, cls_logits = model(images)

                    batch_size = images.size(0)
                    patch_count += batch_size

                    if enable_seg and seg_logits is not None:
                        batch_dice = (
                            dice_coefficient(torch.sigmoid(seg_logits), masks).mean().item()
                        )
                        patch_dice_sum += batch_dice

                    probs = None
                    if enable_cls and cls_logits is not None:
                        patch_acc_sum += accuracy_from_logits(cls_logits, labels)
                        if cls_logits.ndim == 2 and cls_logits.shape[1] == 1:
                            probs = torch.sigmoid(cls_logits).view(-1)
                        elif cls_logits.ndim == 2 and cls_logits.shape[1] == 2:
                            probs = torch.softmax(cls_logits, dim=1)[:, 1]
                        else:
                            roc_supported = False
                            probs = None
                        if probs is not None:
                            patch_probs.append(probs.detach().cpu())
                            patch_labels.append(labels.detach().cpu())

                    if probs is not None:
                        probs_cpu = probs.detach().cpu().view(-1)
                        labels_cpu = labels.detach().cpu().view(-1)
                        for i, sample_name in enumerate(sample_names):
                            sid = infer_slide_id(str(sample_name))
                            prob_val = float(probs_cpu[i].item())
                            patch_label = int(labels_cpu[i].item() > 0.5)
                            slide_label = slide_label_map.get(sid, patch_label)
                            prev_label = wsi_labels.get(sid)
                            if prev_label is not None and prev_label != slide_label:
                                wsi_patch_label_conflict_count += 1
                                continue
                            wsi_labels[sid] = int(slide_label)
                            wsi_probs_by_slide.setdefault(sid, []).append(prob_val)
                            wsi_patch_count[sid] = wsi_patch_count.get(sid, 0) + 1
                            prev_prob = wsi_max_prob.get(sid)
                            if prev_prob is None or prob_val > prev_prob:
                                wsi_max_prob[sid] = prob_val
                                wsi_max_sample[sid] = str(sample_name)
                            if patch_log_f is not None:
                                patch_log_f.write(
                                    json.dumps(
                                        {
                                            "sample_name": str(sample_name),
                                            "slide_id": sid,
                                            "label": patch_label,
                                            "prob": prob_val,
                                        },
                                        ensure_ascii=True,
                                    )
                                    + "\n"
                                )

                    patch_batches += 1
                except Exception:
                    preview = [str(x) for x in list(sample_names)[:3]]
                    LOGGER.exception(
                        (
                            "[TEST-DETECT] batch 失败: batch=%d/%d patch_count=%d "
                            "sample_preview=%s"
                        ),
                        batch_idx,
                        total_batches,
                        patch_count,
                        preview,
                    )
                    raise

                now_t = time.perf_counter()
                if now_t - last_progress_t >= progress_log_interval_sec:
                    elapsed = now_t - infer_start_t
                    speed = patch_count / max(elapsed, 1.0e-6)
                    remaining = max(total_patches - patch_count, 0)
                    eta_sec = (remaining / speed) if speed > 0 else float("inf")
                    eta_log = f"{eta_sec:.1f}s" if np.isfinite(eta_sec) else "NA"
                    progress = (
                        (100.0 * float(patch_count) / float(total_patches))
                        if total_patches > 0
                        else 0.0
                    )
                    LOGGER.info(
                        (
                            "[TEST-DETECT] progress=%.2f%% batch=%d/%d patches=%d/%d "
                            "slides=%d speed=%.1f patch/s elapsed=%.1fs eta=%s"
                        ),
                        progress,
                        batch_idx,
                        total_batches,
                        patch_count,
                        total_patches,
                        len(wsi_probs_by_slide),
                        speed,
                        elapsed,
                        eta_log,
                    )
                    last_progress_t = now_t

        patch_metrics: dict[str, float | None] = {
            "patch_dice": (patch_dice_sum / max(patch_batches, 1)) if enable_seg else None,
            "patch_acc": (patch_acc_sum / max(patch_batches, 1)) if enable_cls else None,
            "patch_auc": None,
        }
        if enable_cls and patch_probs and roc_supported:
            all_probs = torch.cat(patch_probs, dim=0)
            all_labels = torch.cat(patch_labels, dim=0)
            patch_roc = compute_binary_roc(all_probs, all_labels)
            if patch_roc is not None:
                _, _, _, patch_auc = patch_roc
                patch_metrics["patch_auc"] = float(patch_auc)

        wsi_payload: dict[str, float | int | None] = {
            "wsi_count": 0,
            "wsi_acc": None,
            "wsi_balanced_acc": None,
            "wsi_f1": None,
            "wsi_auc": None,
            "wsi_topk": int(wsi_topk),
            "wsi_threshold": float(wsi_threshold),
        }
        if enable_cls and wsi_probs_by_slide:
            slide_ids = sorted(wsi_probs_by_slide.keys())
            slide_scores = torch.tensor(
                [topk_mean_score(wsi_probs_by_slide[sid], wsi_topk) for sid in slide_ids],
                dtype=torch.float32,
            )
            slide_labels = torch.tensor([wsi_labels[sid] for sid in slide_ids], dtype=torch.float32)
            wsi_metrics = binary_metrics_from_probs(slide_labels, slide_scores, wsi_threshold)
            wsi_payload.update(
                {
                    "wsi_count": int(len(slide_ids)),
                    "wsi_acc": float(wsi_metrics["acc"]),
                    "wsi_balanced_acc": float(wsi_metrics["balanced_acc"]),
                    "wsi_f1": float(wsi_metrics["f1"]),
                    "wsi_tp": float(wsi_metrics["tp"]),
                    "wsi_tn": float(wsi_metrics["tn"]),
                    "wsi_fp": float(wsi_metrics["fp"]),
                    "wsi_fn": float(wsi_metrics["fn"]),
                }
            )
            wsi_roc = compute_binary_roc(slide_scores, slide_labels)
            if wsi_roc is not None:
                _, _, _, wsi_auc = wsi_roc
                wsi_payload["wsi_auc"] = float(wsi_auc)

            if slide_log_f is not None:
                for sid in slide_ids:
                    score = float(topk_mean_score(wsi_probs_by_slide[sid], wsi_topk))
                    detail = {
                        "slide_id": sid,
                        "slide_label": int(wsi_labels[sid]),
                        "slide_score_topk": score,
                        "slide_score_max": float(wsi_max_prob.get(sid, 0.0)),
                        "pred_label": int(1 if score > wsi_threshold else 0),
                        "patch_count": int(wsi_patch_count.get(sid, 0)),
                        "max_score_sample": wsi_max_sample.get(sid),
                        "wsi_score_method": "topk_mean",
                        "wsi_topk": int(wsi_topk),
                        "wsi_acc_threshold": float(wsi_threshold),
                    }
                    slide_log_f.write(json.dumps(detail, ensure_ascii=True) + "\n")

        payload = {
            "split": "test_slide_detect",
            "ckpt": ckpt_path,
            "patch_count": int(patch_count),
            "patch_batches": int(patch_batches),
            "wsi_patch_label_conflicts": int(wsi_patch_label_conflict_count),
            **patch_metrics,
            **wsi_payload,
        }
        with open(summary_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

        LOGGER.info(
            "[TEST-DETECT] Done | patch_count=%d patch_acc=%s "
            "patch_dice=%s wsi_count=%d wsi_acc=%s wsi_auc=%s",
            payload["patch_count"],
            f"{payload['patch_acc']:.4f}" if payload.get("patch_acc") is not None else "NA",
            f"{payload['patch_dice']:.4f}" if payload.get("patch_dice") is not None else "NA",
            int(payload.get("wsi_count", 0)),
            f"{payload['wsi_acc']:.4f}" if payload.get("wsi_acc") is not None else "NA",
            f"{payload['wsi_auc']:.4f}" if payload.get("wsi_auc") is not None else "NA",
        )
        LOGGER.info("[TEST-DETECT] summary=%s", summary_log_path)
        if save_slide_scores:
            LOGGER.info("[TEST-DETECT] slide_scores=%s", slide_scores_log_path)
        if save_patch_scores:
            LOGGER.info("[TEST-DETECT] patch_scores=%s", patch_scores_log_path)
    except Exception:
        manifest_cfg_log = cfg.data.get("test_slide_manifest")
        masks_dir_cfg_log = cfg.get("prepare_test", {}).get("masks_dir")
        LOGGER.exception(
            (
                "[TEST-DETECT] 推理异常终止: ckpt=%s manifest=%s masks_dir=%s "
                "patch_count=%d patch_batches=%d"
            ),
            ckpt_path,
            str(manifest_cfg_log),
            str(masks_dir_cfg_log),
            patch_count,
            patch_batches,
        )
        raise
    finally:
        if patch_log_f is not None:
            patch_log_f.close()
        if slide_log_f is not None:
            slide_log_f.close()


if __name__ == "__main__":
    main()
