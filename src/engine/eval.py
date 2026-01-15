"""Test-set evaluation for multitask segmentation + classification."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from src.datasets import DummyPathologyDataset, PathologyDataset
from src.losses import GradNorm, MultiTaskLoss
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


def compute_binary_roc(
    probs: torch.Tensor, labels: torch.Tensor
) -> tuple[list, list, list, float]:
    probs_np = probs.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


def save_roc_curve(fpr: list, tpr: list, roc_auc: float, out_path: str) -> None:
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


def build_test_dataset(cfg: DictConfig) -> torch.utils.data.Dataset:
    normalize_mode = cfg.data.get("normalize", "none")
    normalize_mean = None
    normalize_std = None
    if normalize_mode == "imagenet":
        normalize_mean = cfg.data.get("imagenet_mean", [0.485, 0.456, 0.406])
        normalize_std = cfg.data.get("imagenet_std", [0.229, 0.224, 0.225])

    if cfg.data.get("use_dummy", True):
        test_length = cfg.data.dummy.get(
            "test_samples",
            cfg.data.dummy.get("val_samples", 4),
        )
        return DummyPathologyDataset(
            length=test_length,
            image_size=cfg.data.dummy.image_size,
            num_classes=cfg.model.num_classes,
            seed=cfg.seed + 2,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    test_images = resolve_path(cfg.data.test_images)
    test_masks = resolve_path(cfg.data.test_masks)
    test_labels = resolve_path(cfg.data.test_labels)
    return PathologyDataset(
        test_images,
        test_masks,
        test_labels,
        transform=None,
        debug_log=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    enable_seg: bool,
    enable_cls: bool,
    weighting: str,
    gradnorm: GradNorm | None,
    save_roc: bool,
    roc_run_dir: str,
    file_stamp: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    seg_loss_sum = 0.0
    cls_loss_sum = 0.0
    dice_sum = 0.0
    acc_sum = 0.0
    batches = 0
    roc_probs: list[torch.Tensor] = []
    roc_labels: list[torch.Tensor] = []
    roc_supported = True

    with torch.no_grad():
        for images, masks, labels, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            seg_logits, cls_logits = model(images)
            if weighting == "gradnorm" and gradnorm is not None:
                # GradNorm eval: reuse stored weights to compute weighted total loss.
                task_losses, _, parts = criterion.compute_task_losses(
                    seg_logits=seg_logits,
                    seg_targets=masks,
                    cls_logits=cls_logits,
                    cls_targets=labels,
                )
                loss = gradnorm.weight_losses(task_losses)
            else:
                loss, parts = criterion(seg_logits, masks, cls_logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            if enable_seg:
                seg_loss_sum += parts["seg_loss"] * batch_size
                dice = dice_coefficient(torch.sigmoid(seg_logits), masks).mean().item()
                dice_sum += dice
            if enable_cls:
                cls_loss_sum += parts["cls_loss"] * batch_size
                acc = accuracy_from_logits(cls_logits, labels)
                acc_sum += acc
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

    dataset_len = max(len(dataloader.dataset), 1)
    total_loss /= dataset_len
    seg_loss = seg_loss_sum / dataset_len if enable_seg else None
    cls_loss = cls_loss_sum / dataset_len if enable_cls else None
    dice = dice_sum / max(batches, 1) if enable_seg else None
    acc = acc_sum / max(batches, 1) if enable_cls else None

    roc_payload = {}
    if enable_cls and save_roc and roc_supported and roc_probs:
        all_probs = torch.cat(roc_probs, dim=0)
        all_labels = torch.cat(roc_labels, dim=0)
        fpr, tpr, thresholds, roc_auc = compute_binary_roc(all_probs, all_labels)
        roc_plot_path = os.path.join(roc_run_dir, f"roc_{file_stamp}_test.png")
        save_roc_curve(fpr, tpr, roc_auc, roc_plot_path)
        roc_payload = {
            "roc_auc": roc_auc,
            "roc_fpr": [round(x, 6) for x in fpr],
            "roc_tpr": [round(x, 6) for x in tpr],
            "roc_thresholds": [round(x, 6) for x in thresholds],
            "roc_plot": roc_plot_path,
        }
    elif enable_cls and save_roc and not roc_supported:
        roc_payload = {"roc_note": "ROC 仅支持二分类输出（1 个 logit 或 2 类 softmax）。"}

    payload = {
        "split": "test",
        "total_loss": total_loss,
        "seg_loss": seg_loss,
        "cls_loss": cls_loss,
        "dice": dice,
        "acc": acc,
        **roc_payload,
    }
    return payload


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    enable_seg = cfg.tasks.get("enable_seg", True)
    enable_cls = cfg.tasks.get("enable_cls", True)
    if not enable_seg and not enable_cls:
        raise ValueError("At least one task must be enabled via cfg.tasks.")

    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(cfg.eval.get("save_dir", os.getcwd()))
    roc_run_dir = resolve_path(cfg.log.get("roc_dir", run_dir))
    ensure_dir(run_dir)
    ensure_dir(roc_run_dir)
    params_log_path = os.path.join(run_dir, f"test_metrics_{file_stamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    LOGGER.info("Starting test evaluation with config:\n%s", OmegaConf.to_yaml(cfg))
    weighting = resolve_weighting(cfg.loss)
    if weighting == "gradnorm" and (enable_seg + enable_cls) < 2:
        LOGGER.warning("GradNorm requires at least two tasks; fallback to fixed weighting.")
        weighting = "fixed"
    LOGGER.info("Loss weighting strategy: %s", weighting)

    test_ds = build_test_dataset(cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
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

    ckpt_path = cfg.eval.get("ckpt")
    if ckpt_path is None:
        ckpt_path = find_latest_run_ckpt(PROJECT_ROOT / "run")
    if ckpt_path is None:
        raise FileNotFoundError("找不到可用的 best.pt，请显式传入 cfg.eval.ckpt。")
    ckpt_path = resolve_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    gradnorm = None
    if weighting == "gradnorm":
        gradnorm = GradNorm(
            num_tasks=int(enable_seg) + int(enable_cls),
            alpha=float(cfg.loss.get("gradnorm_alpha", 1.5)),
        ).to(device)
        if isinstance(state, dict) and state.get("gradnorm_state_dict"):
            gradnorm.load_state_dict(state["gradnorm_state_dict"])
        else:
            LOGGER.warning("GradNorm weighting requested but no state found; using default weights.")

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

    payload = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        enable_seg=enable_seg,
        enable_cls=enable_cls,
        weighting=weighting,
        gradnorm=gradnorm,
        save_roc=bool(cfg.eval.get("save_roc", True)),
        roc_run_dir=roc_run_dir,
        file_stamp=file_stamp,
    )

    with open(params_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    LOGGER.info(
        "Test metrics | loss=%.4f seg=%s cls=%s dice=%s acc=%s",
        payload["total_loss"],
        f"{payload['seg_loss']:.4f}" if payload.get("seg_loss") is not None else "NA",
        f"{payload['cls_loss']:.4f}" if payload.get("cls_loss") is not None else "NA",
        f"{payload['dice']:.4f}" if payload.get("dice") is not None else "NA",
        f"{payload['acc']:.4f}" if payload.get("acc") is not None else "NA",
    )
    LOGGER.info("Test metrics saved to %s", params_log_path)


if __name__ == "__main__":
    main()
