import os

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.datasets.camelyon_dataset import PathologyDataset
from src.datasets.transforms import Compose, HorizontalFlip
from src.losses.combined import MultiTaskLoss
from src.models.multitask_model import MultiTaskModel
from src.utils.metrics import accuracy_from_logits, dice_coefficient
from src.utils.misc import get_device, set_seed


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.log.save_dir, exist_ok=True)

    # Transforms（示例，仅水平翻转）
    transform = Compose([HorizontalFlip(p=0.5)])

    # Datasets / Loaders
    train_ds = PathologyDataset(
        cfg.data.train_images,
        cfg.data.train_masks,
        cfg.data.train_labels,
        transform=transform,
    )
    val_ds = PathologyDataset(
        cfg.data.val_images, cfg.data.val_masks, cfg.data.val_labels, transform=None
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # Model & Loss & Optim
    model = MultiTaskModel(
        backbone_name=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        seg_upsample_to_input=cfg.model.seg_upsample_to_input,
    ).to(device)
    criterion = MultiTaskLoss(
        seg_type=cfg.loss.seg,
        cls_type=cfg.loss.cls,
        seg_weight=cfg.loss.seg_weight,
        cls_weight=cfg.loss.cls_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = -1.0
    for epoch in range(1, cfg.num_epochs + 1):
        # ---------------- Train ----------------
        model.train()
        run_loss = 0.0
        for images, masks, labels in train_loader:
            images, masks, labels = (
                images.to(device),
                masks.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            seg_logits, cls_logits = model(images)
            loss, parts = criterion(seg_logits, masks, cls_logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)

        train_loss = run_loss / len(train_ds)

        # ---------------- Validate ----------------
        model.eval()
        val_loss = 0.0
        dice_sum, acc_sum, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images, masks, labels = (
                    images.to(device),
                    masks.to(device),
                    labels.to(device),
                )
                seg_logits, cls_logits = model(images)
                loss, parts = criterion(seg_logits, masks, cls_logits, labels)
                val_loss += loss.item() * images.size(0)
                dice_sum += dice_coefficient(torch.sigmoid(seg_logits), masks)
                acc_sum += accuracy_from_logits(cls_logits, labels)
                n_batches += 1
        val_loss /= len(val_ds)
        val_dice = dice_sum / max(n_batches, 1)
        val_acc = acc_sum / max(n_batches, 1)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        # Save best
        score = (val_dice + val_acc) / 2.0
        if score > best_val and cfg.log.save_ckpt:
            best_val = score
            ckpt_path = os.path.join(cfg.log.save_dir, "best.pt")
            torch.save(
                {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg)},
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
