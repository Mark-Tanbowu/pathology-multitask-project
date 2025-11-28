"""Standalone validation script using saved checkpoint."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets import PathologyDataset
from src.models.multitask_model import MultiTaskModel
from src.utils.metrics import accuracy_from_logits, dice_coefficient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PathologyDataset(args.images, args.masks, args.labels, transform=None, debug_log=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = MultiTaskModel().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    dice_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks, labels in loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            seg_logits, cls_logits = model(images)
            dice_sum += dice_coefficient(torch.sigmoid(seg_logits), masks).mean().item()
            acc_sum += accuracy_from_logits(cls_logits, labels)
            n += 1
    print(f"Val Dice={dice_sum/max(n,1):.4f} | Val Acc={acc_sum/max(n,1):.4f}")


if __name__ == "__main__":
    main()
