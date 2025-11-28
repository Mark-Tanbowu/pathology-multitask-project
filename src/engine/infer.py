"""多任务模型推理脚本。

用途：
    - 便捷加载训练好的权重，对单张图片同时输出分割与分类结果；
    - 若未提供图片或权重，可用随机张量/随机初始化做冒烟验证，确保 CLI 链路正确；
    - 支持保存灰度 mask 与原图叠加效果，方便快速查看模型输出质量。"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.models.multitask_model import MultiTaskModel
from src.utils.visualizer import save_overlay


def load_image(path: str, image_size: Optional[int] = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--mask_out", type=str, default="pred_mask.png")
    parser.add_argument("--overlay_out", type=str, default="overlay.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Inference CLI is wired correctly.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel().to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Loaded checkpoint from {args.ckpt}")
    model.eval()

    if args.image is None:
        image = torch.rand(1, 3, args.image_size, args.image_size)
        print("No image provided; using random tensor for smoke test.")
    else:
        image = load_image(args.image, image_size=args.image_size)
    image = image.to(device)

    with torch.no_grad():
        seg_logits, cls_logits = model(image)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        cls_prob = torch.sigmoid(cls_logits)[0].flatten().cpu().numpy()

    plt.imsave(args.mask_out, seg_prob, cmap="gray")
    save_overlay((image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), seg_prob, args.overlay_out)
    print(f"Classification logits (sigmoid): {cls_prob}")
    print(f"Saved mask to {args.mask_out} and overlay to {args.overlay_out}")


if __name__ == "__main__":
    main()
