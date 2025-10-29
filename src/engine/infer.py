"""简易推理脚本：支持对单张 patch 做分割+分类并输出结果图。"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.models.multitask_model import MultiTaskModel
from src.utils.visualizer import save_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image", type=str, default=None, help="单张patch图像路径（PNG/JPG）"
    )
    ap.add_argument(
        "--ckpt", type=str, default=None, help="模型权重（best.pt），可为空使用随机权重"
    )
    ap.add_argument("--mask_out", type=str, default="pred_mask.png")
    ap.add_argument("--overlay_out", type=str, default="overlay.png")
    ap.add_argument("--dry_run", type=int, default=0, help="仅自检导入路径与依赖")
    args = ap.parse_args()

    if args.dry_run:
        print("[DRY RUN] import/CLI ok")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel().to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    # 读取图像
    img = Image.open(args.image).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits, cls_logits = model(tensor)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        cls_prob = torch.sigmoid(cls_logits)[0, 0].item()

    # 保存二值掩膜与叠加
    plt.imsave(args.mask_out, seg_prob, cmap="gray")
    save_overlay((arr * 255).astype(np.uint8), seg_prob, args.overlay_out, alpha=0.5)

    print(
        f"Done. cls_prob={cls_prob:.3f} | mask_out={args.mask_out} | overlay={args.overlay_out}"
    )


if __name__ == "__main__":
    main()
