from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(image_rgb: np.ndarray, mask_prob: np.ndarray, alpha: float=0.5):
    """将概率掩膜以 heatmap 形式叠加到 RGB 图像上（image_rgb: HxWx3, mask_prob: HxW）。"""
    image = image_rgb.astype(float) / 255.0
    heat = plt.cm.jet(mask_prob)[:, :, :3]  # 0~1
    overlay = (1 - alpha) * image + alpha * heat
    overlay = (overlay * 255.0).clip(0,255).astype(np.uint8)
    return overlay

def save_overlay(image_rgb: np.ndarray, mask_prob: np.ndarray, out_path: str, alpha: float=0.5):
    ov = overlay_mask(image_rgb, mask_prob, alpha=alpha)
    plt.imsave(out_path, ov)
    return out_path
