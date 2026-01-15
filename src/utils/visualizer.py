"""
visualizer.py - 可视化与日志工具
负责保存 loss 曲线、mask 叠加图、训练日志等。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


class LossVisualizer:
    """
    用于训练过程损失与指标可视化。
    每次调用 update() 自动刷新曲线并保存。
    """

    def __init__(self, save_dir: str = "."):
        self.save_dir = save_dir
        self.history = {"epoch": [], "train_total": [], "val_total": [], "dice": [], "acc": []}
        os.makedirs(save_dir, exist_ok=True)

    def update(self, epoch, train_losses: dict, val_losses: dict):
        self.history["epoch"].append(epoch)
        self.history["train_total"].append(train_losses.get("total", 0))
        self.history["val_total"].append(val_losses.get("total", 0))
        self.history["dice"].append(val_losses.get("dice", 0))
        self.history["acc"].append(val_losses.get("acc", 0))
        self._plot_curves()

    def _plot_curves(self):
        """绘制并保存 loss / metric 曲线"""
        epochs = self.history["epoch"]
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["train_total"], label="Train Loss", marker="o")
        plt.plot(epochs, self.history["val_total"], label="Val Loss", marker="o")
        plt.plot(epochs, self.history["dice"], label="Dice", linestyle="--")
        plt.plot(epochs, self.history["acc"], label="Accuracy", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curves.png"))
        plt.close()


def save_overlay(image: np.ndarray, mask: np.ndarray, out_path: str, alpha: float = 0.4):
    """
    叠加预测掩膜到原图，生成直观可视化结果。
    - image: 原RGB图像 (H, W, 3)
    - mask: 预测掩膜 (H, W)，0~1
    - out_path: 保存路径
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_rgb = np.zeros_like(image)
    mask_rgb[..., 0] = (mask * 255).astype(np.uint8)  # 红色通道表示前景

    blended = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, mask_rgb, alpha, 0)
    cv2.imwrite(out_path, blended)
