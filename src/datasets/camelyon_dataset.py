import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PathologyDataset(Dataset):
    """
    数字病理 Patch 数据集（适配二类分割：肿瘤 vs 背景）

    功能说明：
    - 加载 patch 图像、对应 mask 与分类标签；
    - 图像自动缩放至 [0,1]；
    - mask 自动二值化为 {0,1}；
    - 保留详细日志信息，方便调试；
    - 训练集与验证集共享完全一致的处理逻辑。
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        labels_file: str,
        transform: Optional[Callable] = None,
        debug_log: bool = True,  # 是否打印详细日志（仅首样本）
    ):
        """
        Args:
            images_dir: 图像文件路径（patch）
            masks_dir: 掩膜文件路径（对应 _anno.bmp）
            labels_file: 分类标签文件（CSV-like，格式：name,label）
            transform: 可选的数据增强函数
            debug_log: 是否打印首样本调试信息
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.debug_log = debug_log

        # ---------- 读取分类标签 ----------
        with open(labels_file, "r", encoding="utf-8") as f:
            lines = [li.strip() for li in f if li.strip()]
        self.labels_dict = {}
        for line in lines:
            name, label = line.split(",")
            self.labels_dict[name] = int(label)

        self.filenames = list(self.labels_dict.keys())
        self.transform = transform

        if debug_log:
            print("=" * 70)
            print(f"[INFO] 初始化 PathologyDataset")
            print(f"  图像目录: {images_dir}")
            print(f"  掩膜目录: {masks_dir}")
            print(f"  标签文件: {labels_file}")
            print(f"  样本数量: {len(self.filenames)}")
            print("=" * 70)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname + ".bmp")
        mask_path = os.path.join(self.masks_dir, fname + "_anno.bmp")

        # ---------- 读取图像 ----------
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image, dtype=np.float32)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1) / 255.0  # → [0,1]

        # --- 读取掩膜并二值化 ---
        mask = Image.open(mask_path)

        # ✅ 强制转换为灰度模式 (L)，防止调色板或RGB干扰
        mask = mask.convert("L")

        # 转为 numpy 数组
        mask_np = np.array(mask, dtype=np.float32)

        # ✅ 统一归一化（无论范围 0~9 还是 0~255）
        if mask_np.max() > 1.0:
            mask_np /= mask_np.max()

        # ✅ 二值化：所有非零像素视为肿瘤
        mask_bin = (mask_np > 0.01).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)
        # ---------- 读取分类标签 ----------
        label = torch.tensor(self.labels_dict[fname], dtype=torch.float32)

        # ---------- 应用图像增强 ----------
        if self.transform:
            image_tensor, mask_tensor = self.transform((image_tensor, mask_tensor))

        # ---------- 调试日志 ----------
        if self.debug_log and idx == 0:
            print("\n[DEBUG] ✅ 样本加载成功")
            print(f"  样本名：{fname}")
            print(f"  图像路径：{img_path}")
            print(f"  掩膜路径：{mask_path}")
            print(f"  图像 shape: {image_tensor.shape}, dtype={image_tensor.dtype}")
            print(f"  掩膜 shape: {mask_tensor.shape}, dtype={mask_tensor.dtype}")
            print(f"  图像像素范围: ({image_tensor.min().item():.3f}, {image_tensor.max().item():.3f})")
            print(f"  掩膜像素范围: ({mask_tensor.min().item():.3f}, {mask_tensor.max().item():.3f})")
            print(f"  分类标签: {label.item()}")
            print("=" * 70)

        return image_tensor, mask_tensor, label


# ================ 冒烟测试 ================
if __name__ == "__main__":
    from pathlib import Path
    import yaml

    # 自动回到项目根目录
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "defaults.yaml"
    config = yaml.safe_load(open(config_path, encoding="utf-8"))

    # 初始化数据集
    dataset = PathologyDataset(
        images_dir=root / config["data"]["train_images"],
        masks_dir=root / config["data"]["train_masks"],
        labels_file=root / config["data"]["train_labels"],
        debug_log=True,
    )

    # 样本检查
    image, mask, label = dataset[0]
    print("[INFO] 数据集冒烟测试完成 ✅")
