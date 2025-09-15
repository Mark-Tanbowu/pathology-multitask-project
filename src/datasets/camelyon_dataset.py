import os
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class PathologyDataset(Dataset):
    """
    病理切片 Patch 数据集
    - 读取图像 patch、对应 mask 与分类标签
    - 支持 (image, mask) 同步 transform
    """
    def __init__(self, images_dir: str, masks_dir: str, labels_file: str, transform: Optional[callable]=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        self.labels_dict = {}
        for line in lines:
            name, label = line.split(',')
            self.labels_dict[name] = int(label)
        self.filenames = list(self.labels_dict.keys())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname + ".png")
        mask_path = os.path.join(self.masks_dir, fname + "_mask.png")

        image = torch.tensor(plt.imread(img_path))
        mask = torch.tensor(plt.imread(mask_path))

        if image.ndim == 3 and image.shape[2] in [3, 4]:
            image = image.permute(2, 0, 1)[:3]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        image = image.float()
        mask = mask.float()

        if self.transform:
            image, mask = self.transform((image, mask))

        label = torch.tensor(self.labels_dict[fname], dtype=torch.float32)
        return image, mask, label
