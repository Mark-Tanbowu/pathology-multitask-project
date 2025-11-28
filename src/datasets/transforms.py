"""
图像/掩膜 同步变换的简易实现（示例）。
真实项目可直接使用 albumentations 同步增强。

意义：
    - 保证数据增强对图像与掩膜一致，使分割标签不被破坏；
    - 代码保持极简，展示接口约定，方便后续替换为更强的增强库。
"""

import random
from typing import Tuple

import torch


class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        image, mask = sample
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])  # width 翻转，保证肿瘤区域左右对称增强
            mask = torch.flip(mask, dims=[2])
        return image, mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        # 顺序执行一系列同步变换
        for t in self.transforms:
            sample = t(sample)
        return sample
