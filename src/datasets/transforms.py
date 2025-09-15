"""
图像/掩膜 同步变换的简易实现（示例）。
真实项目可直接使用 albumentations 同步增强。
"""
from typing import Tuple, Callable
import torch
import random

class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        image, mask = sample
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])  # width
            mask = torch.flip(mask, dims=[2])
        return image, mask

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
