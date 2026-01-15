"""
图像/掩膜同步增强的小型实现，兼顾分类与分割任务。
真实项目可替换为 albumentations，但在此给出几何/elastic 基线。
"""

import math
import random
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Sample = Tuple[torch.Tensor, Optional[torch.Tensor]]


class HorizontalFlip:
    """水平翻转，支持 mask 可选（分类任务可以只传 image）。"""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        image, mask = sample
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])  # width 维度
            if mask is not None:
                mask = torch.flip(mask, dims=[2])
        return image, mask


class RandomAffine2D:
    """轻量随机仿射变换，包含旋转/缩放/平移/剪切。"""

    def __init__(
        self,
        degrees: float = 10.0,
        translate: float = 0.05,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        shear: float = 5.0,
        p: float = 0.7,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.shear = shear
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        image, mask = sample
        if random.random() >= self.p:
            return image, mask
        angle = math.radians(random.uniform(-self.degrees, self.degrees))
        scale = random.uniform(*self.scale_range)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        shear_x = math.radians(random.uniform(-self.shear, self.shear))
        shear_y = math.radians(random.uniform(-self.shear, self.shear))
        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale
        tan_x = math.tan(shear_x)
        tan_y = math.tan(shear_y)
        matrix = torch.tensor(
            [[cos_a + tan_y * sin_a, -sin_a + tan_x * cos_a, tx],
             [sin_a + tan_y * cos_a, cos_a + tan_x * sin_a, ty]],
            dtype=image.dtype,
            device=image.device,
        )
        grid = F.affine_grid(
            matrix.unsqueeze(0),
            size=(1, image.shape[0], image.shape[1], image.shape[2]),
            align_corners=True,
        )
        image = _apply_grid(image, grid, mode="bilinear")
        if mask is not None:
            mask = _apply_grid(mask, grid, mode="nearest")
        return image, mask


class RandomElasticDeform:
    """轻度 elastic / grid 扭曲，适合多任务统一增强。"""

    def __init__(self, alpha: float = 0.05, grid_size: int = 4, p: float = 0.3) -> None:
        self.alpha = alpha
        self.grid_size = grid_size
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        image, mask = sample
        if random.random() >= self.p:
            return image, mask
        device = image.device
        dtype = image.dtype
        height, width = image.shape[-2:]
        base_grid = _base_grid(height, width, device, dtype)
        noise = torch.randn(1, 2, self.grid_size, self.grid_size, device=device, dtype=dtype)
        displacement = F.interpolate(
            noise,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        grid = torch.clamp(base_grid + displacement * self.alpha, -1.0, 1.0)
        image = _apply_grid(image, grid, mode="bilinear")
        if mask is not None:
            mask = _apply_grid(mask, grid, mode="nearest")
        return image, mask


class Compose:
    """顺序执行一组同步变换。"""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class BaseAug:
    """统一的几何 + 轻量 elastic 增强组合，分类/分割共享设置。"""

    def __init__(
        self,
        flip_p: float = 0.5,
        affine_kwargs: Optional[dict] = None,
        elastic_kwargs: Optional[dict] = None,
    ) -> None:
        transforms = []
        if flip_p > 0:
            transforms.append(HorizontalFlip(p=flip_p))
        transforms.append(RandomAffine2D(**(affine_kwargs or {})))
        transforms.append(RandomElasticDeform(**(elastic_kwargs or {})))
        self.pipeline = Compose(transforms)

    def __call__(self, sample: Sample) -> Sample:
        return self.pipeline(sample)


def _apply_grid(tensor: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    batched = tensor.unsqueeze(0)
    warped = F.grid_sample(
        batched,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(0)


def _base_grid(height: int, width: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return grid
