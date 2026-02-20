from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from prepare.wsi_reader import SlideReader


def otsu_threshold(gray: np.ndarray) -> int:
    """计算 8-bit 灰度图的 Otsu 阈值。"""
    hist = np.bincount(gray.ravel(), minlength=256)
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def compute_tissue_mask(image: Image.Image, invert: bool = True) -> np.ndarray:
    """基于 Otsu 阈值从 RGB 图像生成 tissue mask。"""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = otsu_threshold(gray)
    mask = gray < threshold if invert else gray >= threshold
    return mask


def build_tissue_mask(
    slide: SlideReader, level: int, max_size: int | None = None
) -> tuple[np.ndarray, int]:
    """读取低分辨率 level 并生成 tissue mask，支持限制最大边长。"""
    if max_size is None:
        width, height = slide.level_dimensions[level]
        overview = slide.read_region((0, 0), level, (width, height))
        return compute_tissue_mask(overview), level

    lvl = level
    n_levels = len(slide.level_dimensions)
    width, height = slide.level_dimensions[lvl]
    while max(width, height) > max_size and (lvl + 1) < n_levels:
        lvl += 1
        width, height = slide.level_dimensions[lvl]

    overview = slide.read_region((0, 0), lvl, (width, height))
    if max(overview.size) > max_size:
        overview = overview.copy()
        overview.thumbnail((max_size, max_size))

    return compute_tissue_mask(overview), lvl


def mask_coverage(mask: np.ndarray, x0: int, y0: int, width: int, height: int) -> float:
    """计算 mask 窗口内的前景覆盖率。"""
    if width <= 0 or height <= 0:
        return 0.0
    x1 = max(x0 + width, x0)
    y1 = max(y0 + height, y0)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(mask.shape[1], x1)
    y1 = min(mask.shape[0], y1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    window = mask[y0:y1, x0:x1]
    return float(window.mean())
