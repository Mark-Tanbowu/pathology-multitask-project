from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from prepare.tissue_mask import mask_coverage
from prepare.xml_annotations import PolygonAnnotation


def bbox_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def overlap_ratio_from_polygons(
    polygons: list[PolygonAnnotation],
    patch_x0: int,
    patch_y0: int,
    patch_size: int,
    downsample: float,
) -> float:
    """将相交多边形栅格化为 patch mask 并计算重叠比例。"""
    if not polygons:
        return 0.0

    patch_size_level0 = patch_size * downsample
    patch_bbox = (
        float(patch_x0),
        float(patch_y0),
        float(patch_x0 + patch_size_level0),
        float(patch_y0 + patch_size_level0),
    )
    mask = Image.new("L", (patch_size, patch_size), 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        if not bbox_intersects(poly.bbox, patch_bbox):
            continue
        local_points = [
            ((x - patch_x0) / downsample, (y - patch_y0) / downsample) for x, y in poly.points
        ]
        if len(local_points) < 3:
            continue
        draw.polygon(local_points, outline=255, fill=255)

    mask_np = np.asarray(mask, dtype=np.uint8)
    return float(mask_np.mean() / 255.0)


def overlap_ratio_from_mask(
    mask: np.ndarray,
    patch_x0: int,
    patch_y0: int,
    patch_size_level0: int,
    mask_downsample: float,
) -> float:
    """从全图 mask 计算 patch 的重叠比例。"""
    x0 = int(patch_x0 / mask_downsample)
    y0 = int(patch_y0 / mask_downsample)
    size = int(patch_size_level0 / mask_downsample)
    return mask_coverage(mask, x0, y0, size, size)


def label_from_overlap(
    overlap: float, pos_threshold: float, neg_threshold: float
) -> Optional[int]:
    """按阈值返回 1/0 标签，None 表示忽略样本。"""
    if overlap >= pos_threshold:
        return 1
    if overlap <= neg_threshold:
        return 0
    return None
