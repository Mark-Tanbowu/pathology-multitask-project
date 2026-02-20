from __future__ import annotations

import csv
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from prepare.wsi_reader import SlideReader

try:
    import tifffile
except Exception:
    tifffile = None


def _downsample_nearest(arr: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return arr
    h, w = arr.shape[:2]
    if max(h, w) <= max_size:
        return arr
    scale = max_size / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int64)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int64)
    return arr[np.ix_(ys, xs)]


def _load_mask_array(
    mask_path: Path, mask_level: int | None, mask_max_size: int | None
) -> np.ndarray:
    if tifffile is not None:
        with tifffile.TiffFile(str(mask_path)) as tif:
            series = tif.series[0]
            levels = getattr(series, "levels", None)
            pages = levels if levels else series.pages
            page = pages[-1] if len(pages) > 1 else pages[0]
            if mask_level is not None:
                idx = max(0, min(int(mask_level), len(pages) - 1))
                page = pages[idx]
            elif mask_max_size is not None:
                for candidate in pages:
                    shape = candidate.shape
                    if max(shape[0], shape[1]) <= mask_max_size:
                        page = candidate
                        break
            mask = page.asarray()
    else:
        mask = np.asarray(Image.open(mask_path))

    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]
    if mask_max_size is not None:
        mask = _downsample_nearest(mask, mask_max_size)
    return mask > 0


class SlideCoordsDataset(Dataset):
    """按 slide 坐标文件读取 patch，支持小缓存。"""

    def __init__(
        self,
        slide_manifest_csv: str | Path,
        masks_dir: str | Path,
        mask_suffix: str = "_mask.tif",
        coords_cache_slides: int = 2,
        cache_masks: bool = False,
        mask_level: int | None = None,
        mask_max_size: int | None = None,
        transform: Optional[Any] = None,
        return_meta: bool = False,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ) -> None:
        self.slide_manifest_csv = Path(slide_manifest_csv)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.coords_cache_slides = max(0, int(coords_cache_slides))
        self.cache_masks = bool(cache_masks)
        self.mask_level = mask_level
        self.mask_max_size = mask_max_size
        self.transform = transform
        self.return_meta = return_meta
        self._slides: dict[str, SlideReader] = {}
        self._coords_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mask_cache: dict[str, np.ndarray] = {}
        self.rows: list[dict[str, Any]] = []

        with self.slide_manifest_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {
                    "slide_id": row["slide_id"],
                    "slide_path": row["slide_path"],
                    "level": int(row["level"]),
                    "patch_size": int(row["patch_size"]),
                    "coords_path": row["coords_path"],
                    "num_patches": int(row.get("num_patches", 0)),
                    "slide_label": (
                        int(row["slide_label"])
                        if row.get("slide_label") not in (None, "")
                        else None
                    ),
                }
                # 保留 manifest 里的额外列（如几何统计字段），供训练/评估阶段按需读取。
                for key, value in row.items():
                    if key not in parsed_row:
                        parsed_row[key] = value
                self.rows.append(parsed_row)

        self._cum_counts: list[int] = []
        total = 0
        for row in self.rows:
            total += int(row["num_patches"])
            self._cum_counts.append(total)

        self.normalize_mean = (
            torch.tensor(normalize_mean).view(3, 1, 1) if normalize_mean is not None else None
        )
        self.normalize_std = (
            torch.tensor(normalize_std).view(3, 1, 1) if normalize_std is not None else None
        )

    def __len__(self) -> int:
        return self._cum_counts[-1] if self._cum_counts else 0

    def _get_slide(self, path: str) -> SlideReader:
        if path not in self._slides:
            self._slides[path] = SlideReader(path)
        return self._slides[path]

    def _get_coords(self, slide_id: str, coords_path: str) -> np.ndarray:
        if slide_id in self._coords_cache:
            self._coords_cache.move_to_end(slide_id)
            return self._coords_cache[slide_id]
        coords = np.load(coords_path)
        if self.coords_cache_slides > 0:
            self._coords_cache[slide_id] = coords
            self._coords_cache.move_to_end(slide_id)
            while len(self._coords_cache) > self.coords_cache_slides:
                self._coords_cache.popitem(last=False)
        return coords

    def _get_mask(self, slide_id: str) -> np.ndarray:
        if self.cache_masks and slide_id in self._mask_cache:
            return self._mask_cache[slide_id]
        mask_path = self.masks_dir / f"{slide_id}{self.mask_suffix}"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = _load_mask_array(mask_path, self.mask_level, self.mask_max_size)
        if self.cache_masks:
            self._mask_cache[slide_id] = mask
        return mask

    @staticmethod
    def _crop_with_pad(arr: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
        h, w = arr.shape[:2]
        x1 = x0 + size
        y1 = y0 + size
        out = np.zeros((size, size), dtype=arr.dtype)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)
        if src_x0 >= src_x1 or src_y0 >= src_y1:
            return out
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = (
            arr[src_y0:src_y1, src_x0:src_x1]
        )
        return out

    def __getitem__(self, idx: int):
        slide_idx = bisect_right(self._cum_counts, idx)
        prev = 0 if slide_idx == 0 else self._cum_counts[slide_idx - 1]
        local_idx = idx - prev
        row = self.rows[slide_idx]

        coords = self._get_coords(row["slide_id"], row["coords_path"])
        if local_idx >= len(coords):
            raise IndexError("Local index out of range for coords.")
        x0, y0, label = coords[local_idx].tolist()
        level = row["level"]
        patch_size = row["patch_size"]

        slide = self._get_slide(row["slide_path"])
        patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
        patch_np = np.asarray(patch, dtype=np.float32) / 255.0
        image = torch.from_numpy(patch_np).permute(2, 0, 1)

        mask = self._get_mask(row["slide_id"])
        w0, h0 = slide.level_dimensions[0]
        mh, mw = mask.shape[:2]
        ds_x = w0 / float(mw)
        ds_y = h0 / float(mh)
        mask_downsample = (ds_x + ds_y) / 2.0
        patch_size_level0 = int(round(patch_size * slide.level_downsamples[level]))
        mx0 = int(round(x0 / mask_downsample))
        my0 = int(round(y0 / mask_downsample))
        msize = max(1, int(round(patch_size_level0 / mask_downsample)))
        mask_patch = self._crop_with_pad(mask, mx0, my0, msize)
        if msize != patch_size:
            mask_patch = np.asarray(
                Image.fromarray(mask_patch.astype(np.uint8) * 255).resize(
                    (patch_size, patch_size), resample=Image.NEAREST
                ),
                dtype=np.uint8,
            )
            mask_patch = mask_patch > 0
        mask_tensor = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)

        if self.transform is not None:
            image, mask_tensor = self.transform((image, mask_tensor))

        if self.normalize_mean is not None and self.normalize_std is not None:
            image = (image - self.normalize_mean) / self.normalize_std

        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        name = f"{row['slide_id']}_{x0}_{y0}"
        if self.return_meta:
            return image, mask_tensor, label_tensor, row
        return image, mask_tensor, label_tensor, name

    def close(self) -> None:
        for slide in self._slides.values():
            slide.close()
        self._slides.clear()
        self._coords_cache.clear()
        self._mask_cache.clear()

    def __del__(self) -> None:
        self.close()
