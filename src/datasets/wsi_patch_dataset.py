from __future__ import annotations

import csv
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


class WsiPatchDataset(Dataset):
    """基于 manifest 的 WSI 动态切 patch 数据集（图像 + mask + 标签）。"""

    def __init__(
        self,
        manifest_csv: str | Path,
        masks_dir: str | Path,
        mask_suffix: str = "_mask.tif",
        transform: Optional[Any] = None,
        return_meta: bool = False,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.return_meta = return_meta
        self.rows: list[dict[str, Any]] = []
        self._slides: dict[str, SlideReader] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._mask_downsamples: dict[str, float] = {}

        with self.manifest_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    {
                        "slide_id": row["slide_id"],
                        "slide_path": row["slide_path"],
                        "level": int(row["level"]),
                        "x": int(row["x"]),
                        "y": int(row["y"]),
                        "patch_size": int(row["patch_size"]),
                        "label": int(row["label"]),
                    }
                )

        self.label_list = [row["label"] for row in self.rows]
        self.normalize_mean = (
            torch.tensor(normalize_mean).view(3, 1, 1) if normalize_mean is not None else None
        )
        self.normalize_std = (
            torch.tensor(normalize_std).view(3, 1, 1) if normalize_std is not None else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _get_slide(self, path: str) -> SlideReader:
        if path not in self._slides:
            self._slides[path] = SlideReader(path)
        return self._slides[path]

    def _load_mask(self, slide_id: str, slide: SlideReader) -> tuple[np.ndarray, float]:
        if slide_id in self._masks:
            return self._masks[slide_id], self._mask_downsamples[slide_id]

        mask_path = self.masks_dir / f"{slide_id}{self.mask_suffix}"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if tifffile is not None:
            try:
                mask = tifffile.imread(str(mask_path))
            except ValueError as exc:
                msg = str(exc)
                if "imagecodecs" in msg.lower():
                    raise RuntimeError(
                        "tifffile 需要 imagecodecs 才能解码该压缩格式（如 LZW）。"
                    ) from exc
                raise
        else:
            mask = np.asarray(Image.open(mask_path))

        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = mask[..., 0]
        mask = mask > 0

        w0, h0 = slide.level_dimensions[0]
        mh, mw = mask.shape[:2]
        if mw == 0 or mh == 0:
            raise ValueError(f"Mask has empty shape: {mask_path}")
        ds_x = w0 / float(mw)
        ds_y = h0 / float(mh)
        downsample = (ds_x + ds_y) / 2.0

        self._masks[slide_id] = mask
        self._mask_downsamples[slide_id] = downsample
        return mask, downsample

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
        row = self.rows[idx]
        slide = self._get_slide(row["slide_path"])
        patch = slide.read_region(
            (row["x"], row["y"]), row["level"], (row["patch_size"], row["patch_size"])
        )
        patch_np = np.asarray(patch, dtype=np.float32) / 255.0
        image = torch.from_numpy(patch_np).permute(2, 0, 1)

        mask, mask_downsample = self._load_mask(row["slide_id"], slide)
        patch_size_level0 = int(round(row["patch_size"] * slide.level_downsamples[row["level"]]))
        mx0 = int(round(row["x"] / mask_downsample))
        my0 = int(round(row["y"] / mask_downsample))
        msize = max(1, int(round(patch_size_level0 / mask_downsample)))
        mask_patch = self._crop_with_pad(mask, mx0, my0, msize)
        if msize != row["patch_size"]:
            mask_patch = np.asarray(
                Image.fromarray(mask_patch.astype(np.uint8) * 255).resize(
                    (row["patch_size"], row["patch_size"]), resample=Image.NEAREST
                ),
                dtype=np.uint8,
            )
            mask_patch = mask_patch > 0
        mask_tensor = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)

        if self.transform is not None:
            image, mask_tensor = self.transform((image, mask_tensor))

        if self.normalize_mean is not None and self.normalize_std is not None:
            image = (image - self.normalize_mean) / self.normalize_std

        label = torch.tensor(row["label"], dtype=torch.float32)
        name = f"{row['slide_id']}_{row['x']}_{row['y']}"
        if self.return_meta:
            return image, mask_tensor, label, row
        return image, mask_tensor, label, name

    def close(self) -> None:
        for slide in self._slides.values():
            slide.close()
        self._slides.clear()
        self._masks.clear()
        self._mask_downsamples.clear()

    def __del__(self) -> None:
        self.close()
