from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np

from prepare.wsi_reader import SlideReader

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("PatchManifestDataset requires torch to be installed.") from exc


class PatchManifestDataset(Dataset):
    """从 manifest CSV 读取 WSI patch 的 Dataset。"""

    def __init__(
        self,
        manifest_csv: str | Path,
        transform: Optional[Any] = None,
        return_meta: bool = False,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.transform = transform
        self.return_meta = return_meta
        self.rows: list[dict[str, Any]] = []
        self._slides: dict[str, SlideReader] = {}

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
                        "overlap": float(row.get("overlap", 0.0)),
                        "tissue_ratio": float(row.get("tissue_ratio", 0.0)),
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _get_slide(self, path: str) -> SlideReader:
        if path not in self._slides:
            self._slides[path] = SlideReader(path)
        return self._slides[path]

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        slide = self._get_slide(row["slide_path"])
        patch = slide.read_region(
            (row["x"], row["y"]), row["level"], (row["patch_size"], row["patch_size"])
        )
        patch_np = np.asarray(patch, dtype=np.float32) / 255.0
        image = torch.from_numpy(patch_np).permute(2, 0, 1)
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        if self.return_meta:
            return image, label, row
        return image, label

    def close(self) -> None:
        for slide in self._slides.values():
            slide.close()
        self._slides.clear()

    def __del__(self) -> None:
        self.close()
