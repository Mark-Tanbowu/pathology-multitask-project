"""Inspect mask pixel encoding for one file or a directory of files.

Examples:
  python scripts/check_mask_encoding.py --path data/camelyon16/train/masks/tumor_001_mask.tif
  python scripts/check_mask_encoding.py --path data/camelyon16/train/masks --max-files 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import tifffile
except Exception:
    tifffile = None

try:
    from PIL import Image
except Exception:
    Image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check mask encoding values.")
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="A mask file path or a directory that contains mask files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_mask.tif",
        help="Glob pattern when --path is a directory.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max files to inspect when --path is a directory.",
    )
    parser.add_argument(
        "--max-values",
        type=int,
        default=20,
        help="Max unique values to print per level.",
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="If set, inspect every TIFF pyramid level. Otherwise inspect first level only.",
    )
    return parser.parse_args()


def list_files(path: Path, pattern: str, max_files: int) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    files = sorted(path.rglob(pattern))
    if max_files > 0:
        files = files[:max_files]
    return files


def to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        return arr[..., 0]
    return arr


def summarize_values(arr: np.ndarray, max_values: int) -> tuple[np.ndarray, np.ndarray]:
    unique_vals, counts = np.unique(arr, return_counts=True)
    if max_values > 0 and unique_vals.size > max_values:
        unique_vals = unique_vals[:max_values]
        counts = counts[:max_values]
    return unique_vals, counts


def iter_levels(mask_path: Path, all_levels: bool) -> Iterable[tuple[int, np.ndarray]]:
    if tifffile is not None:
        with tifffile.TiffFile(str(mask_path)) as tf:
            series = tf.series[0]
            levels = getattr(series, "levels", None)
            pages = levels if levels else series.pages
            if not all_levels:
                pages = [pages[0]]
            for idx, page in enumerate(pages):
                yield idx, to_2d(np.asarray(page.asarray()))
        return

    if Image is None:
        raise RuntimeError("Neither tifffile nor PIL is available for reading masks.")
    img = Image.open(mask_path)
    yield 0, to_2d(np.asarray(img))


def inspect_mask(mask_path: Path, max_values: int, all_levels: bool) -> None:
    print(f"\n[FILE] {mask_path}")
    for level_idx, arr in iter_levels(mask_path, all_levels=all_levels):
        values, counts = summarize_values(arr, max_values=max_values)
        pairs = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))
        print(
            f"  level={level_idx} shape={tuple(arr.shape)} dtype={arr.dtype} "
            f"unique_count={np.unique(arr).size}"
        )
        print(f"  values(count) -> {pairs}")


def main() -> None:
    args = parse_args()
    files = list_files(args.path, args.pattern, args.max_files)
    if not files:
        raise FileNotFoundError(
            f"No files matched under {args.path} with pattern {args.pattern}"
        )
    for p in files:
        inspect_mask(p, max_values=args.max_values, all_levels=args.all_levels)


if __name__ == "__main__":
    main()
