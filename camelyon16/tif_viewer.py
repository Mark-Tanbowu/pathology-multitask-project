import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    import multiresolutionimageinterface as mir  # ASAP / mir
except Exception:
    mir = None

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import openslide
except Exception:
    openslide = None

try:
    from PIL import Image
except Exception:
    Image = None


def safe_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    stats = {
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "ndim": int(arr.ndim),
    }
    if arr.size == 0:
        stats.update({"min": None, "max": None, "mean": None})
        stats["unique_values"] = []
        return stats

    stats["min"] = int(arr.min()) if np.issubdtype(arr.dtype, np.integer) else float(arr.min())
    stats["max"] = int(arr.max()) if np.issubdtype(arr.dtype, np.integer) else float(arr.max())
    stats["mean"] = float(arr.mean())

    uniq = np.unique(arr)
    if uniq.size > 1000:
        uniq = uniq[:1000]
        stats["unique_truncated"] = True
    else:
        stats["unique_truncated"] = False
    stats["unique_values"] = uniq.tolist()
    return stats


def downsample_nearest(arr: np.ndarray, max_size: int) -> np.ndarray:
    """Downsample a 2D array with nearest neighbor so labels remain intact."""
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


def find_bbox(
    arr: np.ndarray, labels: Optional[list[int]] = None
) -> Optional[Tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) for foreground or None if empty."""
    if labels is None:
        mask = arr != 0
    else:
        mask = np.isin(arr, labels)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return y0, y1, x0, x1


def expand_bbox(
    bbox: Tuple[int, int, int, int], shape: Tuple[int, int], margin: int
) -> Tuple[int, int, int, int]:
    """Expand bbox by margin pixels and clamp to image shape."""
    y0, y1, x0, x1 = bbox
    h, w = shape[:2]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h - 1, y1 + margin)
    x1 = min(w - 1, x1 + margin)
    return y0, y1, x0, x1


def try_load_with_mir(path: Path, max_size: int) -> tuple[np.ndarray, str]:
    if mir is None:
        raise RuntimeError("mir not available")

    reader = mir.MultiResolutionImageReader()
    wsi = reader.open(str(path))
    try:
        level_count = wsi.getNumberOfLevels()
        chosen_level = level_count - 1
        chosen_w, chosen_h = wsi.getLevelDimensions(chosen_level)

        if max_size and max_size > 0:
            for lvl in range(level_count):
                w, h = wsi.getLevelDimensions(lvl)
                if max(w, h) <= max_size:
                    chosen_level = lvl
                    chosen_w, chosen_h = w, h
                    break

        patch = wsi.getUCharPatch(
            startX=0, startY=0, width=chosen_w, height=chosen_h, level=chosen_level
        )
        arr = np.asarray(patch).squeeze()

        if arr.ndim != 2:
            raise RuntimeError(f"mir returned unexpected shape: {arr.shape}")

        info = f"backend=mir level={chosen_level} size={chosen_w}x{chosen_h}"
        return arr, info
    finally:
        try:
            wsi.close()
        except Exception:
            pass


def try_load_with_tifffile(path: Path, max_size: int) -> tuple[np.ndarray, str]:
    if tifffile is None:
        raise RuntimeError("tifffile not available")

    try:
        arr = tifffile.imread(str(path))
    except ValueError as exc:
        msg = str(exc)
        if "imagecodecs" in msg.lower():
            raise RuntimeError(
                "tifffile 需要 imagecodecs 才能解码该压缩格式（如 LZW）。请先安装：pip install imagecodecs"
            ) from exc
        raise
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise RuntimeError(f"tifffile returned unexpected shape: {arr.shape}")

    arr_small = downsample_nearest(arr, max_size) if max_size else arr
    info = f"backend=tifffile orig={arr.shape[1]}x{arr.shape[0]} preview={arr_small.shape[1]}x{arr_small.shape[0]}"
    return arr_small, info


def try_load_with_openslide(path: Path, max_size: int) -> tuple[np.ndarray, str]:
    if openslide is None:
        raise RuntimeError("openslide not available")

    slide = openslide.OpenSlide(str(path))
    try:
        level_count = slide.level_count
        chosen_level = level_count - 1
        chosen_w, chosen_h = slide.level_dimensions[chosen_level]
        if max_size and max_size > 0:
            for lvl, (w, h) in enumerate(slide.level_dimensions):
                if max(w, h) <= max_size:
                    chosen_level = lvl
                    chosen_w, chosen_h = w, h
                    break
        img = slide.read_region((0, 0), chosen_level, (chosen_w, chosen_h)).convert("L")
        arr = np.array(img).squeeze()
        if arr.ndim != 2:
            raise RuntimeError(f"openslide returned unexpected shape: {arr.shape}")
        info = f"backend=openslide level={chosen_level} size={chosen_w}x{chosen_h}"
        return arr, info
    finally:
        try:
            slide.close()
        except Exception:
            pass


def try_load_with_pil(path: Path, max_size: int, allow_huge: bool) -> tuple[np.ndarray, str]:
    if Image is None:
        raise RuntimeError("PIL not available")

    # decompression bomb settings
    if allow_huge:
        Image.MAX_IMAGE_PIXELS = None

    img = Image.open(path)

    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")

    w, h = img.size
    if max_size and max(w, h) > max_size:
        img.thumbnail((max_size, max_size), Image.NEAREST)

    arr = np.array(img).squeeze()
    if arr.ndim != 2:
        raise RuntimeError(f"PIL returned unexpected shape: {arr.shape}")
    info = f"backend=pil size={img.size[0]}x{img.size[1]} mode={img.mode}"
    return arr, info


def load_mask_preview(path: Path, max_size: int, allow_huge: bool) -> tuple[np.ndarray, str]:
    errors = []

    if mir is not None:
        try:
            return try_load_with_mir(path, max_size)
        except Exception as e:
            errors.append(f"mir failed: {type(e).__name__}: {e}")

    if openslide is not None:
        try:
            return try_load_with_openslide(path, max_size)
        except Exception as e:
            errors.append(f"openslide failed: {type(e).__name__}: {e}")

    if tifffile is not None:
        try:
            return try_load_with_tifffile(path, max_size)
        except Exception as e:
            errors.append(f"tifffile failed: {type(e).__name__}: {e}")

    if Image is not None:
        try:
            return try_load_with_pil(path, max_size, allow_huge)
        except Exception as e:
            errors.append(f"PIL failed: {type(e).__name__}: {e}")

    msg = "All backends failed.\n" + "\n".join(errors)
    raise SystemExit(msg)


def save_previews(
    arr: np.ndarray,
    out_dir: Path,
    stem: str,
    bbox: Optional[Tuple[int, int, int, int]],
    crop_margin: int,
) -> dict:
    out = {}

    gray_path = out_dir / f"{stem}_mask_gray.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="gray")
    plt.title("Mask (gray preview)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(gray_path, dpi=200)
    plt.close()
    out["gray_preview"] = str(gray_path)

    raw_path = out_dir / f"{stem}_mask_raw.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="tab10", vmin=0, vmax=2)
    if bbox is not None:
        y0, y1, x0, x1 = expand_bbox(bbox, arr.shape, crop_margin)
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        plt.gca().add_patch(rect)
    plt.title("Mask (discrete labels)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(raw_path, dpi=200)
    plt.close()
    out["label_preview"] = str(raw_path)

    if bbox is not None:
        y0, y1, x0, x1 = expand_bbox(bbox, arr.shape, crop_margin)
        crop = arr[y0 : y1 + 1, x0 : x1 + 1]
        crop_path = out_dir / f"{stem}_mask_crop.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(crop, cmap="tab10", vmin=0, vmax=2)
        plt.title("Mask (crop preview)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(crop_path, dpi=200)
        plt.close()
        out["crop_preview"] = str(crop_path)

    return out


def show_previews(
    arr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    crop_margin: int,
) -> None:
    if bbox is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(arr, cmap="gray")
        axes[0].set_title("Mask (gray)")
        axes[1].imshow(arr, cmap="tab10", vmin=0, vmax=2)
        axes[1].set_title("Mask (labels)")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        plt.show()
        return

    y0, y1, x0, x1 = expand_bbox(bbox, arr.shape, crop_margin)
    crop = arr[y0 : y1 + 1, x0 : x1 + 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("Mask (gray)")
    axes[1].imshow(arr, cmap="tab10", vmin=0, vmax=2)
    axes[1].set_title("Mask (labels)")
    axes[2].imshow(crop, cmap="tab10", vmin=0, vmax=2)
    axes[2].set_title("Mask (crop)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAMELYON16 mask preview (safe): mir -> tifffile -> PIL fallback, nearest downsample."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to .tif mask (default: camelyon16/masks/normal_001_mask.tif if exists)",
    )
    parser.add_argument("--out_dir", default="run/mask_preview", help="Output directory")
    parser.add_argument("--max_size", type=int, default=2048, help="Max longer side in pixels for preview")
    parser.add_argument("--crop_margin", type=int, default=12, help="Extra margin for crop preview")
    parser.add_argument(
        "--crop_labels",
        default="1,2",
        help="Comma-separated labels for crop bbox (empty -> any non-zero)",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive preview windows")
    parser.add_argument("--allow_huge", action="store_true", help="Disable PIL decompression bomb check")
    args = parser.parse_args()

    default_path = Path("camelyon16/masks/normal_001_mask.tif")
    if args.input:
        in_path = Path(args.input)
    else:
        in_path = default_path
        if not in_path.exists():
            raise FileNotFoundError(
                "Default mask not found. Provide --input or place camelyon16/masks/normal_001_mask.tif"
            )
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr, backend_info = load_mask_preview(in_path, args.max_size, args.allow_huge)
    stats = safe_stats(arr)

    mask_info = {}
    if stats["unique_values"]:
        total = arr.size
        for v in [0, 1, 2, 255]:
            cnt = int((arr == v).sum())
            if cnt > 0:
                mask_info[f"count_{v}"] = cnt
                mask_info[f"ratio_{v}"] = float(cnt / total)

    crop_labels = []
    if args.crop_labels.strip():
        for part in args.crop_labels.split(","):
            part = part.strip()
            if part:
                try:
                    crop_labels.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid crop label: {part}") from None
    bbox = find_bbox(arr, labels=crop_labels or None)

    previews = save_previews(arr, out_dir, in_path.stem, bbox, args.crop_margin)

    log_lines = []
    log_lines.append(f"input={in_path}")
    log_lines.append(backend_info)
    log_lines.append(f"stats={stats}")
    if mask_info:
        log_lines.append(f"mask_info={mask_info}")
    if bbox is not None:
        log_lines.append(f"crop_bbox={bbox} (margin={args.crop_margin})")
    log_lines.append(f"previews={previews}")

    log_path = out_dir / f"{in_path.stem}.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"[OK] wrote to: {out_dir}")
    print(f"  - {previews['label_preview']}")
    print(f"  - {previews['gray_preview']}")
    if "crop_preview" in previews:
        print(f"  - {previews['crop_preview']}")
    print(f"  - log: {log_path}")

    if args.show:
        show_previews(arr, bbox, args.crop_margin)


if __name__ == "__main__":
    main()
