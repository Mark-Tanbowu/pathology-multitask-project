from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from prepare.patch_labeling import (
    label_from_overlap,
    overlap_ratio_from_mask,
    overlap_ratio_from_polygons,
)
from prepare.tissue_mask import build_tissue_mask, mask_coverage
from prepare.wsi_reader import SlideReader
from prepare.xml_annotations import load_asap_xml

SLIDE_EXTS = {".tif", ".tiff", ".svs", ".mrxs", ".ndpi"}

try:
    import tifffile
except Exception:
    tifffile = None

try:
    from PIL import Image
except Exception:
    Image = None


def find_slides(slides_dir: Path) -> list[Path]:
    return [p for p in slides_dir.rglob("*") if p.suffix.lower() in SLIDE_EXTS]


def parse_groups(groups: str | None) -> list[str] | None:
    if not groups:
        return None
    return [g.strip() for g in groups.split(",") if g.strip()]


def resolve_mask_path(slide_path: Path, masks_dir: Path, mask_suffix: str) -> Path:
    return masks_dir / f"{slide_path.stem}{mask_suffix}"


def load_mask_array(mask_path: Path) -> np.ndarray:
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
        if Image is None:
            raise RuntimeError("PIL not available to read mask.")
        mask = np.asarray(Image.open(mask_path))

    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]
    return mask > 0


def compute_mask_downsample(slide: SlideReader, mask: np.ndarray) -> float:
    w0, h0 = slide.level_dimensions[0]
    mh, mw = mask.shape[:2]
    if mw == 0 or mh == 0:
        raise ValueError("Mask has empty shape.")
    ds_x = w0 / float(mw)
    ds_y = h0 / float(mh)
    return (ds_x + ds_y) / 2.0


def build_manifest(
    slides_dir: Path,
    annotations_dir: Path | None,
    masks_dir: Path | None,
    output_csv: Path,
    level: int,
    patch_size: int,
    stride: int,
    tissue_level: int | None,
    min_tissue: float,
    pos_threshold: float,
    neg_threshold: float,
    neg_keep_prob: float,
    seed: int,
    groups: Iterable[str] | None,
    max_patches_per_slide: int | None,
    mask_suffix: str,
    prefer_masks: bool,
) -> None:
    slides = find_slides(slides_dir)
    if not slides:
        raise FileNotFoundError(f"未在目录中找到 WSI 文件：{slides_dir}")

    rng = random.Random(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slide_id",
                "slide_path",
                "level",
                "x",
                "y",
                "patch_size",
                "label",
                "overlap",
                "tissue_ratio",
            ],
        )
        writer.writeheader()

        total = 0
        total_pos = 0
        total_neg = 0
        for slide_path in slides:
            slide_id = slide_path.stem
            xml_path = None
            if annotations_dir is not None:
                candidate = annotations_dir / f"{slide_id}.xml"
                if candidate.exists():
                    xml_path = candidate
            mask_path = None
            if masks_dir is not None:
                candidate = resolve_mask_path(slide_path, masks_dir, mask_suffix)
                if candidate.exists():
                    mask_path = candidate

            with SlideReader(slide_path) as slide:
                if level >= len(slide.level_dimensions):
                    raise ValueError(f"Slide {slide_id} 仅有 {len(slide.level_dimensions)} 个 level。")
                patch_downsample = slide.level_downsamples[level]
                patch_size_level0 = int(patch_size * patch_downsample)

                tissue_level_use = tissue_level if tissue_level is not None else level
                if tissue_level_use >= len(slide.level_dimensions):
                    raise ValueError(
                        f"Slide {slide_id} 仅有 {len(slide.level_dimensions)} 个 level。"
                    )
                tissue_downsample = slide.level_downsamples[tissue_level_use]
                tissue_mask = None
                if min_tissue > 0.0:
                    tissue_mask = build_tissue_mask(slide, tissue_level_use)

                polygons = load_asap_xml(xml_path, include_groups=groups) if xml_path else []
                mask = None
                mask_downsample = None
                if mask_path is not None:
                    mask = load_mask_array(mask_path)
                    mask_downsample = compute_mask_downsample(slide, mask)
                use_mask = mask is not None and (prefer_masks or not polygons)
                width, height = slide.level_dimensions[level]

                patches_written = 0
                for y in range(0, height - patch_size + 1, stride):
                    for x in range(0, width - patch_size + 1, stride):
                        x0 = int(x * patch_downsample)
                        y0 = int(y * patch_downsample)

                        if tissue_mask is not None:
                            tissue_x = int(x0 / tissue_downsample)
                            tissue_y = int(y0 / tissue_downsample)
                            tissue_size = int(patch_size_level0 / tissue_downsample)
                            tissue_ratio = mask_coverage(
                                tissue_mask, tissue_x, tissue_y, tissue_size, tissue_size
                            )
                            if tissue_ratio < min_tissue:
                                continue
                        else:
                            tissue_ratio = 1.0

                        if use_mask and mask_downsample is not None:
                            overlap = overlap_ratio_from_mask(
                                mask, x0, y0, patch_size_level0, mask_downsample
                            )
                        else:
                            overlap = overlap_ratio_from_polygons(
                                polygons, x0, y0, patch_size, patch_downsample
                            )
                        label = label_from_overlap(overlap, pos_threshold, neg_threshold)
                        if label is None:
                            continue
                        if label == 0 and rng.random() > neg_keep_prob:
                            continue

                        writer.writerow(
                            {
                                "slide_id": slide_id,
                                "slide_path": str(slide_path),
                                "level": level,
                                "x": x0,
                                "y": y0,
                                "patch_size": patch_size,
                                "label": label,
                                "overlap": f"{overlap:.6f}",
                                "tissue_ratio": f"{tissue_ratio:.6f}",
                            }
                        )
                        total += 1
                        if label == 1:
                            total_pos += 1
                        else:
                            total_neg += 1
                        patches_written += 1
                        if max_patches_per_slide and patches_written >= max_patches_per_slide:
                            break
                    if max_patches_per_slide and patches_written >= max_patches_per_slide:
                        break

        print(
            "Manifest 生成完成。",
            f"Total={total}",
            f"Pos={total_pos}",
            f"Neg={total_neg}",
        )


def build_manifest_from_config(config: dict) -> None:
    """Hydra 接入占位函数。

    建议配置键（不强制）：
    - prepare.slides_dir
    - prepare.annotations_dir
    - prepare.masks_dir
    - prepare.manifest_path
    - prepare.patch_level
    - prepare.patch_size
    - prepare.patch_stride
    - prepare.tissue_level
    - prepare.min_tissue
    - prepare.pos_threshold
    - prepare.neg_threshold
    - prepare.neg_keep_prob
    - prepare.seed
    - prepare.groups
    - prepare.max_patches_per_slide
    - prepare.mask_suffix
    - prepare.prefer_masks
    """
    data_cfg = config.get("prepare", config.get("data", {}))
    tissue_level = data_cfg.get("tissue_level")
    max_patches = data_cfg.get("max_patches_per_slide")
    build_manifest(
        slides_dir=Path(data_cfg["slides_dir"]),
        annotations_dir=Path(data_cfg["annotations_dir"]) if data_cfg.get("annotations_dir") else None,
        masks_dir=Path(data_cfg["masks_dir"]) if data_cfg.get("masks_dir") else None,
        output_csv=Path(data_cfg["manifest_path"]),
        level=int(data_cfg.get("patch_level", 0)),
        patch_size=int(data_cfg.get("patch_size", 256)),
        stride=int(data_cfg.get("patch_stride", 256)),
        tissue_level=int(tissue_level) if tissue_level is not None else None,
        min_tissue=float(data_cfg.get("min_tissue", 0.5)),
        pos_threshold=float(data_cfg.get("pos_threshold", 0.5)),
        neg_threshold=float(data_cfg.get("neg_threshold", 0.0)),
        neg_keep_prob=float(data_cfg.get("neg_keep_prob", 1.0)),
        seed=int(data_cfg.get("seed", 42)),
        groups=parse_groups(data_cfg.get("groups", "Tumor")),
        max_patches_per_slide=int(max_patches) if max_patches is not None else None,
        mask_suffix=str(data_cfg.get("mask_suffix", "_mask.tif")),
        prefer_masks=bool(data_cfg.get("prefer_masks", True)),
    )


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CAMELYON 的 patch manifest。")
    parser.add_argument("--config", type=Path, help="YAML 配置路径（优先使用）")
    parser.add_argument("--slides-dir", type=Path)
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--mask-suffix", type=str, default="_mask.tif")
    parser.add_argument("--prefer-masks", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tissue-level", type=int)
    parser.add_argument("--min-tissue", type=float, default=0.5)
    parser.add_argument("--pos-threshold", type=float, default=0.5)
    parser.add_argument("--neg-threshold", type=float, default=0.0)
    parser.add_argument("--neg-keep-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=str, default="Tumor")
    parser.add_argument("--max-patches-per-slide", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config = load_yaml_config(args.config)
        build_manifest_from_config(config)
        return
    if args.slides_dir is None or args.output_csv is None:
        raise ValueError("需要 --slides-dir 与 --output-csv，或使用 --config。")
    groups = parse_groups(args.groups)
    build_manifest(
        slides_dir=args.slides_dir,
        annotations_dir=args.annotations_dir,
        masks_dir=args.masks_dir,
        output_csv=args.output_csv,
        level=args.level,
        patch_size=args.patch_size,
        stride=args.stride,
        tissue_level=args.tissue_level,
        min_tissue=args.min_tissue,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        neg_keep_prob=args.neg_keep_prob,
        seed=args.seed,
        groups=groups,
        max_patches_per_slide=args.max_patches_per_slide,
        mask_suffix=args.mask_suffix,
        prefer_masks=args.prefer_masks,
    )


if __name__ == "__main__":
    main()
