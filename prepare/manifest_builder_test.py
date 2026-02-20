from __future__ import annotations

import argparse
import csv
import random
import time
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
from prepare.xml_annotations import PolygonAnnotation, load_asap_xml

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


def infer_camelyon_slide_label(slide_id: str) -> int | None:
    """从 CAMELYON 命名前缀推断 slide 标签。"""
    sid = slide_id.lower()
    if sid.startswith("tumor_"):
        return 1
    if sid.startswith("normal_"):
        return 0
    return None


def load_slide_label_map(slide_labels_csv: Path | None) -> dict[str, int]:
    if slide_labels_csv is None:
        return {}
    if not slide_labels_csv.exists():
        raise FileNotFoundError(f"slide_labels_csv not found: {slide_labels_csv}")

    label_map: dict[str, int] = {}
    with slide_labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("slide_id", "")).strip()
            if not sid:
                continue
            raw_label = row.get("slide_label", row.get("label"))
            if raw_label in (None, ""):
                continue
            try:
                label_map[sid] = int(raw_label)
            except ValueError as exc:
                raise ValueError(f"Invalid slide label for {sid}: {raw_label}") from exc
    return label_map


def resolve_mask_path(slide_path: Path, masks_dir: Path, mask_suffix: str) -> Path:
    return masks_dir / f"{slide_path.stem}{mask_suffix}"


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


def load_mask_array_with_options(
    mask_path: Path,
    mask_level: int | None,
    mask_max_size: int | None,
) -> np.ndarray:
    if tifffile is not None:
        try:
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
    if mask_max_size is not None:
        mask = _downsample_nearest(mask, mask_max_size)
    return mask > 0


def compute_mask_downsample(slide: SlideReader, mask: np.ndarray) -> float:
    w0, h0 = slide.level_dimensions[0]
    mh, mw = mask.shape[:2]
    if mw == 0 or mh == 0:
        raise ValueError("Mask has empty shape.")
    ds_x = w0 / float(mw)
    ds_y = h0 / float(mh)
    return (ds_x + ds_y) / 2.0


def select_stratified_random(
    pos_candidates: list[dict[str, float | int]],
    neg_candidates: list[dict[str, float | int]],
    max_patches_per_slide: int,
    target_pos_ratio: float | None,
    rng: random.Random,
) -> list[dict[str, float | int]]:
    """按类别分层随机抽样，控制每张 slide 的 patch 上限。"""
    total_available = len(pos_candidates) + len(neg_candidates)
    if total_available == 0:
        return []
    k = min(int(max_patches_per_slide), total_available)
    if k <= 0:
        return []

    if target_pos_ratio is None:
        ratio = len(pos_candidates) / float(total_available)
    else:
        ratio = float(max(0.0, min(1.0, target_pos_ratio)))

    take_pos = min(len(pos_candidates), int(round(k * ratio)))
    take_neg = min(len(neg_candidates), k - take_pos)
    remaining = k - (take_pos + take_neg)
    if remaining > 0:
        extra_pos = min(len(pos_candidates) - take_pos, remaining)
        take_pos += extra_pos
        remaining -= extra_pos
    if remaining > 0:
        extra_neg = min(len(neg_candidates) - take_neg, remaining)
        take_neg += extra_neg

    selected: list[dict[str, float | int]] = []
    if take_pos > 0:
        selected.extend(rng.sample(pos_candidates, take_pos))
    if take_neg > 0:
        selected.extend(rng.sample(neg_candidates, take_neg))
    rng.shuffle(selected)
    return selected


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Shoelace 公式计算多边形面积（level-0 像素面积）。"""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def polygon_perimeter(points: list[tuple[float, float]]) -> float:
    """计算多边形周长（level-0 像素长度）。"""
    if len(points) < 2:
        return 0.0
    p = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        p += float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return p


def summarize_polygons(polygons: list[PolygonAnnotation], slide_area: float) -> dict[str, float]:
    """汇总可选几何统计：数量、面积、周长、占比等。"""
    if not polygons:
        return {
            "count": 0.0,
            "total_area": 0.0,
            "mean_area": 0.0,
            "max_area": 0.0,
            "total_perimeter": 0.0,
            "mean_perimeter": 0.0,
            "area_ratio": 0.0,
        }

    areas = [polygon_area(poly.points) for poly in polygons]
    perimeters = [polygon_perimeter(poly.points) for poly in polygons]
    total_area = float(sum(areas))
    return {
        "count": float(len(polygons)),
        "total_area": total_area,
        "mean_area": float(total_area / max(len(areas), 1)),
        "max_area": float(max(areas)),
        "total_perimeter": float(sum(perimeters)),
        "mean_perimeter": float(sum(perimeters) / max(len(perimeters), 1)),
        "area_ratio": float(total_area / max(slide_area, 1.0)),
    }


def resolve_slide_label(
    slide_id: str,
    gt_slide_label: int | None,
    xml_path: Path | None,
    sampled_has_pos: int,
    slide_label_mode: str,
    slide_label_fallback_to_sampled: bool,
) -> int:
    """统一处理 test split 的 slide 标签来源，优先级：
    CSV 显式标注 > 配置策略 > sampled 回退。
    """
    if gt_slide_label is not None:
        return int(gt_slide_label)

    mode = slide_label_mode.strip().lower()
    if mode == "xml_presence":
        return 1 if xml_path is not None else 0
    if mode == "camelyon_prefix":
        inferred = infer_camelyon_slide_label(slide_id)
        if inferred is not None:
            return int(inferred)
    if mode == "sampled":
        return int(sampled_has_pos)
    if mode not in {"none", ""}:
        raise ValueError(
            f"Unknown slide_label_mode={slide_label_mode}. "
            "Use one of: xml_presence, camelyon_prefix, sampled, none."
        )

    if slide_label_fallback_to_sampled:
        return int(sampled_has_pos)
    raise ValueError(
        f"Cannot resolve ground-truth slide_label for slide_id={slide_id}. "
        "Set slide_labels_csv, choose slide_label_mode, or enable slide_label_fallback_to_sampled."
    )


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
    ignore_groups: Iterable[str] | None,
    ignore_overlap_threshold: float,
    max_patches_per_slide: int | None,
    sampling_mode: str,
    target_pos_ratio: float | None,
    slide_labels_csv: Path | None,
    slide_label_mode: str,
    slide_label_fallback_to_sampled: bool,
    mask_suffix: str,
    prefer_masks: bool,
    progress_interval: int,
    mask_level: int | None,
    mask_max_size: int | None,
    tissue_max_size: int | None,
    coords_out_dir: Path | None,
    slide_manifest_path: Path | None,
    write_patch_manifest: bool,
    enable_geometry_stats: bool,
) -> None:
    slides = find_slides(slides_dir)
    if not slides:
        raise FileNotFoundError(f"未在目录中找到 WSI 文件：{slides_dir}")

    rng = random.Random(seed)
    sampling_mode = str(sampling_mode).strip().lower()
    if sampling_mode not in {"scan_order", "stratified_random"}:
        raise ValueError(
            f"Unknown sampling_mode={sampling_mode}. Use scan_order or stratified_random."
        )
    slide_label_map = load_slide_label_map(slide_labels_csv)
    if slide_label_map:
        print(f"[TEST][INFO] loaded {len(slide_label_map)} slide labels from {slide_labels_csv}")
    if write_patch_manifest:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        patch_f = output_csv.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            patch_f,
            fieldnames=[
                "slide_id",
                "slide_path",
                "level",
                "x",
                "y",
                "patch_size",
                "label",
                "overlap",
                "ignore_overlap",
                "tissue_ratio",
            ],
        )
        writer.writeheader()
    else:
        patch_f = None
        writer = None

    slide_rows: list[dict[str, str]] = []
    if coords_out_dir is not None:
        coords_out_dir.mkdir(parents=True, exist_ok=True)
    if slide_manifest_path is not None:
        slide_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        total = 0
        total_pos = 0
        total_neg = 0
        total_sampled_gt_mismatch = 0
        total_excluded = 0
        overall_start = time.perf_counter()
        for slide_idx, slide_path in enumerate(slides, start=1):
            slide_id = slide_path.stem
            print(f"[TEST][INFO] slide {slide_idx}/{len(slides)}: {slide_id}")
            gt_slide_label = slide_label_map.get(slide_id)
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
                    raise ValueError(
                        f"Slide {slide_id} 仅有 {len(slide.level_dimensions)} 个 level。"
                    )
                patch_downsample = slide.level_downsamples[level]
                patch_size_level0 = int(patch_size * patch_downsample)

                tissue_level_use = tissue_level if tissue_level is not None else level
                if tissue_level_use >= len(slide.level_dimensions):
                    raise ValueError(
                        f"Slide {slide_id} 仅有 {len(slide.level_dimensions)} 个 level。"
                    )
                tissue_mask = None
                tissue_level_actual = tissue_level_use
                if min_tissue > 0.0:
                    tissue_mask, tissue_level_actual = build_tissue_mask(
                        slide, tissue_level_use, max_size=tissue_max_size
                    )
                tissue_downsample = slide.level_downsamples[tissue_level_actual]

                tumor_polygons = load_asap_xml(xml_path, include_groups=groups) if xml_path else []
                ignore_polygons = (
                    load_asap_xml(xml_path, include_groups=ignore_groups)
                    if xml_path and ignore_groups
                    else []
                )

                mask = None
                mask_downsample = None
                if mask_path is not None:
                    mask = load_mask_array_with_options(mask_path, mask_level, mask_max_size)
                    mask_downsample = compute_mask_downsample(slide, mask)
                # test 默认保持与 train/val 一致：配置了 annotations_dir 时优先按 XML 标注。
                use_annotations_for_labels = annotations_dir is not None
                use_mask = (mask is not None and (prefer_masks or not tumor_polygons)) and (
                    not use_annotations_for_labels
                )
                width, height = slide.level_dimensions[level]
                slide_area = float(slide.level_dimensions[0][0] * slide.level_dimensions[0][1])
                if enable_geometry_stats:
                    tumor_stats = summarize_polygons(tumor_polygons, slide_area)
                    ignore_stats = summarize_polygons(ignore_polygons, slide_area)
                else:
                    tumor_stats = summarize_polygons([], slide_area)
                    ignore_stats = summarize_polygons([], slide_area)

                all_candidates: list[dict[str, float | int]] = []
                pos_candidates: list[dict[str, float | int]] = []
                neg_candidates: list[dict[str, float | int]] = []
                x_steps = (width - patch_size) // stride + 1
                y_steps = (height - patch_size) // stride + 1
                total_positions = max(x_steps * y_steps, 1)
                visited_positions = 0
                excluded_by_ignore = 0
                slide_start = time.perf_counter()
                for y in range(0, height - patch_size + 1, stride):
                    for x in range(0, width - patch_size + 1, stride):
                        visited_positions += 1
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

                        # 可忽略区规则：与 Exclusion 重叠超过阈值即跳过，不参与正负监督。
                        if ignore_polygons:
                            ignore_overlap = overlap_ratio_from_polygons(
                                ignore_polygons, x0, y0, patch_size, patch_downsample
                            )
                            if ignore_overlap >= ignore_overlap_threshold:
                                excluded_by_ignore += 1
                                continue
                        else:
                            ignore_overlap = 0.0

                        if use_mask and mask_downsample is not None:
                            overlap = overlap_ratio_from_mask(
                                mask, x0, y0, patch_size_level0, mask_downsample
                            )
                        else:
                            overlap = overlap_ratio_from_polygons(
                                tumor_polygons, x0, y0, patch_size, patch_downsample
                            )
                        label = label_from_overlap(overlap, pos_threshold, neg_threshold)
                        if label is None:
                            continue
                        if label == 0 and rng.random() > neg_keep_prob:
                            continue

                        candidate = {
                            "x": x0,
                            "y": y0,
                            "label": int(label),
                            "overlap": float(overlap),
                            "ignore_overlap": float(ignore_overlap),
                            "tissue_ratio": float(tissue_ratio),
                        }
                        if sampling_mode == "stratified_random":
                            if label == 1:
                                pos_candidates.append(candidate)
                            else:
                                neg_candidates.append(candidate)
                        else:
                            all_candidates.append(candidate)
                        if progress_interval > 0 and visited_positions % progress_interval == 0:
                            elapsed = time.perf_counter() - slide_start
                            ratio = visited_positions / total_positions
                            eta = elapsed * (1.0 / max(ratio, 1e-6) - 1.0)
                            scan_kept = (
                                len(pos_candidates) + len(neg_candidates)
                                if sampling_mode == "stratified_random"
                                else len(all_candidates)
                            )
                            print(
                                f"[TEST][INFO] slide={slide_id} "
                                f"scan={visited_positions}/{total_positions} "
                                f"({ratio:.1%}) kept={scan_kept} excluded={excluded_by_ignore} "
                                f"eta={eta:.0f}s total={total} pos={total_pos} neg={total_neg}"
                            )
                        if (
                            sampling_mode == "scan_order"
                            and max_patches_per_slide
                            and len(all_candidates) >= max_patches_per_slide
                        ):
                            break
                    if (
                        sampling_mode == "scan_order"
                        and max_patches_per_slide
                        and len(all_candidates) >= max_patches_per_slide
                    ):
                        break
                if sampling_mode == "stratified_random":
                    if max_patches_per_slide is not None:
                        selected = select_stratified_random(
                            pos_candidates=pos_candidates,
                            neg_candidates=neg_candidates,
                            max_patches_per_slide=max_patches_per_slide,
                            target_pos_ratio=target_pos_ratio,
                            rng=rng,
                        )
                    else:
                        selected = pos_candidates + neg_candidates
                        rng.shuffle(selected)
                else:
                    selected = all_candidates

                patches_written = len(selected)
                slide_pos = sum(int(item["label"]) for item in selected)
                slide_neg = patches_written - slide_pos
                total += patches_written
                total_pos += slide_pos
                total_neg += slide_neg
                total_excluded += excluded_by_ignore

                if writer is not None:
                    for item in selected:
                        writer.writerow(
                            {
                                "slide_id": slide_id,
                                "slide_path": str(slide_path),
                                "level": level,
                                "x": int(item["x"]),
                                "y": int(item["y"]),
                                "patch_size": patch_size,
                                "label": int(item["label"]),
                                "overlap": f"{float(item['overlap']):.6f}",
                                "ignore_overlap": f"{float(item['ignore_overlap']):.6f}",
                                "tissue_ratio": f"{float(item['tissue_ratio']):.6f}",
                            }
                        )

                coords: list[list[int]] = []
                if coords_out_dir is not None:
                    coords = [
                        [int(item["x"]), int(item["y"]), int(item["label"])]
                        for item in selected
                    ]

            sampled_has_pos = 1 if slide_pos > 0 else 0
            locked_slide_label = resolve_slide_label(
                slide_id=slide_id,
                gt_slide_label=gt_slide_label,
                xml_path=xml_path,
                sampled_has_pos=sampled_has_pos,
                slide_label_mode=slide_label_mode,
                slide_label_fallback_to_sampled=slide_label_fallback_to_sampled,
            )

            if sampled_has_pos != locked_slide_label:
                total_sampled_gt_mismatch += 1
                print(
                    "[TEST][WARN] sampled_has_pos="
                    f"{sampled_has_pos} differs from locked slide_label="
                    f"{locked_slide_label} for {slide_id}"
                )

            slide_elapsed = time.perf_counter() - slide_start
            if coords_out_dir is not None:
                split = "tumor" if locked_slide_label > 0 else "normal"
                out_dir = coords_out_dir / split
                out_dir.mkdir(parents=True, exist_ok=True)
                coords_path = out_dir / f"{slide_id}.npy"
                coords_arr = np.asarray(coords, dtype=np.int64)
                np.save(coords_path, coords_arr)
                if slide_manifest_path is not None:
                    slide_rows.append(
                        {
                            "slide_id": slide_id,
                            "slide_path": str(slide_path),
                            "level": str(level),
                            "patch_size": str(patch_size),
                            "coords_path": str(coords_path),
                            "num_patches": str(patches_written),
                            "num_pos": str(slide_pos),
                            "num_neg": str(slide_neg),
                            "slide_label": str(locked_slide_label),
                            "sampled_has_pos": str(sampled_has_pos),
                            "xml_matched": str(1 if xml_path is not None else 0),
                            "excluded_by_ignore": str(excluded_by_ignore),
                            "tumor_polygon_count": str(int(tumor_stats["count"])),
                            "tumor_total_area": f"{tumor_stats['total_area']:.2f}",
                            "tumor_area_ratio": f"{tumor_stats['area_ratio']:.8f}",
                            "tumor_total_perimeter": f"{tumor_stats['total_perimeter']:.2f}",
                            "ignore_polygon_count": str(int(ignore_stats["count"])),
                            "ignore_total_area": f"{ignore_stats['total_area']:.2f}",
                            "ignore_area_ratio": f"{ignore_stats['area_ratio']:.8f}",
                            "ignore_total_perimeter": f"{ignore_stats['total_perimeter']:.2f}",
                        }
                    )
            print(
                f"[TEST][INFO] done slide={slide_id} patches={patches_written} "
                f"excluded={excluded_by_ignore} xml={1 if xml_path else 0} "
                f"elapsed={slide_elapsed:.1f}s total={total} pos={total_pos} neg={total_neg}"
            )

        overall_elapsed = time.perf_counter() - overall_start
        if slide_manifest_path is not None:
            with slide_manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer_slide = csv.DictWriter(
                    f,
                    fieldnames=[
                        "slide_id",
                        "slide_path",
                        "level",
                        "patch_size",
                        "coords_path",
                        "num_patches",
                        "num_pos",
                        "num_neg",
                        "slide_label",
                        "sampled_has_pos",
                        "xml_matched",
                        "excluded_by_ignore",
                        "tumor_polygon_count",
                        "tumor_total_area",
                        "tumor_area_ratio",
                        "tumor_total_perimeter",
                        "ignore_polygon_count",
                        "ignore_total_area",
                        "ignore_area_ratio",
                        "ignore_total_perimeter",
                    ],
                )
                writer_slide.writeheader()
                for row in slide_rows:
                    writer_slide.writerow(row)
        print(
            "[TEST][DONE] Manifest 生成完成。",
            f"Total={total}",
            f"Pos={total_pos}",
            f"Neg={total_neg}",
            f"ExcludedByIgnore={total_excluded}",
            f"SampledVsGTMismatch={total_sampled_gt_mismatch}",
            f"Elapsed={overall_elapsed:.1f}s",
        )
    finally:
        if patch_f is not None:
            patch_f.close()


def build_manifest_from_config(config: dict) -> None:
    """从配置构建 test manifest，默认读取 prepare_test。"""
    data_cfg = config.get("prepare", config.get("data", {}))
    tissue_level = data_cfg.get("tissue_level")
    max_patches = data_cfg.get("max_patches_per_slide")
    sampling_mode = str(data_cfg.get("sampling_mode", "scan_order"))
    target_pos_ratio = data_cfg.get("target_pos_ratio")
    slide_labels_csv = data_cfg.get("slide_labels_csv")
    slide_label_mode = str(data_cfg.get("slide_label_mode", "xml_presence"))
    slide_label_fallback_to_sampled = bool(data_cfg.get("slide_label_fallback_to_sampled", False))
    mask_level = data_cfg.get("mask_level")
    mask_max_size = data_cfg.get("mask_max_size")
    tissue_max_size = data_cfg.get("tissue_max_size")
    ignore_overlap_threshold = float(data_cfg.get("ignore_overlap_threshold", 0.05))
    coords_out_dir = data_cfg.get("coords_out_dir", "data/processed/coords/test")
    slide_manifest_path = data_cfg.get(
        "slide_manifest_path", "data/processed/manifest_slides_test.csv"
    )
    write_patch_manifest = bool(data_cfg.get("write_patch_manifest", True))
    enable_geometry_stats = bool(data_cfg.get("enable_geometry_stats", True))
    build_manifest(
        slides_dir=Path(data_cfg["slides_dir"]),
        annotations_dir=(
            Path(data_cfg["annotations_dir"]) if data_cfg.get("annotations_dir") else None
        ),
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
        ignore_groups=parse_groups(data_cfg.get("ignore_groups", "Exclusion")),
        ignore_overlap_threshold=ignore_overlap_threshold,
        max_patches_per_slide=int(max_patches) if max_patches is not None else None,
        sampling_mode=sampling_mode,
        target_pos_ratio=float(target_pos_ratio) if target_pos_ratio is not None else None,
        slide_labels_csv=Path(slide_labels_csv) if slide_labels_csv else None,
        slide_label_mode=slide_label_mode,
        slide_label_fallback_to_sampled=slide_label_fallback_to_sampled,
        mask_suffix=str(data_cfg.get("mask_suffix", "_mask.tif")),
        prefer_masks=bool(data_cfg.get("prefer_masks", False)),
        progress_interval=int(data_cfg.get("progress_interval", 5000)),
        mask_level=int(mask_level) if mask_level is not None else None,
        mask_max_size=int(mask_max_size) if mask_max_size is not None else None,
        tissue_max_size=int(tissue_max_size) if tissue_max_size is not None else None,
        coords_out_dir=Path(coords_out_dir) if coords_out_dir else None,
        slide_manifest_path=Path(slide_manifest_path) if slide_manifest_path else None,
        write_patch_manifest=write_patch_manifest,
        enable_geometry_stats=enable_geometry_stats,
    )


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CAMELYON test split 的 patch manifest。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="YAML 配置路径（默认: configs/defaults.yaml）",
    )
    parser.add_argument("--slides-dir", type=Path)
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--mask-suffix", type=str, default="_mask.tif")
    parser.add_argument("--prefer-masks", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--mask-level", type=int)
    parser.add_argument("--mask-max-size", type=int)
    parser.add_argument("--coords-out-dir", type=Path)
    parser.add_argument(
        "--slide-manifest-path",
        type=Path,
        default=Path("data/processed/manifest_slides_test.csv"),
    )
    parser.add_argument("--write-patch-manifest", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tissue-level", type=int)
    parser.add_argument("--min-tissue", type=float, default=0.5)
    parser.add_argument("--tissue-max-size", type=int)
    parser.add_argument("--pos-threshold", type=float, default=0.5)
    parser.add_argument("--neg-threshold", type=float, default=0.0)
    parser.add_argument("--neg-keep-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=str, default="Tumor")
    parser.add_argument("--ignore-groups", type=str, default="Exclusion")
    parser.add_argument("--ignore-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--max-patches-per-slide", type=int)
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="scan_order",
        choices=("scan_order", "stratified_random"),
    )
    parser.add_argument("--target-pos-ratio", type=float)
    parser.add_argument("--slide-labels-csv", type=Path)
    parser.add_argument(
        "--slide-label-mode",
        type=str,
        default="xml_presence",
        choices=("xml_presence", "camelyon_prefix", "sampled", "none"),
    )
    parser.add_argument("--slide-label-fallback-to-sampled", action="store_true")
    parser.add_argument("--disable-geometry-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_config = args.config is not None and args.slides_dir is None and args.output_csv is None
    if use_config:
        config = load_yaml_config(args.config)
        prepare_cfg = config.get("prepare_test") or config.get("prepare")
        if prepare_cfg is None:
            raise ValueError("Missing prepare_test (or prepare) section in config.")
        build_manifest_from_config({"prepare": prepare_cfg})
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
        ignore_groups=parse_groups(args.ignore_groups),
        ignore_overlap_threshold=args.ignore_overlap_threshold,
        max_patches_per_slide=args.max_patches_per_slide,
        sampling_mode=args.sampling_mode,
        target_pos_ratio=args.target_pos_ratio,
        slide_labels_csv=args.slide_labels_csv,
        slide_label_mode=args.slide_label_mode,
        slide_label_fallback_to_sampled=args.slide_label_fallback_to_sampled,
        mask_suffix=args.mask_suffix,
        prefer_masks=args.prefer_masks,
        progress_interval=args.progress_interval,
        mask_level=args.mask_level,
        mask_max_size=args.mask_max_size,
        tissue_max_size=args.tissue_max_size,
        coords_out_dir=args.coords_out_dir,
        slide_manifest_path=args.slide_manifest_path,
        write_patch_manifest=args.write_patch_manifest,
        enable_geometry_stats=not args.disable_geometry_stats,
    )


if __name__ == "__main__":
    main()
