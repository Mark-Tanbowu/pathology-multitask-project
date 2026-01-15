from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class PolygonAnnotation:
    """以 level-0 坐标保存的多边形标注。"""

    points: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    group: Optional[str] = None
    name: Optional[str] = None


def _points_to_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_asap_xml(
    xml_path: str | Path, include_groups: Optional[Iterable[str]] = None
) -> list[PolygonAnnotation]:
    """解析 ASAP 风格 XML 标注并返回多边形列表。"""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    include_set = {g.strip() for g in include_groups} if include_groups else None

    polygons: list[PolygonAnnotation] = []
    for annotation in root.findall(".//Annotation"):
        ann_type = (annotation.get("Type") or "").lower()
        if ann_type and ann_type != "polygon":
            continue

        group = annotation.get("PartOfGroup") or annotation.get("Group")
        if include_set is not None and group not in include_set:
            continue

        coords_node = annotation.find("Coordinates") or annotation.find(".//Coordinates")
        if coords_node is None:
            continue

        points: list[tuple[float, float]] = []
        for coord in coords_node.findall("Coordinate"):
            try:
                x = float(coord.get("X"))
                y = float(coord.get("Y"))
            except (TypeError, ValueError):
                continue
            points.append((x, y))

        if len(points) < 3:
            continue

        polygons.append(
            PolygonAnnotation(
                points=points,
                bbox=_points_to_bbox(points),
                group=group,
                name=annotation.get("Name"),
            )
        )

    return polygons
