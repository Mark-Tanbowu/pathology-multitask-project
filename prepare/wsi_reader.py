from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from PIL import Image

try:
    from openslide import OpenSlide
except Exception:  # pragma: no cover - optional dependency
    OpenSlide = None  # type: ignore[assignment]


class SlideReader:
    """统一的 WSI 读取封装（仅 OpenSlide）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._slide = None
        self._level_downsamples: list[float] = [1.0]
        self._level_dimensions: list[Tuple[int, int]] = []

        if OpenSlide is None:
            msg = (
                "OpenSlide 不可用（未安装或系统库缺失）。"
                "请安装 openslide-python 并确保系统已安装 openslide-tools/libopenslide。"
            )
            logging.getLogger(__name__).error(msg)
            raise RuntimeError(msg)

        try:
            self._slide = OpenSlide(str(self.path))
            self._level_downsamples = list(self._slide.level_downsamples)
            self._level_dimensions = list(self._slide.level_dimensions)
        except Exception as exc:
            msg = f"OpenSlide 打开失败：{self.path}. 原因：{type(exc).__name__}: {exc}"
            logging.getLogger(__name__).error(msg)
            raise RuntimeError(msg) from exc

    @property
    def level_downsamples(self) -> list[float]:
        return self._level_downsamples

    @property
    def level_dimensions(self) -> list[Tuple[int, int]]:
        return self._level_dimensions

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image.Image:
        """读取指定 level 的局部区域。"""
        if self._slide is None:
            raise RuntimeError("OpenSlide 未初始化，无法读取 WSI。")
        return self._slide.read_region(location, level, size).convert("RGB")


    def get_best_level_for_downsample(self, target_downsample: float) -> int:
        """根据目标下采样倍率选择最接近的 level。"""
        candidates = range(len(self._level_downsamples))
        return min(candidates, key=lambda i: abs(self._level_downsamples[i] - target_downsample))

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()

    def __enter__(self) -> "SlideReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
