from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import openslide


def _downsample_at_level(slide: openslide.OpenSlide, level: int) -> float:
    return float(slide.level_downsamples[level])


def tissue_fraction_rgb(rgb: np.ndarray, s_min: float, v_min: float) -> float:
    """Approximate tissue occupancy using HSV thresholds on uint8 RGB."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (s >= s_min) & (v >= v_min)
    return float(mask.mean())


def collect_patch_locations(
    slide: openslide.OpenSlide,
    level: int,
    patch_size: int,
    stride: int,
    tissue_cfg: dict,
) -> list[tuple[int, int]]:
    """Return (x0, y0) in level-0 coordinates."""
    if level >= slide.level_count:
        raise ValueError(f"level={level} >= level_count={slide.level_count}")
    ds = _downsample_at_level(slide, level)
    w0, h0 = slide.level_dimensions[0]
    tile0_w = int(round(patch_size * ds))
    tile0_h = int(round(patch_size * ds))
    step0 = max(int(round(stride * ds)), 1)
    xs = list(range(0, max(w0 - tile0_w, 0) + 1, step0))
    ys = list(range(0, max(h0 - tile0_h, 0) + 1, step0))
    locs: list[tuple[int, int]] = []
    s_min = float(tissue_cfg.get("hsv_s_min", 0.05))
    v_min = float(tissue_cfg.get("hsv_v_min", 0.08))
    bg_max = float(tissue_cfg.get("max_background_frac", 0.9))
    min_tissue = 1.0 - bg_max
    for y0 in ys:
        for x0 in xs:
            tile = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert(
                "RGB"
            )
            arr = np.asarray(tile, dtype=np.uint8)
            tf = tissue_fraction_rgb(arr, s_min, v_min)
            if tf >= min_tissue:
                locs.append((x0, y0))
    return locs


@contextmanager
def open_slide(path: str | Path) -> Iterator[openslide.OpenSlide]:
    s = openslide.OpenSlide(str(path))
    try:
        yield s
    finally:
        s.close()
