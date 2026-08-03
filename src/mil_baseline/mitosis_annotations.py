"""Parse QuPath GeoJSON structure-level annotations (mitotic figures + Nottingham
A/B/C component regions) produced by extract_mitosis_annotations.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MITOSIS_CLASS_NAMES = {"mitoza", "mitóza"}


@dataclass
class Polygon:
    points: np.ndarray  # (N, 2) float64, level-0 pixel coordinates
    name: str
    metadata: dict

    @property
    def centroid(self) -> tuple[float, float]:
        pts = self.points
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )


@dataclass
class SlideAnnotation:
    slide_id: str
    mitoses: list[Polygon]
    field_polygon: Polygon | None
    component_scores: dict[str, int | float | None]
    extra: list[Polygon] = field(default_factory=list)


def _polygon_from_feature(f: dict) -> Polygon:
    geom = f["geometry"]
    if geom["type"] != "Polygon":
        raise ValueError(f"Expected Polygon geometry, got {geom['type']!r}")
    ring = geom["coordinates"][0]
    pts = np.array(ring, dtype=np.float64)
    props = f.get("properties", {})
    name = props.get("classification", {}).get("name", "<none>")
    meta = props.get("metadata", {})
    return Polygon(points=pts, name=name, metadata=meta)


def _parse_first_int(text: str) -> int | None:
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


def _nhg_grade_from_score(score: int | None) -> int | None:
    if score is None:
        return None
    if 3 <= score <= 5:
        return 1
    if 6 <= score <= 7:
        return 2
    if 8 <= score <= 9:
        return 3
    return None


def parse_slide_annotation(geojson_path: Path) -> SlideAnnotation:
    slide_id = geojson_path.stem
    with open(geojson_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    mitoses: list[Polygon] = []
    field_polygon: Polygon | None = None
    extra: list[Polygon] = []
    raw_component_text: dict[str, str] = {}

    for f in data.get("features", []):
        poly = _polygon_from_feature(f)
        if poly.name in MITOSIS_CLASS_NAMES:
            mitoses.append(poly)
        elif poly.name.startswith("(A)"):
            raw_component_text["A"] = poly.metadata.get("ANNOTATION_DESCRIPTION", "")
        elif poly.name.startswith("(B)"):
            raw_component_text["B"] = poly.metadata.get("ANNOTATION_DESCRIPTION", "")
        elif poly.name.startswith("(C)"):
            raw_component_text["C"] = poly.metadata.get("ANNOTATION_DESCRIPTION", "")
            field_polygon = poly
        elif "Nottingham" in poly.name:
            raw_component_text["tot"] = poly.metadata.get("ANNOTATION_DESCRIPTION", "")
        else:
            extra.append(poly)

    a = _parse_first_int(raw_component_text.get("A", ""))
    b = _parse_first_int(raw_component_text.get("B", ""))
    c = _parse_first_int(raw_component_text.get("C", ""))
    tot_text = raw_component_text.get("tot", "")
    tot_m = re.search(r"skore\s*(\d+)", tot_text)
    grade_m = re.search(r"grade\s*(\d+)", tot_text)
    sum_abc = (a + b + c) if all(v is not None for v in (a, b, c)) else None

    component_scores = {
        "A": a,
        "B": b,
        "C": c,
        "sum_abc": sum_abc,
        "total_annotated": int(tot_m.group(1)) if tot_m else None,
        "grade_annotated": int(grade_m.group(1)) if grade_m else None,
        "grade_from_sum": _nhg_grade_from_score(sum_abc),
    }

    return SlideAnnotation(
        slide_id=slide_id,
        mitoses=mitoses,
        field_polygon=field_polygon,
        component_scores=component_scores,
        extra=extra,
    )


def load_all_annotations(annotations_dir: Path) -> dict[str, SlideAnnotation]:
    out: dict[str, SlideAnnotation] = {}
    for p in sorted(Path(annotations_dir).glob("*.geojson")):
        out[p.stem] = parse_slide_annotation(p)
    return out
