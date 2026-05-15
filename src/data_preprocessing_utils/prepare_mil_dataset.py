from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LabelRecord:
    image: str
    label: int

    @property
    def slide_id(self) -> str:
        return Path(self.image).stem


def _load_label_records(labels_json: Path) -> list[LabelRecord]:
    with open(labels_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {labels_json}, got {type(rows)!r}")

    out: list[LabelRecord] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} is not an object: {row!r}")
        image = row.get("image")
        label = row.get("label")
        if not isinstance(image, str):
            raise ValueError(f"Row {idx} has invalid image value: {image!r}")
        if not isinstance(label, int):
            raise ValueError(f"Row {idx} has non-integer label value: {label!r}")
        out.append(LabelRecord(image=image, label=label))
    return out


def _build_candidate_relpaths(image_name: str) -> list[Path]:
    image_path = Path(image_name)
    stem = image_path.stem
    candidates = [
        image_path,
        Path(stem),
        Path(stem) / "Slidedat.ini",
    ]
    return candidates


def _resolve_slide_candidates(
    image_name: str, allowed_years: list[str]
) -> list[Path]:
    candidates: list[Path] = []
    for year in allowed_years:
        for suffix in _build_candidate_relpaths(image_name):
            candidates.append(Path(year) / suffix)
    # Keep order stable while removing duplicates.
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _glob_fallback(
    image_name: str, slides_root: Path, allowed_years: list[str]
) -> list[Path]:
    """Recursively search year subdirs for stem matches.

    Handles non-standard layouts where slides are nested deeper than
    ``<year>/<slide>`` (e.g. ``2025/2025_01/DS_A01R_01S.mrxs``).
    Returns relative paths from ``slides_root``.
    """
    stem = Path(image_name).stem
    hits: list[Path] = []
    for year in allowed_years:
        year_dir = slides_root / year
        if not year_dir.is_dir():
            continue
        # Prefer actual .mrxs files; fall back to directories with matching stem.
        for hit in sorted(year_dir.rglob(f"{stem}.mrxs")):
            hits.append(hit.relative_to(slides_root))
        if not hits:
            for hit in sorted(year_dir.rglob(stem)):
                if hit.is_dir():
                    hits.append(hit.relative_to(slides_root))
    return hits


def _resolve_records(
    records: list[LabelRecord], slides_root: Path, allowed_years: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for rec in records:
        rel_candidates = _resolve_slide_candidates(rec.image, allowed_years)
        existing: list[Path] = []
        for rel in rel_candidates:
            abs_path = slides_root / rel
            if abs_path.exists():
                existing.append(rel)

        # Fallback: recursive glob handles sub-subfolder layouts
        # (e.g. 2025/2025_01/slide.mrxs) missed by the fixed candidate list.
        if not existing:
            existing = _glob_fallback(rec.image, slides_root, allowed_years)

        existing = _select_unique_match(existing)

        if len(existing) == 0:
            missing.append(
                {
                    "image": rec.image,
                    "label": rec.label,
                    "slide_id": rec.slide_id,
                    "checked_candidates": [str(x) for x in rel_candidates],
                }
            )
            continue
        if len(existing) > 1:
            ambiguous.append(
                {
                    "image": rec.image,
                    "label": rec.label,
                    "slide_id": rec.slide_id,
                    "matches": [str(x) for x in existing],
                }
            )
            continue

        matched = existing[0]
        prepared.append(
            {
                "image": str(matched),
                "label": rec.label,
                "source_image": rec.image,
                "slide_id": rec.slide_id,
            }
        )
    return prepared, missing, ambiguous


def _match_priority(rel_path: Path) -> tuple[int, int]:
    """
    Prefer true slide files first, then slide directories, then metadata files.
    """
    suffix = rel_path.suffix.lower()
    name = rel_path.name.lower()
    if suffix == ".mrxs":
        return (0, len(rel_path.parts))
    if name == "slidedat.ini":
        return (2, len(rel_path.parts))
    return (1, len(rel_path.parts))


def _select_unique_match(existing: list[Path]) -> list[Path]:
    if not existing:
        return []
    # Keep one representative per logical slide stem, then apply path-type priority.
    by_stem: dict[tuple[str, str], list[Path]] = {}
    for rel in existing:
        year = rel.parts[0] if rel.parts else ""
        if rel.name.lower() == "slidedat.ini":
            stem = rel.parent.name
        else:
            stem = rel.stem
        key = (year, stem)
        by_stem.setdefault(key, []).append(rel)

    winners: list[Path] = []
    for _, paths in by_stem.items():
        winners.append(sorted(paths, key=_match_priority)[0])
    return sorted(winners)


def _build_derived_config(
    baseline_config: Path, prepared_labels_json: Path, slides_root: Path, output_dir: Path
) -> dict[str, Any]:
    with open(baseline_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["slides_dir"] = str(slides_root)
    cfg["labels_json"] = str(prepared_labels_json)
    cfg.setdefault(
        "dataset_preprocessing",
        {
            "mode": "manifest_only",
            "note": "Generated by data_preprocessing_utils.prepare_mil_dataset",
        },
    )
    cfg["dataset_preprocessing"]["output_dir"] = str(output_dir)
    return cfg


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare manifest-only MIL dataset from mounted NFS data using labels.json."
        )
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=Path(".mnt/nfs-data/labels.json"),
        help="Path to source labels JSON.",
    )
    parser.add_argument(
        "--slides-root",
        type=Path,
        default=Path(".mnt/nfs-data"),
        help="Root containing year folders (2023/2024/2025).",
    )
    parser.add_argument(
        "--allowed-years",
        nargs="+",
        default=["2023", "2024", "2025"],
        help="Year subfolders that are allowed for slide lookup.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Baseline YAML config to copy/update for prepared dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared_mil_dataset"),
        help="Output folder for prepared labels/report/config.",
    )
    parser.add_argument(
        "--prepared-labels-filename",
        default="labels_prepared.json",
        help="File name for generated labels JSON inside output dir.",
    )
    parser.add_argument(
        "--report-filename",
        default="prepare_report.json",
        help="File name for generated summary report inside output dir.",
    )
    parser.add_argument(
        "--derived-config-filename",
        default="baseline_prepared.yaml",
        help="File name for generated config YAML inside output dir.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if missing or ambiguous records are found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_label_records(args.labels_json)
    prepared, missing, ambiguous = _resolve_records(
        records=records,
        slides_root=args.slides_root,
        allowed_years=args.allowed_years,
    )

    output_dir = args.output_dir
    prepared_labels_json = output_dir / args.prepared_labels_filename
    report_json = output_dir / args.report_filename
    derived_config_yaml = output_dir / args.derived_config_filename

    prepared_rows = [{"image": row["image"], "label": row["label"]} for row in prepared]
    _write_json(prepared_labels_json, prepared_rows)

    report = {
        "input_labels_json": str(args.labels_json),
        "slides_root": str(args.slides_root),
        "allowed_years": list(args.allowed_years),
        "baseline_config": str(args.baseline_config),
        "output_dir": str(output_dir),
        "n_input_labels": len(records),
        "n_prepared": len(prepared),
        "n_missing": len(missing),
        "n_ambiguous": len(ambiguous),
        "prepared": prepared,
        "missing": missing,
        "ambiguous": ambiguous,
    }
    _write_json(report_json, report)

    derived_cfg = _build_derived_config(
        baseline_config=args.baseline_config,
        prepared_labels_json=prepared_labels_json,
        slides_root=args.slides_root,
        output_dir=output_dir,
    )
    _write_yaml(derived_config_yaml, derived_cfg)

    print(
        (
            f"Prepared labels: {len(prepared)}/{len(records)} "
            f"(missing={len(missing)}, ambiguous={len(ambiguous)})"
        )
    )
    print(f"labels_json: {prepared_labels_json}")
    print(f"report:      {report_json}")
    print(f"config:      {derived_config_yaml}")

    if args.strict and (missing or ambiguous):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

