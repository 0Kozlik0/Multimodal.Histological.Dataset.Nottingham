"""Extract QuPath GeoJSON structure-level annotations (mitotic figures + Nottingham
A/B/C component regions) from the mounted NFS annotation archive into a local,
reproducible, ASCII-named folder for the mitosis ROI classifier pipeline.

Usage:
    python -m src.data_preprocessing_utils.extract_mitosis_annotations \
        --zip-path path/to/annotations.zip \
        --output-dir path/to/output/folder
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def extract_annotations(zip_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_features_total = 0
    class_counts: dict[str, int] = {}
    written_files: list[str] = []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".geojson"):
                continue
            slide_id = Path(name).stem
            data = json.loads(z.read(name).decode("utf-8"))
            feats = data.get("features", [])
            for f in feats:
                cname = f.get("properties", {}).get("classification", {}).get("name", "<none>")
                class_counts[cname] = class_counts.get(cname, 0) + 1
            n_features_total += len(feats)

            out_path = output_dir / f"{slide_id}.geojson"
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=True)
            written_files.append(out_path.name)
            n_written += 1

    report = {
        "source_zip": str(zip_path),
        "output_dir": str(output_dir),
        "n_slides": n_written,
        "n_features_total": n_features_total,
        "class_counts": class_counts,
        "files": sorted(written_files),
    }
    with open(output_dir / "extraction_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(".mnt/nfs-data/2025/Anotacie_KMK_zaznacene250328.zip"),
        help="Path to the source annotation zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/mitosis_kmk"),
        help="Output folder for extracted GeoJSON files + extraction_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = extract_annotations(args.zip_path, args.output_dir)
    print(f"Extracted {report['n_slides']} slide annotation file(s) -> {args.output_dir}")
    print(f"Total features: {report['n_features_total']}")
    print("Class counts:")
    for cname, cnt in sorted(report["class_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {cname}: {cnt}")


if __name__ == "__main__":
    main()
