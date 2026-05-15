from __future__ import annotations

import argparse
from pathlib import Path

from mil_baseline.cv import run_train_cv
from mil_baseline.feature_extract import run_extract


def _cfg_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def main_extract() -> None:
    ap = argparse.ArgumentParser(
        description="Extract frozen patch features for each WSI."
    )
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_extract(_cfg_path(args.config), overwrite=args.overwrite)


def main_train_cv() -> None:
    ap = argparse.ArgumentParser(
        description="Patient-level grouped K-fold MIL training and evaluation."
    )
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    ap.add_argument(
        "--export-publication",
        action="store_true",
        help="After CV, generate publication_outputs/ with PDF figures and LaTeX tables.",
    )
    args = ap.parse_args()
    out = run_train_cv(_cfg_path(args.config), export_publication=args.export_publication)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MIL baseline CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="Patch + encode slides → data/features/*.pt")
    p_ex.add_argument("--config", type=str, default="configs/baseline.yaml")
    p_ex.add_argument("--overwrite", action="store_true")

    p_tr = sub.add_parser(
        "train-cv",
        help="Patient-level grouped K-fold training + pooled metrics JSON",
    )
    p_tr.add_argument("--config", type=str, default="configs/baseline.yaml")
    p_tr.add_argument(
        "--export-publication",
        action="store_true",
        help="After CV, write publication_outputs/ (PDF confusion matrices, LaTeX tables, …).",
    )

    args = ap.parse_args()
    cfg = _cfg_path(args.config)
    if args.cmd == "extract":
        run_extract(cfg, overwrite=args.overwrite)
    elif args.cmd == "train-cv":
        export = getattr(args, "export_publication", False)
        out = run_train_cv(cfg, export_publication=export)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
