#!/usr/bin/env python3
"""CLI wrapper for publication artifact export.

Usage
-----
python scripts/export_publication_artifacts.py \\
    --results-dir results_prepared \\
    [--config configs/baseline_prepared.yaml] \\
    [--run-id my_run]

Reads ``cv_summary.json`` and ``fold_*/test_predictions.pt`` from ``results-dir``
and writes a self-contained ``publication_outputs/<model>_<run_id>/`` folder with
PDF confusion matrices, ROC curves, LaTeX + CSV tables, and a README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run directly (outside venv activation)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from mil_baseline.export import export_publication_artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="Export publication-ready figures and tables.")
    ap.add_argument(
        "--results-dir",
        required=True,
        help="Path to the results directory containing cv_summary.json and fold_*/ subdirs.",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Optional path to the YAML config used for training (recorded in outputs).",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Short string appended to the output folder name (default: timestamp).",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (_repo_root / results_dir).resolve()

    config_path = Path(args.config).resolve() if args.config else None

    out_dir = export_publication_artifacts(
        results_dir=results_dir,
        config_path=config_path,
        run_id=args.run_id,
    )
    print(f"Done. Artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
