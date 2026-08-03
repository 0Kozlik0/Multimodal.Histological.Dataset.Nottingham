"""Build a publication-ready benchmark table across multiple model results dirs.

Usage:
    python scripts/build_benchmark_table.py \\
        --results-dir results_publication_meanlinear:MeanLinear \\
        --results-dir results_publication_abmil:ABMIL \\
        --results-dir results_publication_clam_sb:CLAM-SB \\
        --results-dir results_publication_maxpool:MaxPool \\
        --linear-probe results_prepared_linear/cv_summary_linear_probe.json:ResNet50 \\
        --linear-probe results_pub_linear_uni/cv_summary_linear_probe.json:UNI \\
        --out publication_outputs/benchmark_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mil_baseline.stats import bootstrap_ci, permutation_test_macro_f1  # noqa: E402


def _load_per_fold(results_dir: Path) -> list[dict]:
    s = json.load(open(results_dir / "cv_summary.json"))
    return s


def _compute_row(name: str, summary: dict, results_dir: Path) -> dict:
    cv_ms = summary["cv_mean_std"]
    f1_vals = [fr["test"]["macro_f1"] for fr in summary["per_fold"]]
    bacc_vals = [fr["test"]["balanced_accuracy"] for fr in summary["per_fold"]]
    auc_vals = [fr["test"]["macro_auc_ovr"] for fr in summary["per_fold"]]
    f1_lo, f1_hi = bootstrap_ci(np.array(f1_vals), n_boot=1000, seed=42)
    bacc_lo, bacc_hi = bootstrap_ci(np.array(bacc_vals), n_boot=1000, seed=42)

    # Pooled predictions for permutation test
    import torch
    n_seeds = summary.get("n_seeds", 1)
    n_splits = summary["n_splits"]
    y_true_pool, y_pred_pool = [], []
    for seed_idx in range(n_seeds):
        for fold in range(n_splits):
            if n_seeds > 1:
                pt = results_dir / f"seed_{seed_idx}" / f"fold_{fold}" / "test_predictions.pt"
            else:
                pt = results_dir / f"fold_{fold}" / "test_predictions.pt"
            if not pt.exists():
                continue
            obj = torch.load(pt, map_location="cpu", weights_only=False)
            y_true_pool.extend(obj["y_true"].tolist())
            y_pred_pool.extend(obj["y_pred"].tolist())
    perm_p: float | str = ""
    if y_true_pool:
        try:
            _, perm_p = permutation_test_macro_f1(
                np.array(y_true_pool), np.array(y_pred_pool), n_perm=1000, seed=42,
            )
        except Exception:
            perm_p = "—"

    n_collapsed = sum(
        1 for fr in summary["per_fold"]
        if fr.get("test_n_unique_pred", 0) < int(summary.get("num_classes", len(summary.get("class_names", []))))
    )
    collapse_rate = n_collapsed / len(summary["per_fold"]) if summary["per_fold"] else 0.0

    return {
        "model": name,
        "encoder": summary.get("encoder", ""),
        "n_folds": len(summary["per_fold"]),
        "f1_mean": cv_ms["macro_f1"]["mean"],
        "f1_std":  cv_ms["macro_f1"]["std"],
        "f1_ci_lo": f1_lo,
        "f1_ci_hi": f1_hi,
        "bacc_mean": cv_ms["balanced_accuracy"]["mean"],
        "bacc_std":  cv_ms["balanced_accuracy"]["std"],
        "bacc_ci_lo": bacc_lo,
        "bacc_ci_hi": bacc_hi,
        "auc_mean": cv_ms["macro_auc_ovr"]["mean"],
        "auc_std":  cv_ms["macro_auc_ovr"]["std"],
        "perm_p":   perm_p,
        "collapse_rate": collapse_rate,
        "ensembled": summary.get("ensembled_test_metrics"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir", action="append", default=[],
        help="path:display_name (repeatable)",
    )
    ap.add_argument(
        "--linear-probe", action="append", default=[],
        help="path[:display_label] — repeatable; display_label overrides encoder name",
    )
    ap.add_argument("--out", type=str, default="benchmark_table.md")
    args = ap.parse_args()

    rows: list[dict] = []
    for entry in args.results_dir:
        path, _, name = entry.partition(":")
        rd = Path(path)
        if not (rd / "cv_summary.json").exists():
            print(f"[skip] {rd}: no cv_summary.json")
            continue
        summary = _load_per_fold(rd)
        rows.append(_compute_row(name or rd.name, summary, rd))

    for lp_entry in args.linear_probe:
        lp_path, _, lp_label = lp_entry.partition(":")
        lp = json.load(open(lp_path))
        enc_name = lp_label or lp.get("encoder", Path(lp_path).parent.name)
        for mdl in ("logreg", "knn"):
            if mdl not in lp:
                continue
            cv_ms = lp[mdl]["cv_mean_std"]
            f1_vals = [fr["test"]["macro_f1"] for fr in lp[mdl]["per_fold"]]
            bacc_vals = [fr["test"]["balanced_accuracy"] for fr in lp[mdl]["per_fold"]]
            f1_lo, f1_hi = bootstrap_ci(np.array(f1_vals))
            bacc_lo, bacc_hi = bootstrap_ci(np.array(bacc_vals))
            rows.append({
                "model": f"{enc_name}+LinearProbe-{mdl}",
                "encoder": enc_name,
                "n_folds": len(lp[mdl]["per_fold"]),
                "f1_mean": cv_ms["macro_f1"]["mean"],
                "f1_std":  cv_ms["macro_f1"]["std"],
                "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
                "bacc_mean": cv_ms["balanced_accuracy"]["mean"],
                "bacc_std":  cv_ms["balanced_accuracy"]["std"],
                "bacc_ci_lo": bacc_lo, "bacc_ci_hi": bacc_hi,
                "auc_mean": cv_ms["macro_auc_ovr"]["mean"],
                "auc_std":  cv_ms["macro_auc_ovr"]["std"],
                "perm_p": "—",
                "collapse_rate": float("nan"),
                "ensembled": None,
            })

if __name__ == "__main__":
    main()
