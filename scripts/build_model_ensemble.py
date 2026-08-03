"""Build a multi-model + multi-seed ensemble from publication results dirs.

For each slide, average softmax probabilities across ALL provided models AND all seeds,
then compute metrics on the consensus prediction set.

Usage:
    python scripts/build_model_ensemble.py \\
        --results-dir results_publication_meanlinear \\
        --results-dir results_publication_maxpool \\
        --results-dir results_publication_abmil
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mil_baseline.metrics import compute_metrics  # noqa: E402
from mil_baseline.stats import permutation_test_macro_f1  # noqa: E402


def _load_all_fold_predictions(results_dirs: list[Path]) -> dict[str, list[np.ndarray]]:
    """Collect per-slide softmax probabilities across all models and all seeds.

    Returns dict: slide_id → list of probability vectors (one per (model, seed, fold) it appeared in).
    """
    slide_probs: dict[str, list[np.ndarray]] = {}
    slide_true: dict[str, int] = {}

    for rd in results_dirs:
        s = json.load(open(rd / "cv_summary.json"))
        n_seeds = s.get("n_seeds", 1)
        n_splits = s["n_splits"]
        for seed_idx in range(n_seeds):
            for fold in range(n_splits):
                if n_seeds > 1:
                    pt = rd / f"seed_{seed_idx}" / f"fold_{fold}" / "test_predictions.pt"
                else:
                    pt = rd / f"fold_{fold}" / "test_predictions.pt"
                if not pt.exists():
                    continue
                obj = torch.load(pt, map_location="cpu", weights_only=False)
                sids = obj.get("slide_ids", [])
                y_true = np.asarray(obj["y_true"])
                y_prob = np.asarray(obj["y_prob"])
                for i, sid in enumerate(sids):
                    slide_probs.setdefault(sid, []).append(y_prob[i])
                    slide_true[sid] = int(y_true[i])
    return slide_probs, slide_true


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", action="append", required=True)
    ap.add_argument("--out", type=str, default="publication_outputs/model_ensemble.json")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.results_dir]
    print(f"Ensembling: {[d.name for d in dirs]}")

    slide_probs, slide_true = _load_all_fold_predictions(dirs)

    sids_sorted = sorted(slide_probs.keys())
    y_true = np.array([slide_true[s] for s in sids_sorted], dtype=np.int64)
    y_prob = np.vstack([np.mean(np.vstack(slide_probs[s]), axis=0) for s in sids_sorted])
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"Slides: {len(sids_sorted)}  predictions per slide (avg): {np.mean([len(slide_probs[s]) for s in sids_sorted]):.1f}")
    print()
    print(f"=== Model + seed ensemble ({len(dirs)} models) ===")
    print(f"  macro F1        {metrics['macro_f1']:.3f}")
    print(f"  balanced acc    {metrics['balanced_accuracy']:.3f}")
    print(f"  macro AUROC     {metrics['macro_auc_ovr']:.3f}")
    print(f"  weighted F1     {metrics['weighted_f1']:.3f}")

    # Permutation test
    try:
        _, perm_p = permutation_test_macro_f1(y_true, y_pred, n_perm=5000, seed=42)
        print(f"  permutation p   {perm_p:.4f}")
    except Exception:
        perm_p = None

    # Per-class F1
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    print()
    print("=== Per-class ===")
    for i in range(len(p)):
        print(f"  Grade {i+1}: precision={p[i]:.3f}  recall={r[i]:.3f}  F1={f[i]:.3f}  n={supp[i]}")

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "models": [d.name for d in dirs],
        "n_slides": len(sids_sorted),
        "metrics": metrics,
        "perm_p_macro_f1": perm_p,
        "per_class": {
            "precision": p.tolist(), "recall": r.tolist(),
            "f1": f.tolist(), "support": supp.tolist(),
        },
    }, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
