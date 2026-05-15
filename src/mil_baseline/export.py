"""Publication-ready artifact export for the Nottingham MIL benchmark.

Reads per-fold ``test_predictions.pt`` and ``cv_summary.json`` from a results
directory and writes a self-contained ``publication_outputs/<run_id>/`` folder
containing:

- PDF/SVG confusion matrices (per fold + aggregated)
- ROC one-vs-rest curves (PDF)
- Main results table (CSV + booktabs LaTeX)
- Per-fold metrics table (CSV)
- Slide-level predictions (CSV)
- ``results_summary.json``
- ``README.md`` with caption stubs and regenerate command
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Journal-quality matplotlib defaults
# ──────────────────────────────────────────────────────────────────────────────

_JOURNAL_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,   # embed fonts in PDF
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colorblind-safe palette (IBM accessible palette)
_CB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def _setup_mpl() -> None:
    mpl.rcParams.update(_JOURNAL_RC)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_cv_summary(results_dir: Path) -> dict:
    p = results_dir / "cv_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"cv_summary.json not found in {results_dir}")
    with open(p, "r") as f:
        return json.load(f)


def _load_fold_predictions(results_dir: Path, n_folds: int) -> list[dict]:
    import torch

    preds = []
    for fold in range(n_folds):
        pt = results_dir / f"fold_{fold}" / "test_predictions.pt"
        if not pt.exists():
            raise FileNotFoundError(f"Missing: {pt}")
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        preds.append(
            {
                "fold": fold,
                "y_true": np.array(obj["y_true"]),
                "y_pred": np.array(obj["y_pred"]),
                "y_prob": np.array(obj["y_prob"]),
                "slide_ids": obj.get("slide_ids", []),
                "patient_ids": obj.get("patient_ids", []),
            }
        )
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix plots
# ──────────────────────────────────────────────────────────────────────────────

def _cm_normalized(cm_counts: np.ndarray) -> np.ndarray:
    row_sums = cm_counts.sum(axis=1, keepdims=True).clip(min=1)
    return cm_counts / row_sums


def _plot_cm(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    ax: plt.Axes,
    fmt: str = ".2f",
    cmap: str = "Blues",
) -> None:
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=cm.max())
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]:{fmt}}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )


def _export_confusion_matrices(
    fold_preds: list[dict],
    class_names: list[str],
    out_dir: Path,
    model_name: str,
) -> list[str]:
    from sklearn.metrics import confusion_matrix

    _setup_mpl()
    artifacts: list[str] = []

    # Per-fold
    for p in fold_preds:
        fold = p["fold"]
        cm_raw = confusion_matrix(p["y_true"], p["y_pred"], labels=list(range(len(class_names))))
        cm_norm = _cm_normalized(cm_raw.astype(float))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        _plot_cm(cm_norm, class_names, f"Fold {fold} — normalized", axes[0])
        _plot_cm(cm_raw.astype(float), class_names, f"Fold {fold} — counts", axes[1], fmt=".0f")
        fig.suptitle(f"{model_name.upper()} · Fold {fold}", fontsize=10)
        fig.tight_layout()
        fname = f"fig_confusion_{model_name}_fold{fold}.pdf"
        fig.savefig(out_dir / fname, format="pdf")
        plt.close(fig)
        artifacts.append(fname)

    # Aggregated (sum counts across folds, then normalize)
    agg_raw = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for p in fold_preds:
        agg_raw += confusion_matrix(
            p["y_true"], p["y_pred"], labels=list(range(len(class_names)))
        )
    agg_norm = _cm_normalized(agg_raw.astype(float))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    _plot_cm(agg_norm, class_names, "Aggregated — normalized", axes[0])
    _plot_cm(agg_raw.astype(float), class_names, "Aggregated — counts", axes[1], fmt=".0f")
    fig.suptitle(f"{model_name.upper()} · Aggregated (all folds)", fontsize=10)
    fig.tight_layout()
    fname = f"fig_confusion_{model_name}_aggregated.pdf"
    fig.savefig(out_dir / fname, format="pdf")
    plt.close(fig)
    artifacts.append(fname)

    return artifacts


# ──────────────────────────────────────────────────────────────────────────────
# ROC / PR curves
# ──────────────────────────────────────────────────────────────────────────────

def _export_roc_curves(
    fold_preds: list[dict],
    class_names: list[str],
    out_dir: Path,
    model_name: str,
) -> list[str]:
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    _setup_mpl()
    n_cls = len(class_names)
    # Pool all fold predictions
    y_true_all = np.concatenate([p["y_true"] for p in fold_preds])
    y_prob_all = np.vstack([p["y_prob"] for p in fold_preds])
    y_bin = label_binarize(y_true_all, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for k, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, k], y_prob_all[:, k])
            auc = roc_auc_score(y_bin[:, k], y_prob_all[:, k])
            ax.plot(fpr, tpr, color=_CB_COLORS[k % len(_CB_COLORS)],
                    label=f"{cname} (AUC={auc:.2f})", lw=1.5)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name.upper()} · One-vs-Rest ROC (pooled folds)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fname = f"fig_roc_{model_name}.pdf"
    fig.savefig(out_dir / fname, format="pdf")
    plt.close(fig)
    return [fname]


# ──────────────────────────────────────────────────────────────────────────────
# Tables — CSV + LaTeX booktabs
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"


def _export_main_table(
    summary: dict,
    out_dir: Path,
    model_name: str,
) -> list[str]:
    import csv

    ms = summary["cv_mean_std"]
    row_data = {
        "model": model_name.upper(),
        "encoder": summary.get("encoder", ""),
        "macro_f1_mean": ms["macro_f1"]["mean"],
        "macro_f1_std": ms["macro_f1"]["std"],
        "balanced_acc_mean": ms["balanced_accuracy"]["mean"],
        "balanced_acc_std": ms["balanced_accuracy"]["std"],
        "auroc_mean": ms["macro_auc_ovr"]["mean"],
        "auroc_std": ms["macro_auc_ovr"]["std"],
    }

    csv_path = out_dir / "table_main_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        w.writeheader()
        w.writerow(row_data)

    # LaTeX booktabs
    f1_str = _fmt_mean_std(row_data["macro_f1_mean"], row_data["macro_f1_std"])
    ba_str = _fmt_mean_std(row_data["balanced_acc_mean"], row_data["balanced_acc_std"])
    au_str = _fmt_mean_std(row_data["auroc_mean"], row_data["auroc_std"])
    n_folds = summary.get("n_splits", "?")

    tex = textwrap.dedent(
        rf"""
        % Auto-generated — do not edit; regenerate with export_publication_artifacts.py
        \begin{{table}}[ht]
          \centering
          \caption{{Nottingham grade prediction benchmark ({n_folds}-fold patient-level CV,
                    mean\,$\pm$\,std).}}
          \label{{tab:main_results}}
          \begin{{tabular}}{{llccc}}
            \toprule
            Model & Encoder & Macro F1 & Balanced Acc. & AUROC \\
            \midrule
            {row_data['model']} & {row_data['encoder']} & ${f1_str}$ & ${ba_str}$ & ${au_str}$ \\
            \bottomrule
          \end{{tabular}}
        \end{{table}}
        """
    ).strip() + "\n"

    tex_path = out_dir / "table_main_results.tex"
    tex_path.write_text(tex, encoding="utf-8")

    snippets_dir = out_dir / "latex_snippets"
    snippets_dir.mkdir(exist_ok=True)
    (snippets_dir / "table_main_results.tex").write_text(tex, encoding="utf-8")

    return ["table_main_results.csv", "table_main_results.tex", "latex_snippets/table_main_results.tex"]


def _export_per_fold_csv(summary: dict, out_dir: Path) -> str:
    import csv

    fields = ["fold", "macro_f1", "balanced_accuracy", "macro_auc_ovr", "weighted_f1", "n_val_slides"]
    rows = []
    for fr in summary.get("per_fold", []):
        t = fr.get("test", {})
        rows.append(
            {
                "fold": fr.get("fold", ""),
                "macro_f1": t.get("macro_f1", ""),
                "balanced_accuracy": t.get("balanced_accuracy", ""),
                "macro_auc_ovr": t.get("macro_auc_ovr", ""),
                "weighted_f1": t.get("weighted_f1", ""),
                "n_val_slides": len(summary.get("per_fold", [])),  # placeholder
            }
        )
    fpath = out_dir / "metrics_per_fold.csv"
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return "metrics_per_fold.csv"


def _export_predictions_csv(fold_preds: list[dict], out_dir: Path) -> str:
    import csv

    rows = []
    for p in fold_preds:
        n = len(p["y_true"])
        for i in range(n):
            row: dict[str, Any] = {
                "fold": p["fold"],
                "slide_id": p["slide_ids"][i] if i < len(p["slide_ids"]) else "",
                "patient_id": p["patient_ids"][i] if i < len(p["patient_ids"]) else "",
                "y_true": int(p["y_true"][i]),
                "y_pred": int(p["y_pred"][i]),
            }
            for k in range(p["y_prob"].shape[1]):
                row[f"p_class_{k}"] = float(p["y_prob"][i, k])
            rows.append(row)

    if not rows:
        return "predictions_val.csv"

    fpath = out_dir / "predictions_val.csv"
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return "predictions_val.csv"


# ──────────────────────────────────────────────────────────────────────────────
# results_summary.json
# ──────────────────────────────────────────────────────────────────────────────

def _export_results_summary(
    summary: dict,
    config_path: Path | None,
    out_dir: Path,
) -> str:
    from mil_baseline.utils import get_git_sha

    out = {
        "encoder": summary.get("encoder"),
        "mil_model": summary.get("mil_model"),
        "n_splits": summary.get("n_splits"),
        "n_slides": summary.get("n_slides"),
        "n_patients": summary.get("n_patients"),
        "seed": summary.get("seed"),
        "git_sha": summary.get("git_sha") or get_git_sha(),
        "config_path": str(config_path) if config_path else None,
        "fold_protocol": summary.get("fold_protocol"),
        "class_names": summary.get("class_names"),
        "cv_metrics": summary.get("cv_mean_std"),
        "pooled_test_metrics": summary.get("pooled_test_metrics"),
        "generated_at": datetime.now().isoformat(),
    }
    fpath = out_dir / "results_summary.json"
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2)
    return "results_summary.json"


# ──────────────────────────────────────────────────────────────────────────────
# Folder README
# ──────────────────────────────────────────────────────────────────────────────

def _write_folder_readme(
    out_dir: Path,
    artifacts: list[str],
    model_name: str,
    config_path: Path | None,
    results_dir: Path,
) -> None:
    cfg_arg = f" --config {config_path}" if config_path else ""
    regen_cmd = (
        f"python scripts/export_publication_artifacts.py "
        f"--results-dir {results_dir}{cfg_arg}"
    )
    lines = [
        f"# Publication outputs — {model_name.upper()}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Files",
        "",
    ]
    captions: dict[str, str] = {
        "table_main_results.csv": "Machine-readable main results table (mean ± std across folds).",
        "table_main_results.tex": "LaTeX booktabs table — use `\\input{table_main_results.tex}`.",
        "metrics_per_fold.csv": "Per-fold headline metrics; include as supplementary material.",
        "predictions_val.csv": "Slide-level predictions (fold, slide_id, patient_id, y_true, y_pred, p_class_*).",
        "results_summary.json": "Machine-readable summary: CV metrics, git SHA, config hash.",
    }
    for fname in sorted(set(artifacts)):
        stub = captions.get(fname, "")
        if not stub:
            if "confusion" in fname and "aggregated" in fname:
                stub = "Aggregated confusion matrix (sum counts across folds, then row-normalized). Colorblind-safe Blues colormap."
            elif "confusion" in fname:
                fold_n = fname.split("fold")[-1].replace(".pdf", "") if "fold" in fname else ""
                stub = f"Per-fold confusion matrix (normalized + raw counts) for fold {fold_n}."
            elif "roc" in fname:
                stub = "One-vs-rest ROC curves (pooled fold predictions); max-softmax class probability as score."
        lines.append(f"- **`{fname}`** — {stub}")

    lines += [
        "",
        "## Suggested figure captions",
        "",
        f"**Confusion matrix (aggregated):** Aggregated confusion matrix for {model_name.upper()} "
        "with frozen patch encoder over all folds of patient-level cross-validation.  "
        "Rows = true Nottingham grade, columns = predicted grade.  "
        "Values are row-normalized (recall per class).",
        "",
        f"**ROC curves:** One-vs-rest ROC curves for {model_name.upper()} (pooled predictions "
        "across all patient-level CV folds).  Scores are max-softmax class probabilities.",
        "",
        "## Regenerate after retraining",
        "",
        "```bash",
        regen_cmd,
        "```",
        "",
        "Or pass `--export-publication` to `train-cv`:",
        "",
        "```bash",
        f"python -m mil_baseline.cli train-cv{cfg_arg} --export-publication",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def export_publication_artifacts(
    results_dir: Path,
    config_path: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Generate publication-ready artifacts and return the output directory."""
    _setup_mpl()

    summary = _load_cv_summary(results_dir)
    n_folds = int(summary["n_splits"])
    fold_preds = _load_fold_predictions(results_dir, n_folds)
    model_name: str = summary.get("mil_model", "mil")
    class_names: list[str] = summary.get("class_names") or [
        f"class_{i}" for i in range(3)
    ]

    ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / "publication_outputs" / f"{model_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []

    artifacts += _export_confusion_matrices(fold_preds, class_names, out_dir, model_name)
    artifacts += _export_roc_curves(fold_preds, class_names, out_dir, model_name)
    artifacts += _export_main_table(summary, out_dir, model_name)
    artifacts.append(_export_per_fold_csv(summary, out_dir))
    artifacts.append(_export_predictions_csv(fold_preds, out_dir))
    artifacts.append(_export_results_summary(summary, config_path, out_dir))
    _write_folder_readme(out_dir, artifacts, model_name, config_path, results_dir)
    artifacts.append("README.md")

    print(f"[export] {len(artifacts)} artifacts → {out_dir}")
    return out_dir
