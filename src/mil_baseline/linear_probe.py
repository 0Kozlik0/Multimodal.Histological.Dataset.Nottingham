"""Linear-probe baselines: Logistic Regression + k-NN on mean-pooled patch features.

Uses the same patient-level grouped CV as cv.py. No MIL training required — just
mean-pool each slide's patch embeddings into a single vector, then fit sklearn classifiers.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mil_baseline.cv import _audit_leakage, _make_patient_folds
from mil_baseline.manifest import load_labels_json
from mil_baseline.metrics import compute_metrics
from mil_baseline.utils import get_git_sha, load_yaml, repo_root, resolve_path, seed_everything


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _mean_pool_features(feats_dir: Path, slide_ids: list[str]) -> np.ndarray:
    """Load each slide's .pt file and mean-pool patches → (n_slides, feat_dim)."""
    rows = []
    for sid in slide_ids:
        obj = torch.load(feats_dir / f"{sid}.pt", map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            t = obj.get("features", next(iter(obj.values())))
        else:
            t = obj
        rows.append(t.float().mean(dim=0).numpy())
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Probe fitting and evaluation
# ---------------------------------------------------------------------------

def _fit_probe(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, dict]:
    """Fit LogisticRegression + k-NN; return metrics and predictions for each."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_fit)
    Xf, Xt = sc.transform(X_fit), sc.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        lr.fit(Xf, y_fit)
    lr_pred = lr.predict(Xt)
    lr_prob = lr.predict_proba(Xt)

    k = min(5, len(Xf))
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(Xf, y_fit)
    knn_pred = knn.predict(Xt)
    knn_prob = knn.predict_proba(Xt)

    return {
        "logreg": {
            "metrics": compute_metrics(y_test, lr_pred, lr_prob),
            "y_pred": lr_pred,
            "y_prob": lr_prob,
        },
        "knn": {
            "metrics": compute_metrics(y_test, knn_pred, knn_prob),
            "y_pred": knn_pred,
            "y_prob": knn_prob,
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_linear_probe(config_path: Path, export_publication: bool = False) -> Path:
    cfg = load_yaml(config_path)
    root = repo_root()
    labels_path = resolve_path(root, cfg["labels_json"])
    slides_dir  = resolve_path(root, cfg["slides_dir"])
    feats_dir   = resolve_path(root, cfg["features_dir"])

    base_results = Path(cfg["results_dir"])
    results_dir = resolve_path(
        root,
        cfg.get("results_dir_linear") or str(base_results.with_name(base_results.name + "_linear")),
    )

    raw_seed = cfg["train"].get("seed")
    seed = int(raw_seed) if raw_seed is not None else None
    seed_everything(seed)

    df = load_labels_json(labels_path, slides_dir, int(cfg["num_classes"]))
    # Linear probe only needs cached features — raw .mrxs files are not required

    missing = [sid for sid in df["slide_id"] if not (feats_dir / f"{sid}.pt").exists()]
    if missing:
        raise FileNotFoundError(
            "Feature files missing — run extract first.\nMissing: "
            + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        )

    slide_ids   = df["slide_id"].tolist()
    patient_ids = df["patient_id"].tolist()
    ys          = [int(x) for x in df["y"].tolist()]
    n_classes   = int(cfg["num_classes"])
    class_names: list[str] = cfg.get("class_names") or [f"class_{i}" for i in range(n_classes)]
    encoder_name = cfg["encoder"]["name"]

    n_splits  = int(cfg["cv"]["n_splits"])
    fold_seed = seed if seed is not None else 42
    folds, actual_n_splits = _make_patient_folds(slide_ids, patient_ids, ys, n_splits, fold_seed)

    wandb_cfg = cfg.get("wandb") or {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name: str = wandb_cfg.get("group") or f"{encoder_name}_linearprobe_{ts}"

    print(f"[linear-probe] Loading {len(slide_ids)} slide features from {feats_dir} …")
    X_all = _mean_pool_features(feats_dir, slide_ids)
    y_all = np.array(ys, dtype=np.int64)
    print(f"[linear-probe] Feature matrix: {X_all.shape}  |  {actual_n_splits}-fold patient-level CV")

    fold_rows: dict[str, list[dict]] = {"logreg": [], "knn": []}
    pooled:    dict[str, dict[str, list]] = {
        "logreg": {"true": [], "prob": []},
        "knn":    {"true": [], "prob": []},
    }

    for fold, (tr_idx, te_idx) in enumerate(folds):
        fit_pids = [patient_ids[i] for i in tr_idx]
        te_pids  = [patient_ids[i] for i in te_idx]
        # No inner val split for linear probes — leakage audit on fit/test only
        _audit_leakage(
            [slide_ids[i] for i in tr_idx], [],
            [slide_ids[i] for i in te_idx],
            fit_pids, [], te_pids, fold,
        )
        print(
            f"[Fold {fold}/{actual_n_splits - 1}]"
            f"  fit={len(tr_idx)} slides/{len(set(fit_pids))} patients"
            f"  test={len(te_idx)}/{len(set(te_pids))}"
        )

        probe_results = _fit_probe(
            X_all[tr_idx], y_all[tr_idx],
            X_all[te_idx], y_all[te_idx],
            fold_seed + fold,
        )

        for mdl in ("logreg", "knn"):
            m = probe_results[mdl]["metrics"]
            fold_rows[mdl].append({"fold": fold, "test": m})
            pooled[mdl]["true"].extend(y_all[te_idx].tolist())
            pooled[mdl]["prob"].append(probe_results[mdl]["y_prob"])

        print(
            f"  logreg: macro_f1={probe_results['logreg']['metrics']['macro_f1']:.3f}"
            f"  balanced_acc={probe_results['logreg']['metrics']['balanced_accuracy']:.3f}"
            f"  |  knn: macro_f1={probe_results['knn']['metrics']['macro_f1']:.3f}"
            f"  balanced_acc={probe_results['knn']['metrics']['balanced_accuracy']:.3f}"
        )
        _wandb_fold(wandb_cfg, group_name, fold, probe_results)

    # ── Aggregate across folds ─────────────────────────────────────────────
    summary: dict[str, Any] = {
        "n_slides": len(slide_ids),
        "n_patients": len(set(patient_ids)),
        "n_splits": actual_n_splits,
        "fold_protocol": f"{actual_n_splits}-fold patient-level StratifiedGroupKFold",
        "encoder": encoder_name,
        "seed": seed,
        "git_sha": get_git_sha(),
        "class_names": class_names,
    }

    for mdl in ("logreg", "knn"):
        p_true = np.array(pooled[mdl]["true"], dtype=np.int64)
        p_prob = np.vstack(pooled[mdl]["prob"])
        p_pred = p_prob.argmax(axis=1)
        pooled_metrics = compute_metrics(p_true, p_pred, p_prob)

        def _ms(key: str, _rows: list = fold_rows[mdl]) -> dict[str, float]:
            vals = [fr["test"][key] for fr in _rows]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        cv_ms = {k: _ms(k) for k in ("macro_f1", "balanced_accuracy", "macro_auc_ovr")}
        summary[mdl] = {
            "cv_mean_std": cv_ms,
            "pooled_test_metrics": pooled_metrics,
            "per_fold": fold_rows[mdl],
        }
        print(
            f"\n[{mdl}]"
            f"\n  macro F1      {cv_ms['macro_f1']['mean']:.3f} ± {cv_ms['macro_f1']['std']:.3f}"
            f"\n  balanced acc  {cv_ms['balanced_accuracy']['mean']:.3f} ± {cv_ms['balanced_accuracy']['std']:.3f}"
            f"\n  macro AUROC   {cv_ms['macro_auc_ovr']['mean']:.3f} ± {cv_ms['macro_auc_ovr']['std']:.3f}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "cv_summary_linear_probe.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _wandb_cv_summary(wandb_cfg, group_name, summary)

    if export_publication:
        from mil_baseline.export import export_publication_artifacts
        # Write per-model result dirs for publication export
        for mdl in ("logreg", "knn"):
            mdl_dir = results_dir / mdl
            mdl_dir.mkdir(exist_ok=True)
            mdl_summary = {
                **{k: summary[k] for k in ("n_slides", "n_patients", "n_splits", "fold_protocol",
                                            "encoder", "seed", "git_sha", "class_names")},
                "mil_model": f"linear_{mdl}",
                "cv_mean_std": summary[mdl]["cv_mean_std"],
                "per_fold": summary[mdl]["per_fold"],
                "pooled_test_metrics": summary[mdl]["pooled_test_metrics"],
            }
            with open(mdl_dir / "cv_summary.json", "w") as f:
                json.dump(mdl_summary, f, indent=2)
            print(f"Publication artifacts (skipped — no fold .pt files for {mdl})")

    return out_json


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def _wandb_fold(wandb_cfg: dict, group_name: str, fold: int, probe_results: dict) -> None:
    try:
        import wandb

        mode = wandb_cfg.get("mode") or None
        wandb.init(
            project=wandb_cfg.get("project", "nottingham-mil-benchmark"),
            entity=wandb_cfg.get("entity") or None,
            group=group_name,
            name=f"fold_{fold}_linear",
            job_type="train",
            reinit=True,
            mode=mode,
        )
        log_dict: dict = {}
        for mdl in ("logreg", "knn"):
            m = probe_results[mdl]["metrics"]
            log_dict[f"fold_{fold}/{mdl}/test_macro_f1"] = m["macro_f1"]
            log_dict[f"fold_{fold}/{mdl}/test_balanced_acc"] = m["balanced_accuracy"]
            log_dict[f"fold_{fold}/{mdl}/test_macro_auc"] = m["macro_auc_ovr"]
        wandb.log(log_dict)
        wandb.finish()
    except Exception:
        pass


def _wandb_cv_summary(wandb_cfg: dict, group_name: str, summary: dict) -> None:
    try:
        import wandb

        mode = wandb_cfg.get("mode") or None
        wandb.init(
            project=wandb_cfg.get("project", "nottingham-mil-benchmark"),
            entity=wandb_cfg.get("entity") or None,
            group=group_name,
            name="cv_summary_linear",
            job_type="eval",
            reinit=True,
            mode=mode,
        )
        flat: dict = {}
        for mdl in ("logreg", "knn"):
            if mdl not in summary:
                continue
            ms = summary[mdl]["cv_mean_std"]
            for metric, stats in ms.items():
                flat[f"cv/{mdl}/mean_{metric}"] = stats["mean"]
                flat[f"cv/{mdl}/std_{metric}"] = stats["std"]
        flat["cv/n_slides"] = summary["n_slides"]
        flat["cv/n_patients"] = summary.get("n_patients", summary["n_slides"])
        flat["cv/n_folds"] = summary["n_splits"]
        wandb.log(flat)
        wandb.finish()
    except Exception:
        pass
