from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from mil_baseline.manifest import load_labels_json, require_existing_slides
from mil_baseline.metrics import compute_metrics, macro_ovr_auc
from mil_baseline.trainer import train_one_fold
from mil_baseline.utils import get_git_sha, load_yaml, repo_root, resolve_path, seed_everything


# ---------------------------------------------------------------------------
# Patient-level fold construction
# ---------------------------------------------------------------------------

def _make_patient_folds(
    slide_ids: list[str],
    patient_ids: list[str],
    ys: list[int],
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Build outer CV folds respecting patient disjointness.

    Tries StratifiedGroupKFold(n_splits) first; falls back to 3-fold if that
    fails (too few samples per class), then to unstratified GroupKFold.
    Returns (folds, actual_n_splits).
    """
    from sklearn.model_selection import GroupKFold

    idx = np.arange(len(slide_ids))
    ys_arr = np.array(ys)
    groups = np.array(patient_ids)

    for k in ([n_splits] if n_splits <= 3 else [n_splits, 3]):
        try:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            folds = list(sgkf.split(idx, ys_arr, groups=groups))
            # Verify every training fold contains all classes
            for tr_idx, _ in folds:
                if len(set(ys_arr[tr_idx])) < len(set(ys)):
                    raise ValueError("Not all classes represented in a training fold")
            if k != n_splits:
                warnings.warn(
                    f"StratifiedGroupKFold({n_splits}) failed — using 3-fold instead.",
                    stacklevel=3,
                )
            return folds, k
        except Exception as exc:
            warnings.warn(
                f"StratifiedGroupKFold(n_splits={k}) failed ({exc}); trying fallback.",
                stacklevel=3,
            )

    # Final fallback: unstratified GroupKFold
    n_patients = len(set(patient_ids))
    k = min(n_splits, n_patients)
    warnings.warn(
        f"Falling back to unstratified GroupKFold(n_splits={k}).  "
        "Patient disjointness is preserved but class proportions may differ across folds.",
        stacklevel=3,
    )
    gkf = GroupKFold(n_splits=k)
    return list(gkf.split(idx, ys_arr, groups=groups)), k


def _inner_patient_split(
    slide_ids: list[str],
    patient_ids: list[str],
    ys: list[int],
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[int], list[str], list[str], list[int]]:
    """Split train slides into fit / early-stopping-val keeping patients disjoint."""
    idx = np.arange(len(slide_ids))
    groups = np.array(patient_ids)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(idx, np.array(ys), groups=groups))
    fit_ids = [slide_ids[i] for i in tr_idx]
    fit_pids = [patient_ids[i] for i in tr_idx]
    fit_ys = [ys[i] for i in tr_idx]
    va_ids = [slide_ids[i] for i in va_idx]
    va_pids = [patient_ids[i] for i in va_idx]
    va_ys = [ys[i] for i in va_idx]
    return fit_ids, fit_pids, fit_ys, va_ids, va_pids, va_ys


# ---------------------------------------------------------------------------
# Main CV entry point
# ---------------------------------------------------------------------------

def _audit_leakage(
    fit_ids: list[str],
    va_ids: list[str],
    te_ids: list[str],
    fit_pids: list[str],
    va_pids: list[str],
    te_pids: list[str],
    fold_idx: int,
) -> None:
    """Assert no slide or patient appears in more than one split."""
    fit_s, va_s, te_s = set(fit_ids), set(va_ids), set(te_ids)
    fit_p, va_p, te_p = set(fit_pids), set(va_pids), set(te_pids)

    slide_fv = fit_s & va_s
    slide_ft = fit_s & te_s
    slide_vt = va_s & te_s
    pat_fv = fit_p & va_p
    pat_ft = fit_p & te_p
    pat_vt = va_p & te_p

    print(
        f"[CV leakage audit fold {fold_idx}]"
        f"  fit={len(fit_ids)} slides/{len(fit_p)} patients"
        f"  val={len(va_ids)}/{len(va_p)}"
        f"  test={len(te_ids)}/{len(te_p)}"
        f"  slide overlaps fit∩val={len(slide_fv)} fit∩test={len(slide_ft)} val∩test={len(slide_vt)}"
        f"  patient overlaps fit∩val={len(pat_fv)} fit∩test={len(pat_ft)} val∩test={len(pat_vt)}"
    )
    assert not slide_fv, f"Fold {fold_idx}: fit/val slide overlap: {slide_fv}"
    assert not slide_ft, f"Fold {fold_idx}: fit/test slide overlap: {slide_ft}"
    assert not slide_vt, f"Fold {fold_idx}: val/test slide overlap: {slide_vt}"
    assert not pat_fv, f"Fold {fold_idx}: fit/val patient overlap: {pat_fv}"
    assert not pat_ft, f"Fold {fold_idx}: fit/test patient overlap: {pat_ft}"
    assert not pat_vt, f"Fold {fold_idx}: val/test patient overlap: {pat_vt}"


def run_train_cv(config_path: Path, export_publication: bool = False) -> Path:
    cfg = load_yaml(config_path)
    root = repo_root()
    slides_dir = resolve_path(root, cfg["slides_dir"])
    labels_path = resolve_path(root, cfg["labels_json"])
    feats_dir = resolve_path(root, cfg["features_dir"])
    results_dir = resolve_path(root, cfg["results_dir"])
    raw_seed = cfg["train"].get("seed")
    seed = int(raw_seed) if raw_seed is not None else None
    seed_everything(seed)

    df = load_labels_json(labels_path, slides_dir, int(cfg["num_classes"]))
    df = require_existing_slides(df)

    missing_feats = [
        row["slide_id"]
        for _, row in df.iterrows()
        if not (feats_dir / f"{row['slide_id']}.pt").exists()
    ]
    if missing_feats:
        raise FileNotFoundError(
            "Feature files missing. Run: python -m mil_baseline.cli extract --config ...\n"
            "Missing: "
            + ", ".join(missing_feats[:8])
            + (" ..." if len(missing_feats) > 8 else "")
        )

    slide_ids = df["slide_id"].tolist()
    patient_ids = df["patient_id"].tolist()
    ys = [int(x) for x in df["y"].tolist()]
    class_names: list[str] = cfg.get("class_names") or [
        f"class_{i}" for i in range(int(cfg["num_classes"]))
    ]

    n_splits = int(cfg["cv"]["n_splits"])
    fold_seed = seed if seed is not None else 42  # reproducible fold splits even when model init is unseeded
    folds, actual_n_splits = _make_patient_folds(slide_ids, patient_ids, ys, n_splits, fold_seed)
    if actual_n_splits != n_splits:
        print(f"[CV] Using {actual_n_splits}-fold CV (requested {n_splits}).")

    # wandb group name: ties all fold runs together in the UI
    wandb_cfg: dict = cfg.get("wandb") or {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_name = cfg["encoder"]["name"]
    mil_name = cfg["mil"]["model"]
    group_name: str = wandb_cfg.get("group") or f"{encoder_name}_{mil_name}_{ts}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows: list[dict] = []
    pooled_true: list[int] = []
    pooled_prob: list[np.ndarray] = []

    val_frac = float(cfg["train"]["val_frac"])

    for fold, (tr_idx, te_idx) in enumerate(folds):
        fold_dir = results_dir / f"fold_{fold}"

        tr_slide_ids = [slide_ids[i] for i in tr_idx]
        tr_patient_ids = [patient_ids[i] for i in tr_idx]
        tr_ys = [ys[i] for i in tr_idx]
        te_ids = [slide_ids[i] for i in te_idx]
        te_pids = [patient_ids[i] for i in te_idx]
        te_ys = [ys[i] for i in te_idx]

        inner_seed = fold_seed + fold
        fit_ids, fit_pids, fit_ys, va_ids, va_pids, va_ys = _inner_patient_split(
            tr_slide_ids, tr_patient_ids, tr_ys, val_frac, inner_seed
        )

        _audit_leakage(fit_ids, va_ids, te_ids, fit_pids, va_pids, te_pids, fold)

        print(
            f"[Fold {fold}/{actual_n_splits - 1}]  "
            f"fit={len(fit_ids)} slides / {len(set(fit_pids))} patients  "
            f"val={len(va_ids)} / {len(set(va_pids))}  "
            f"test={len(te_ids)} / {len(set(te_pids))}"
        )

        # wandb: one run per fold
        _wandb_init_fold(
            wandb_cfg=wandb_cfg,
            group_name=group_name,
            fold=fold,
            cfg=cfg,
            actual_n_splits=actual_n_splits,
            n_fit=len(fit_ids),
            n_val=len(va_ids),
            n_test=len(te_ids),
            labels_path=labels_path,
        )

        try:
            row = train_one_fold(
                fit_ids,
                fit_ys,
                va_ids,
                va_ys,
                te_ids,
                te_ys,
                feats_dir,
                cfg,
                device,
                fold_dir,
                train_pids=fit_pids,
                val_pids=va_pids,
                test_pids=te_pids,
                fold_idx=fold,
                class_names=class_names,
                n_folds=actual_n_splits,
            )
        finally:
            _wandb_finish()

        row["fold"] = fold
        fold_rows.append(row)

        pred = torch.load(
            fold_dir / "test_predictions.pt", map_location="cpu", weights_only=False
        )
        pooled_true.extend(pred["y_true"].tolist())
        pooled_prob.append(pred["y_prob"])

    # ------------------------------------------------------------------
    # Aggregate metrics across folds
    # ------------------------------------------------------------------
    pooled_true_arr = np.array(pooled_true, dtype=np.int64)
    pooled_prob_arr = np.vstack(pooled_prob)
    pooled_pred_arr = pooled_prob_arr.argmax(axis=1)
    pooled_metrics = compute_metrics(pooled_true_arr, pooled_pred_arr, pooled_prob_arr)

    def mean_std(key: str) -> dict[str, float]:
        vals = [fr["test"][key] for fr in fold_rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    cv_mean_std = {
        "macro_f1": mean_std("macro_f1"),
        "macro_auc_ovr": mean_std("macro_auc_ovr"),
        "balanced_accuracy": mean_std("balanced_accuracy"),
        "weighted_f1": mean_std("weighted_f1"),
    }

    print(
        f"\n[CV summary — {actual_n_splits}-fold patient-level]"
        f"\n  macro F1        {cv_mean_std['macro_f1']['mean']:.3f} ± {cv_mean_std['macro_f1']['std']:.3f}"
        f"\n  balanced acc    {cv_mean_std['balanced_accuracy']['mean']:.3f} ± {cv_mean_std['balanced_accuracy']['std']:.3f}"
        f"\n  macro AUROC     {cv_mean_std['macro_auc_ovr']['mean']:.3f} ± {cv_mean_std['macro_auc_ovr']['std']:.3f}"
    )

    summary: dict[str, Any] = {
        "n_slides": len(slide_ids),
        "n_patients": len(set(patient_ids)),
        "n_splits": actual_n_splits,
        "fold_protocol": f"{actual_n_splits}-fold patient-level StratifiedGroupKFold",
        "mil_model": mil_name,
        "encoder": encoder_name,
        "seed": seed,
        "git_sha": get_git_sha(),
        "class_names": class_names,
        "pooled_test_metrics": pooled_metrics,
        "pooled_macro_auc_ovr": float(macro_ovr_auc(pooled_true_arr, pooled_prob_arr)),
        "per_fold": fold_rows,
        "cv_mean_std": cv_mean_std,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "cv_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Log wandb CV summary run
    _wandb_log_cv_summary(wandb_cfg, group_name, cv_mean_std, summary)

    if export_publication:
        from mil_baseline.export import export_publication_artifacts

        pub_dir = export_publication_artifacts(results_dir, config_path)
        print(f"Publication artifacts written to: {pub_dir}")

    return out_json


# ---------------------------------------------------------------------------
# wandb helpers — all failures are silently swallowed so the pipeline never
# crashes when wandb is unavailable or disabled.
# ---------------------------------------------------------------------------

def _wandb_init_fold(
    wandb_cfg: dict,
    group_name: str,
    fold: int,
    cfg: dict,
    actual_n_splits: int,
    n_fit: int,
    n_val: int,
    n_test: int,
    labels_path: Path,
) -> None:
    try:
        import wandb

        mode = wandb_cfg.get("mode") or None  # None → use WANDB_MODE env var
        wandb.init(
            project=wandb_cfg.get("project", "nottingham-mil-benchmark"),
            entity=wandb_cfg.get("entity") or None,
            group=group_name,
            name=f"fold_{fold}",
            job_type="train",
            reinit=True,
            mode=mode,
            config={
                "fold": fold,
                "n_folds": actual_n_splits,
                "seed": cfg["train"]["seed"],
                "encoder": cfg["encoder"]["name"],
                "mil_model": cfg["mil"]["model"],
                "patch_size": cfg["patch"]["size"],
                "patch_level": cfg["patch"]["level"],
                "max_patches_per_slide": cfg["patch"]["max_patches_per_slide"],
                "n_train_slides": n_fit,
                "n_val_slides": n_val,
                "n_test_slides": n_test,
                "labels_json": str(labels_path),
                "git_sha": get_git_sha(),
            },
        )
    except Exception as exc:
        warnings.warn(f"wandb.init failed (continuing without wandb): {exc}", stacklevel=2)


def _wandb_finish() -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _wandb_log_cv_summary(
    wandb_cfg: dict,
    group_name: str,
    cv_mean_std: dict,
    summary: dict,
) -> None:
    try:
        import wandb

        mode = wandb_cfg.get("mode") or None
        wandb.init(
            project=wandb_cfg.get("project", "nottingham-mil-benchmark"),
            entity=wandb_cfg.get("entity") or None,
            group=group_name,
            name="cv_summary",
            job_type="eval",
            reinit=True,
            mode=mode,
        )
        flat: dict = {}
        for metric, stats in cv_mean_std.items():
            flat[f"cv/mean_{metric}"] = stats["mean"]
            flat[f"cv/std_{metric}"] = stats["std"]
        flat["cv/n_slides"] = summary["n_slides"]
        flat["cv/n_patients"] = summary.get("n_patients", summary["n_slides"])
        flat["cv/n_folds"] = summary["n_splits"]
        wandb.log(flat)
        wandb.finish()
    except Exception:
        pass
