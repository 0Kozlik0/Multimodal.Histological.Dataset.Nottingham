"""Mitotic-figure ROI classification (structure-level technical validation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mil_baseline.cv import _audit_leakage, _make_patient_folds
from mil_baseline.encoder import build_patch_encoder
from mil_baseline.metrics import compute_metrics
from mil_baseline.mitosis_annotations import SlideAnnotation, load_all_annotations
from mil_baseline.patching import open_slide
from mil_baseline.utils import get_git_sha, load_yaml, repo_root, resolve_path, seed_everything


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Standard ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _sample_negative_centers(
    ann: SlideAnnotation,
    n_needed: int,
    min_sep_px: float,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> list[tuple[float, float]]:
    """Rejection-sample negative ROI centers inside the (C) counting-field polygon
    (or, if absent, a padded bbox around all mitoses), at least min_sep_px away
    from every mitotic figure and every other sampled negative."""
    pos_centers = [m.centroid for m in ann.mitoses]

    if ann.field_polygon is not None:
        poly_pts = ann.field_polygon.points
        xmin, ymin, xmax, ymax = ann.field_polygon.bbox
    else:
        poly_pts = None
        if not pos_centers:
            return []
        xs = np.array([p[0] for p in pos_centers])
        ys = np.array([p[1] for p in pos_centers])
        pad = min_sep_px * 4
        xmin, ymin, xmax, ymax = xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad

    min_sep2 = min_sep_px**2
    negs: list[tuple[float, float]] = []
    tries = 0
    while len(negs) < n_needed and tries < max_tries:
        tries += 1
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if poly_pts is not None and not _point_in_polygon(x, y, poly_pts):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < min_sep2 for px, py in pos_centers):
            continue
        if any((x - nx) ** 2 + (y - ny) ** 2 < min_sep2 for nx, ny in negs):
            continue
        negs.append((x, y))
    return negs


# ---------------------------------------------------------------------------
# ROI manifest
# ---------------------------------------------------------------------------

def build_roi_manifest(
    annotations_dir: Path,
    neg_per_pos: float,
    min_sep_px: float,
    seed: int,
) -> pd.DataFrame:
    anns = load_all_annotations(annotations_dir)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for slide_id, ann in anns.items():
        for i, m in enumerate(ann.mitoses):
            cx, cy = m.centroid
            rows.append(
                {"roi_id": f"{slide_id}__pos{i}", "slide_id": slide_id, "x": cx, "y": cy, "label": 1}
            )
        n_neg = max(1, int(round(len(ann.mitoses) * neg_per_pos)))
        negs = _sample_negative_centers(ann, n_neg, min_sep_px, rng)
        for i, (nx, ny) in enumerate(negs):
            rows.append(
                {"roi_id": f"{slide_id}__neg{i}", "slide_id": slide_id, "x": nx, "y": ny, "label": 0}
            )
    if not rows:
        raise RuntimeError(f"No ROIs built from annotations in {annotations_dir}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Slide path resolution
# ---------------------------------------------------------------------------

def _resolve_slide_path(
    slides_dir: Path, slide_id: str, years: tuple[str, ...] = ("2023", "2024", "2025")
) -> Path | None:
    for year in years:
        year_dir = slides_dir / year
        if not year_dir.is_dir():
            continue
        hits = sorted(year_dir.rglob(f"{slide_id}.mrxs"))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Patch extraction + encoding
# ---------------------------------------------------------------------------

def run_mitosis_extract(config_path: Path, overwrite: bool = False) -> Path:
    cfg = load_yaml(config_path)
    root = repo_root()
    slides_dir = resolve_path(root, cfg["slides_dir"])
    annotations_dir = resolve_path(root, cfg["annotations_dir"])
    feats_dir = resolve_path(root, cfg["features_dir"])
    manifest_path = resolve_path(root, cfg.get("manifest_path", "data/mitosis_roi/roi_manifest.csv"))

    seed = int(cfg.get("train", {}).get("seed") or 42)
    seed_everything(seed)

    sampling_cfg = cfg["sampling"]
    manifest = build_roi_manifest(
        annotations_dir,
        neg_per_pos=float(sampling_cfg["neg_per_pos"]),
        min_sep_px=float(sampling_cfg["min_sep_px"]),
        seed=int(sampling_cfg.get("seed", seed)),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(
        f"[mitosis-extract] ROI manifest: {len(manifest)} ROIs "
        f"({int((manifest['label'] == 1).sum())} positive / {int((manifest['label'] == 0).sum())} negative) "
        f"across {manifest['slide_id'].nunique()} slides -> {manifest_path}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_encoder = build_patch_encoder(cfg["encoder"], device)
    patch_cfg = cfg["patch"]
    level = int(patch_cfg["level"])
    size = int(patch_cfg["size"])
    half = size // 2
    batch_size = int(cfg["encoder"].get("batch_size", 64))

    feats_dir.mkdir(parents=True, exist_ok=True)
    slide_groups = list(manifest.groupby("slide_id"))
    for slide_id, grp in tqdm(slide_groups, desc="Slides", unit="slide"):
        out_pt = feats_dir / f"{slide_id}.pt"
        if out_pt.exists() and not overwrite:
            continue
        slide_path = _resolve_slide_path(slides_dir, slide_id)
        if slide_path is None:
            raise FileNotFoundError(
                f"No resolvable .mrxs path for annotated slide '{slide_id}' under {slides_dir}."
            )

        feats_chunks: list[torch.Tensor] = []
        roi_ids: list[str] = []
        labels: list[int] = []
        coords: list[tuple[int, int]] = []
        with open_slide(slide_path) as slide:
            # QuPath (MIRAX/.mrxs) annotation coordinates are relative to the
            # scanner's "bounds" sub-region, not OpenSlide's full-canvas level-0
            bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
            bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
            buf: list[np.ndarray] = []
            buf_meta: list[tuple[str, int, int, int]] = []
            for _, row in grp.iterrows():
                x0 = int(round(row["x"] + bounds_x - half))
                y0 = int(round(row["y"] + bounds_y - half))
                tile = slide.read_region((x0, y0), level, (size, size)).convert("RGB")
                buf.append(np.asarray(tile, dtype=np.uint8))
                buf_meta.append((row["roi_id"], int(row["label"]), x0, y0))
                if len(buf) >= batch_size:
                    feats_chunks.append(patch_encoder.encode_numpy_list(buf).float())
                    roi_ids += [m[0] for m in buf_meta]
                    labels += [m[1] for m in buf_meta]
                    coords += [(m[2], m[3]) for m in buf_meta]
                    buf.clear()
                    buf_meta.clear()
            if buf:
                feats_chunks.append(patch_encoder.encode_numpy_list(buf).float())
                roi_ids += [m[0] for m in buf_meta]
                labels += [m[1] for m in buf_meta]
                coords += [(m[2], m[3]) for m in buf_meta]

        feats = torch.cat(feats_chunks, dim=0)
        torch.save(
            {
                "features": feats,
                "roi_ids": roi_ids,
                "labels": torch.tensor(labels, dtype=torch.int64),
                "coords": np.array(coords, dtype=np.int32),
                "meta": {
                    "slide_id": slide_id,
                    "encoder": cfg["encoder"]["name"],
                    "patch_size": size,
                    "level": level,
                    "n_rois": len(roi_ids),
                },
            },
            out_pt,
        )

    print(f"[mitosis-extract] Done. Features saved to {feats_dir}/")
    return manifest_path


# ---------------------------------------------------------------------------
# Training / CV
# ---------------------------------------------------------------------------

def _load_roi_features(
    feats_dir: Path, slide_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    Xs, ys, roi_ids, sids = [], [], [], []
    for sid in slide_ids:
        obj = torch.load(feats_dir / f"{sid}.pt", map_location="cpu", weights_only=False)
        Xs.append(obj["features"].numpy())
        ys.append(obj["labels"].numpy())
        roi_ids.extend(list(obj["roi_ids"]))
        sids.extend([sid] * len(obj["roi_ids"]))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y, roi_ids, sids


def _fit_mitosis_logreg(
    X_fit: np.ndarray, y_fit: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_fit)
    Xf, Xt = sc.transform(X_fit), sc.transform(X_test)
    clf = LogisticRegression(
        max_iter=2000, random_state=seed, C=1.0, solver="lbfgs", class_weight="balanced"
    )
    clf.fit(Xf, y_fit)
    y_pred = clf.predict(Xt)
    y_prob = clf.predict_proba(Xt)
    return compute_metrics(y_test, y_pred, y_prob), y_pred, y_prob


def run_mitosis_cv(config_path: Path, export_publication: bool = False) -> Path:
    cfg = load_yaml(config_path)
    root = repo_root()
    feats_dir = resolve_path(root, cfg["features_dir"])
    results_dir = resolve_path(root, cfg["results_dir"])
    seed = int(cfg.get("train", {}).get("seed") or 42)
    seed_everything(seed)

    slide_ids = sorted(p.stem for p in feats_dir.glob("*.pt"))
    if not slide_ids:
        raise FileNotFoundError(
            f"No ROI feature files in {feats_dir}. Run mitosis-extract first."
        )
    X, y, roi_ids, sids = _load_roi_features(feats_dir, slide_ids)
    class_names: list[str] = cfg.get("class_names") or ["non_mitotic", "mitotic"]
    print(
        f"[mitosis-train-cv] {len(roi_ids)} ROIs "
        f"({int((y == 1).sum())} positive / {int((y == 0).sum())} negative) "
        f"across {len(slide_ids)} slides"
    )

    n_splits = int(cfg["cv"]["n_splits"])
    folds, actual_n_splits = _make_patient_folds(roi_ids, sids, y.tolist(), n_splits, seed)
    if actual_n_splits != n_splits:
        print(f"[mitosis-train-cv] Using {actual_n_splits}-fold CV (requested {n_splits}).")

    fold_rows: list[dict] = []
    pooled_true: list[int] = []
    pooled_prob: list[np.ndarray] = []

    for fold, (tr_idx, te_idx) in enumerate(folds):
        fit_sids = [sids[i] for i in tr_idx]
        te_sids = [sids[i] for i in te_idx]
        _audit_leakage(
            [roi_ids[i] for i in tr_idx], [], [roi_ids[i] for i in te_idx],
            fit_sids, [], te_sids, fold,
        )
        metrics, y_pred, y_prob = _fit_mitosis_logreg(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], seed + fold
        )
        fold_dir = results_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "y_true": y[te_idx],
                "y_pred": y_pred,
                "y_prob": y_prob,
                "slide_ids": [roi_ids[i] for i in te_idx],
                "patient_ids": te_sids,
            },
            fold_dir / "test_predictions.pt",
        )
        fold_rows.append({"fold": fold, "test": metrics})
        pooled_true.extend(y[te_idx].tolist())
        pooled_prob.append(y_prob)
        print(
            f"[Fold {fold}/{actual_n_splits - 1}]  n_test={len(te_idx)} "
            f"({len(set(te_sids))} slides)  macro_f1={metrics['macro_f1']:.3f}  "
            f"balanced_acc={metrics['balanced_accuracy']:.3f}  auroc={metrics['macro_auc_ovr']:.3f}"
        )

    pooled_true_arr = np.array(pooled_true, dtype=np.int64)
    pooled_prob_arr = np.vstack(pooled_prob)
    pooled_pred_arr = pooled_prob_arr.argmax(axis=1)
    pooled_metrics = compute_metrics(pooled_true_arr, pooled_pred_arr, pooled_prob_arr)

    def mean_std(key: str) -> dict[str, float]:
        vals = [fr["test"][key] for fr in fold_rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    cv_mean_std = {
        k: mean_std(k) for k in ("macro_f1", "balanced_accuracy", "macro_auc_ovr", "weighted_f1")
    }

    print(
        f"\n[mitosis CV summary — {actual_n_splits}-fold, slide-grouped]"
        f"\n  macro F1        {cv_mean_std['macro_f1']['mean']:.3f} ± {cv_mean_std['macro_f1']['std']:.3f}"
        f"\n  balanced acc    {cv_mean_std['balanced_accuracy']['mean']:.3f} ± {cv_mean_std['balanced_accuracy']['std']:.3f}"
        f"\n  macro AUROC     {cv_mean_std['macro_auc_ovr']['mean']:.3f} ± {cv_mean_std['macro_auc_ovr']['std']:.3f}"
    )

    summary: dict[str, Any] = {
        "n_slides": len(slide_ids),
        "n_patients": len(set(sids)),
        "n_rois": len(roi_ids),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_splits": actual_n_splits,
        "n_seeds": 1,
        "total_folds": len(fold_rows),
        "fold_protocol": f"{actual_n_splits}-fold slide-grouped StratifiedGroupKFold (ROI-level, group=source slide)",
        "mil_model": "mitosis_logreg",
        "encoder": cfg["encoder"]["name"],
        "seed": seed,
        "git_sha": get_git_sha(),
        "class_names": class_names,
        "per_fold": fold_rows,
        "cv_mean_std": cv_mean_std,
        "pooled_test_metrics": pooled_metrics,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "cv_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if export_publication:
        from mil_baseline.export import export_publication_artifacts

        pub_dir = export_publication_artifacts(results_dir, config_path)
        print(f"Publication artifacts written to: {pub_dir}")

    return out_json
