"""Statistical utilities: bootstrap CIs and permutation tests."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of *scores* (one value per fold/sample)."""
    rng = np.random.default_rng(seed)
    boots = np.array(
        [rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def permutation_test_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """One-tailed permutation test: observed macro F1 vs label-shuffle null."""
    rng = np.random.default_rng(seed)
    n_classes = int(y_pred.max()) + 1
    labels = list(range(n_classes))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        observed = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
        null_dist = np.array(
            [
                f1_score(
                    rng.permutation(y_true), y_pred,
                    average="macro", labels=labels, zero_division=0,
                )
                for _ in range(n_perm)
            ]
        )
    p_val = float((null_dist >= observed).sum() / n_perm)
    return observed, p_val


def permutation_test_balanced_acc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Same as permutation_test_macro_f1 but for balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    observed = float(balanced_accuracy_score(y_true, y_pred))
    null_dist = np.array(
        [balanced_accuracy_score(rng.permutation(y_true), y_pred) for _ in range(n_perm)]
    )
    p_val = float((null_dist >= observed).sum() / n_perm)
    return observed, p_val


def ci_summary(
    per_fold_scores: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute mean, std, and bootstrap CI for a list of per-fold scores."""
    arr = np.array(per_fold_scores)
    lo, hi = bootstrap_ci(arr, n_boot=n_boot, ci=ci, seed=seed)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        f"ci_{int(ci*100)}_lo": lo,
        f"ci_{int(ci*100)}_hi": hi,
    }
