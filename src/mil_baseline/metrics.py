from __future__ import annotations

import warnings as _warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    # Pass the full label set inferred from proba so metrics are well-defined even
    # when a fold's y_true or y_pred is missing one of the Nottingham grades.
    # The "y_pred contains classes not in y_true" warning is expected and benign for
    # small folds — suppress it here so it doesn't pollute training output.
    n_classes = proba.shape[1]
    labels = list(range(n_classes))
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        out = {
            "macro_auc_ovr": macro_ovr_auc(y_true, proba),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "weighted_f1": float(
                f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }
    return out
