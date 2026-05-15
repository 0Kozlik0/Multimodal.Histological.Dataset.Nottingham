from __future__ import annotations

import warnings as _w
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mil_baseline.dataset import SlideFeatureDataset
from mil_baseline.metrics import compute_metrics
from mil_baseline.models import CLAM_SB, build_mil


def _wlog(metrics: dict, step: int | None = None) -> None:
    """Log to the active wandb run; silently no-ops when wandb is disabled or absent."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def _collect_probs(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[int] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"][0].to(device)
            y = int(batch["y"][0].item())
            logits, _ = model(feats)
            p = F.softmax(logits, dim=-1).detach().cpu().numpy()
            ys.append(y)
            probs.append(p)
    return np.array(ys, dtype=np.int64), np.stack(probs, axis=0)


def train_one_fold(
    train_ids: list[str],
    train_y: list[int],
    val_ids: list[str],
    val_y: list[int],
    test_ids: list[str],
    test_y: list[int],
    features_dir: Path,
    cfg: dict,
    device: torch.device,
    fold_dir: Path,
    train_pids: list[str] | None = None,
    val_pids: list[str] | None = None,
    test_pids: list[str] | None = None,
    fold_idx: int = 0,
    class_names: list[str] | None = None,
    n_folds: int = 1,
) -> dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    n_classes = int(cfg["num_classes"])
    mil_cfg = cfg["mil"]
    tr_cfg = cfg["train"]
    clam_cfg = cfg.get("clam", {})
    inst_weight = float(clam_cfg.get("instance_weight", 0.3))
    model_name = mil_cfg["model"].upper()

    sample = torch.load(
        features_dir / f"{train_ids[0]}.pt", map_location="cpu", weights_only=False
    )
    input_dim = int(sample["features"].shape[1])

    print(
        f"\n{'─'*60}\n"
        f"  Fold {fold_idx + 1}/{n_folds}  ·  {model_name}  ·  feat_dim={input_dim}\n"
        f"  train={len(train_ids)} slides  val={len(val_ids)}  test={len(test_ids)}\n"
        f"{'─'*60}"
    )

    model = build_mil(mil_cfg["model"], input_dim, n_classes, mil_cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )

    def make_loader(
        ids: list[str], ys: list[int], pids: list[str] | None, shuffle: bool
    ) -> DataLoader:
        ds = SlideFeatureDataset(ids, ys, features_dir, patient_ids=pids)
        return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_ids, train_y, train_pids, True)
    val_loader = make_loader(val_ids, val_y, val_pids, False)
    test_loader = make_loader(test_ids, test_y, test_pids, False)

    best_auc = -1.0
    bad = 0
    patience = int(tr_cfg["patience"])
    epochs = int(tr_cfg["epochs"])
    is_clam = isinstance(model, CLAM_SB)
    global_step = 0
    _labels = list(range(n_classes))

    epoch_bar = tqdm(
        range(epochs),
        desc=f"Fold {fold_idx + 1}/{n_folds}",
        unit="ep",
        leave=True,
        dynamic_ncols=True,
    )

    for ep in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            feats = batch["features"][0].to(device)
            y_int = int(batch["y"][0].item())
            y = torch.tensor([y_int], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)

            if is_clam:
                logits, attn, h = model.forward_with_h(feats)
                bag_loss = F.cross_entropy(logits.unsqueeze(0), y)
                i_loss = model.instance_loss(h, attn, y_int)
                loss = bag_loss + inst_weight * i_loss
            else:
                logits, _ = model(feats)
                loss = F.cross_entropy(logits.unsqueeze(0), y)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        vy, vp = _collect_probs(model, val_loader, device, n_classes)
        try:
            val_auc = float(roc_auc_score(vy, vp, multi_class="ovr", average="macro"))
        except ValueError:
            val_auc = 0.0
        vp_pred = vp.argmax(axis=1)
        try:
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                val_bacc = float(balanced_accuracy_score(vy, vp_pred))
                val_f1 = float(
                    f1_score(vy, vp_pred, average="macro", labels=_labels, zero_division=0)
                )
        except Exception:
            val_bacc = val_f1 = 0.0

        improved = val_auc > best_auc
        epoch_bar.set_postfix(
            loss=f"{avg_train_loss:.3f}",
            val_auc=f"{val_auc:.3f}",
            val_f1=f"{val_f1:.3f}",
            bad=f"{bad}/{patience}",
            best="✓" if improved else "",
        )

        _wlog(
            {
                "train/loss": avg_train_loss,
                "val/macro_auc": val_auc,
                "val/balanced_acc": val_bacc,
                "val/macro_f1": val_f1,
                "epoch": ep,
                "fold": fold_idx,
            },
            step=global_step,
        )

        if improved:
            best_auc = val_auc
            bad = 0
            torch.save(model.state_dict(), fold_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                epoch_bar.write(
                    f"  Early stop at epoch {ep + 1} "
                    f"(no val AUC improvement for {patience} epochs)"
                )
                break

    epoch_bar.close()

    # ── Evaluate best checkpoint on test set ──────────────────────────────
    print(f"  Evaluating best model on {len(test_ids)} test slides …")
    model.load_state_dict(
        torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    )
    ty, tp = _collect_probs(model, test_loader, device, n_classes)
    y_pred = tp.argmax(axis=1)
    test_metrics = compute_metrics(ty, y_pred, tp)

    print(
        f"  Fold {fold_idx + 1} test results:\n"
        f"    macro F1      = {test_metrics['macro_f1']:.3f}\n"
        f"    balanced acc  = {test_metrics['balanced_accuracy']:.3f}\n"
        f"    macro AUROC   = {test_metrics['macro_auc_ovr']:.3f}"
    )

    torch.save(
        {
            "y_true": ty,
            "y_prob": tp,
            "y_pred": y_pred,
            "slide_ids": test_ids,
            "patient_ids": test_pids if test_pids is not None else test_ids,
            "fold": fold_idx,
        },
        fold_dir / "test_predictions.pt",
    )

    _log_fold_end(ty, y_pred, tp, test_metrics, fold_idx, class_names)

    return {"val_best_macro_auc": best_auc, "test": test_metrics}


def _log_fold_end(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict,
    fold_idx: int,
    class_names: list[str] | None,
) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        names = class_names or [f"class_{i}" for i in range(y_prob.shape[1])]
        wandb.log(
            {
                f"fold_{fold_idx}/test_macro_f1": metrics["macro_f1"],
                f"fold_{fold_idx}/test_balanced_acc": metrics["balanced_accuracy"],
                f"fold_{fold_idx}/test_macro_auc": metrics["macro_auc_ovr"],
                f"fold_{fold_idx}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true.tolist(),
                    preds=y_pred.tolist(),
                    class_names=names,
                ),
            }
        )
    except Exception:
        pass
