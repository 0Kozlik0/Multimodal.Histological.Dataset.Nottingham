from __future__ import annotations

import io
import warnings as _w
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc as sklearn_auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

from mil_baseline.dataset import SlideFeatureDataset
from mil_baseline.metrics import compute_metrics
from mil_baseline.models import CLAM_SB, build_mil

# IBM colorblind-safe palette (blue, magenta, gold, violet, orange)
_CB = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#FE6100"]


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Load EMA weights into *model*, return original weights for restoration."""
        original: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                original[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])
        return original

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, original: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in original:
                p.data.copy_(original[name])


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def _wlog(metrics: dict, step: int | None = None) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def _fig_to_wandb_image(fig):
    import wandb
    from PIL import Image as _PILImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    pil = _PILImage.open(buf).copy()
    buf.close()
    return wandb.Image(pil)


# ---------------------------------------------------------------------------
# matplotlib figure builders
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    fold_idx: int,
) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(class_names)
    labels = list(range(n))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig = Figure(figsize=(4.6, 4.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(f"Confusion Matrix — Fold {fold_idx + 1}", fontsize=10, fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                ha="center", va="center", fontsize=8, color=color, fontweight="bold",
            )

    fig.tight_layout()
    return fig


def _plot_confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    fold_idx: int,
    subtitle: str = "",
) -> Any:
    """Raw-count confusion matrix: Prediction labels on top, colorbar scaled to counts."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(class_names)
    labels = list(range(n))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = Figure(figsize=(4.6, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=0, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Prediction", fontsize=10, labelpad=8)
    ax.set_ylabel("Actual", fontsize=10)

    title = f"Confusion Matrix — Fold {fold_idx + 1}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=14)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > thresh else "#1a1a2e"
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=13, color=color, fontweight="bold",
            )

    fig.tight_layout()
    return fig


def _plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    fold_idx: int,
) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    if n == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig = Figure(figsize=(5.0, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    for i, (name, color) in enumerate(zip(class_names, _CB)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={roc_auc:.3f}")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(f"ROC Curves (One-vs-Rest) — Fold {fold_idx + 1}", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def _plot_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    fold_idx: int,
) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(class_names)
    labels = list(range(n))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    x = np.arange(n)
    width = 0.24

    fig = Figure(figsize=(5.2, 3.9))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    bars_p = ax.bar(x - width, p, width, label="Precision", color=_CB[0], alpha=0.85)
    bars_r = ax.bar(x, r, width, label="Recall", color=_CB[1], alpha=0.85)
    bars_f = ax.bar(x + width, f, width, label="F1", color=_CB[2], alpha=0.85)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

    # Annotate support (n per class) below x-axis labels
    for i, s in enumerate(support):
        ax.text(x[i], -0.08, f"n={s}", ha="center", va="top", fontsize=7, color="grey",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_ylim([0, 1.22])
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(f"Per-class Precision / Recall / F1 — Fold {fold_idx + 1}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def _plot_training_curves(history: dict[str, list[float]], fold_idx: int, best_epoch: int = 0) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = Figure(figsize=(5.5, 3.8))
    FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(111)

    ax1.plot(epochs, history["train_loss"], color=_CB[1], lw=1.5, label="Train loss")
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Loss", fontsize=9, color=_CB[1])
    ax1.tick_params(axis="y", labelcolor=_CB[1], labelsize=8)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_auc"], color=_CB[0], lw=1.5, label="Val AUROC")
    ax2.plot(epochs, history["val_f1"], color=_CB[2], lw=1.5, linestyle="--", label="Val macro-F1")
    ax2.set_ylabel("Metric", fontsize=9)
    ax2.set_ylim([0, 1])
    ax2.tick_params(labelsize=8)
    ax2.spines["top"].set_visible(False)

    # Mark best epoch with a vertical line
    if best_epoch > 0 and best_epoch <= len(epochs):
        ax1.axvline(best_epoch, color="grey", lw=1.0, linestyle=":", alpha=0.8)
        ax1.text(
            best_epoch + 0.3, ax1.get_ylim()[1] * 0.95,
            f"best\nep {best_epoch}",
            fontsize=6.5, color="grey", va="top",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=7.5, framealpha=0.9)

    ax1.set_title(f"Training Curves — Fold {fold_idx + 1}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

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


def _collect_probs_with_aug(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    augment_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as _collect_probs but applies augment_fn to features before forward."""
    model.eval()
    ys: list[int] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"][0].to(device)
            feats = augment_fn(feats)
            y = int(batch["y"][0].item())
            logits, _ = model(feats)
            p = F.softmax(logits, dim=-1).detach().cpu().numpy()
            ys.append(y)
            probs.append(p)
    return np.array(ys, dtype=np.int64), np.stack(probs, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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

    early_stop_metric = str(tr_cfg.get("early_stop_metric", "balanced_acc"))
    min_epochs = int(tr_cfg.get("min_epochs", 10))
    label_smoothing = float(tr_cfg.get("label_smoothing", 0.0))
    ema_decay = float(tr_cfg.get("ema_decay", 0.0))  # >0 enables EMA
    # VAL uses EMA only when requested; test always uses EMA if enabled.
    eval_with_ema = bool(tr_cfg.get("eval_with_ema", False))
    lr_scheduler_kind = str(tr_cfg.get("lr_scheduler", "none")).lower()  # none|cosine
    max_class_weight = tr_cfg.get("max_class_weight")
    tta_passes = int(tr_cfg.get("tta_passes", 1))

    aug_cfg = mil_cfg.get("augment") or {}
    gauss_std = float(aug_cfg.get("gaussian_noise_std", 0.0))
    feat_drop = float(aug_cfg.get("feature_dropout", 0.0))
    patch_drop = float(aug_cfg.get("patch_dropout", 0.0))
    _use_aug = gauss_std > 0 or feat_drop > 0 or patch_drop > 0
    if _use_aug:
        print(
            f"  augmentation: gaussian_noise_std={gauss_std}"
            f"  feature_dropout={feat_drop}"
            f"  patch_dropout={patch_drop}"
        )

    def _augment(feats: torch.Tensor) -> torch.Tensor:
        """Apply feature augmentations in-place; called during training only."""
        if gauss_std > 0:
            feats = feats + torch.randn_like(feats) * gauss_std
        if feat_drop > 0:
            # Mask random feature dimensions 
            mask = (torch.rand(feats.shape[-1], device=feats.device) >= feat_drop).float()
            feats = feats * mask.unsqueeze(0)
        if patch_drop > 0:
            # Drop random patch rows
            keep = torch.rand(feats.shape[0], device=feats.device) >= patch_drop
            if not keep.any():
                keep[0] = True
            feats = feats[keep]
        return feats

    # Class-weighted CE — computed from this fold's train labels only.
    _counts   = Counter(train_y)
    _n_total  = len(train_y)
    class_weight_mode = str(tr_cfg.get("class_weight", "balanced") or "none").lower()
    if class_weight_mode in ("none", "uniform", "off"):
        _raw_w = [1.0] * n_classes
    elif class_weight_mode in ("sqrt", "sqrt_balanced"):
        # Square-root inverse-frequency
        _raw_w = [(_n_total / (n_classes * max(_counts.get(i, 1), 1))) ** 0.5 for i in range(n_classes)]
    else:  # "balanced" — full inverse frequency
        _raw_w = [_n_total / (n_classes * max(_counts.get(i, 1), 1)) for i in range(n_classes)]
    if max_class_weight is not None:
        _cap = float(max_class_weight)
        _raw_w = [min(w, _cap) for w in _raw_w]
    _weights = torch.tensor(_raw_w, dtype=torch.float, device=device)
    print(
        f"  class weights ({class_weight_mode}): "
        + "  ".join(f"G{i+1}={_weights[i].item():.2f}({_counts.get(i,0)} slides)" for i in range(n_classes))
        + (f"  [capped at {max_class_weight}]" if max_class_weight is not None else "")
    )

    model = build_mil(mil_cfg["model"], input_dim, n_classes, mil_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )

    # Optional cosine LR scheduler
    scheduler = None
    if lr_scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(tr_cfg["epochs"]), eta_min=0.0,
        )
        print(f"  lr_scheduler: cosine annealing over {tr_cfg['epochs']} epochs")

    # Optional EMA for stable test-time predictions on small val sets
    ema: ModelEMA | None = None
    if ema_decay > 0:
        ema = ModelEMA(model, decay=ema_decay)
        print(f"  EMA enabled (decay={ema_decay}, eval_with_ema={eval_with_ema})")

    def make_loader(
        ids: list[str], ys: list[int], pids: list[str] | None, shuffle: bool
    ) -> DataLoader:
        ds = SlideFeatureDataset(ids, ys, features_dir, patient_ids=pids)
        return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_ids, train_y, train_pids, True)
    val_loader = make_loader(val_ids, val_y, val_pids, False)
    test_loader = make_loader(test_ids, test_y, test_pids, False)

    best_score = -1.0       # checkpoint selection metric (early_stop_metric)
    best_val_auc = 0.0      # actual val AUROC at the best checkpoint (for logging)
    best_epoch = 1
    best_val_preds: tuple[np.ndarray, np.ndarray] | None = None
    bad = 0
    patience = int(tr_cfg["patience"])
    epochs = int(tr_cfg["epochs"])
    is_clam = isinstance(model, CLAM_SB)
    global_step = 0
    _labels = list(range(n_classes))

    epoch_history: dict[str, list[float]] = {
        "train_loss": [],
        "val_auc": [],
        "val_bacc": [],
        "val_f1": [],
    }

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
            if _use_aug:
                feats = _augment(feats)
            y_int = int(batch["y"][0].item())
            y = torch.tensor([y_int], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)

            if is_clam:
                logits, attn, h = model.forward_with_h(feats)
                bag_loss = F.cross_entropy(
                    logits.unsqueeze(0), y, weight=_weights, label_smoothing=label_smoothing
                )
                i_loss = model.instance_loss(h, attn, y_int)
                loss = bag_loss + inst_weight * i_loss
            else:
                logits, _ = model(feats)
                loss = F.cross_entropy(
                    logits.unsqueeze(0), y, weight=_weights, label_smoothing=label_smoothing
                )

            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += loss.item()
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # When EMA is enabled and we want val to reflect deployed weights, swap in EMA for eval
        if eval_with_ema and ema is not None:
            _orig = ema.apply_to(model)
            vy, vp = _collect_probs(model, val_loader, device, n_classes)
            ema.restore(model, _orig)
        else:
            vy, vp = _collect_probs(model, val_loader, device, n_classes)
        try:
            val_auc = float(roc_auc_score(vy, vp, multi_class="ovr", average="macro"))
        except ValueError:
            val_auc = float("nan")
        vp_pred = vp.argmax(axis=1)
        n_unique_val = int(np.unique(vp_pred).shape[0])
        collapse_warning = n_unique_val < 2
        try:
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                val_bacc = float(balanced_accuracy_score(vy, vp_pred))
                val_f1 = float(
                    f1_score(vy, vp_pred, average="macro", labels=_labels, zero_division=0)
                )
        except Exception:
            val_bacc = val_f1 = 0.0

        if early_stop_metric == "macro_auc":
            raw_ckpt = val_auc if (not np.isnan(val_auc) and n_unique_val >= 2) else 0.0
        elif early_stop_metric == "macro_f1":
            raw_ckpt = val_f1
        elif early_stop_metric in ("fixed_epoch", "none"):
            # Fixed-budget: keep latest epoch weights.
            raw_ckpt = float(ep + 1)
        else:
            raw_ckpt = val_bacc
        if early_stop_metric in ("fixed_epoch", "none"):
            ckpt_score = raw_ckpt
        else:
            # Prefer non-collapsed val predictions for checkpoint selection.
            ckpt_score = 0.0 if collapse_warning else raw_ckpt

        if collapse_warning:
            epoch_bar.write(
                f"  [collapse] fold {fold_idx} ep {ep + 1}: "
                f"val predicts only class(es) {np.unique(vp_pred).tolist()}"
            )

        epoch_history["train_loss"].append(avg_train_loss)
        epoch_history["val_auc"].append(val_auc if not np.isnan(val_auc) else 0.0)
        epoch_history["val_bacc"].append(val_bacc)
        epoch_history["val_f1"].append(val_f1)

        improved = ckpt_score > best_score
        val_auc_display = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A"
        epoch_bar.set_postfix(
            loss=f"{avg_train_loss:.3f}",
            val_bacc=f"{val_bacc:.3f}",
            val_f1=f"{val_f1:.3f}",
            val_auc=val_auc_display,
            bad=f"{bad}/{patience}",
            best="✓" if improved else "",
        )

        _wlog(
            {
                "train/loss": avg_train_loss,
                "val/macro_auc": val_auc if not np.isnan(val_auc) else 0.0,
                "val/balanced_acc": val_bacc,
                "val/macro_f1": val_f1,
                "val/ckpt_score": ckpt_score,
                "val/n_unique_pred": n_unique_val,
                "val/collapse_warning": int(collapse_warning),
                "epoch": ep,
                "fold": fold_idx,
            },
            step=global_step,
        )

        if improved:
            best_score = ckpt_score
            best_val_auc = val_auc if not np.isnan(val_auc) else 0.0
            best_epoch = ep + 1
            best_val_preds = (vy.copy(), vp.copy())
            bad = 0
            ckpt_blob: dict = {
                "state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_auc": best_val_auc,
                "val_ckpt_score": best_score,
                "early_stop_metric": early_stop_metric,
                "fold": fold_idx,
            }
            if ema is not None:
                ckpt_blob["ema_state_dict"] = {k: v.clone() for k, v in ema.shadow.items()}
            torch.save(ckpt_blob, fold_dir / "best.pt")
        else:
            # Only count patience after min_epochs warmup
            if ep + 1 >= min_epochs:
                bad += 1
                if bad >= patience:
                    epoch_bar.write(
                        f"  Early stop at epoch {ep + 1} "
                        f"(no {early_stop_metric} improvement for {patience} epochs)"
                    )
                    break

    epoch_bar.close()
    print(
        f"  Best checkpoint: epoch {best_epoch}/{ep + 1}"
        f"  ({early_stop_metric}={best_score:.3f}  val_auc={best_val_auc:.3f})"
    )

    if ema is not None and (fold_dir / "best.pt").exists():
        try:
            _bckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
            _bckpt["ema_state_dict"] = {k: v.clone().cpu() for k, v in ema.shadow.items()}
            _bckpt["ema_epoch"] = ep + 1
            torch.save(_bckpt, fold_dir / "best.pt")
            print(f"  Saved final EMA shadow (after {ep + 1} epochs of accumulation).")
        except Exception as exc:
            print(f"  [warn] could not refresh EMA in best.pt: {exc}")

    # ── Evaluate on test set ──────────────────────────────────────────────
    print(f"  Evaluating best model on {len(test_ids)} test slides …")
    ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    # When EMA is enabled, ALWAYS use EMA weights for the final test eval.
    if ema_decay > 0 and "ema_state_dict" in ckpt:
        for name, p in model.named_parameters():
            if name in ckpt["ema_state_dict"]:
                p.data.copy_(ckpt["ema_state_dict"][name].to(device))
        print("  Using EMA weights for test evaluation.")

    # Test-time augmentation: average softmax probs over N augmented passes
    if tta_passes > 1 and _use_aug:
        print(f"  TTA: averaging predictions over {tta_passes} augmented passes …")
        ty, _tp_first = _collect_probs(model, test_loader, device, n_classes)
        tp_acc = _tp_first.copy()
        for _ in range(tta_passes - 1):
            # Build a TTA loader that applies augmentations during forward
            _, tp_extra = _collect_probs_with_aug(
                model, test_loader, device, n_classes, _augment,
            )
            tp_acc = tp_acc + tp_extra
        tp = tp_acc / tta_passes
    else:
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
            "best_epoch": best_epoch,
        },
        fold_dir / "test_predictions.pt",
    )

    _log_fold_end(
        ty, y_pred, tp, test_metrics, fold_idx, class_names,
        epoch_history, best_epoch=best_epoch, best_val_preds=best_val_preds,
    )

    n_unique_test = int(np.unique(y_pred).shape[0])
    if n_unique_test < 2:
        print(f"  [collapse] fold {fold_idx}: test set predicts only {np.unique(y_pred).tolist()}")

    return {
        "val_best_macro_auc": best_val_auc,   # actual AUROC at best checkpoint
        "val_best_score": best_score,          # checkpoint selection metric value
        "val_best_metric": early_stop_metric,
        "test_n_unique_pred": n_unique_test,
        "test": test_metrics,
    }


# ---------------------------------------------------------------------------
# Fold-end wandb logging — scalars + four matplotlib figures
# ---------------------------------------------------------------------------

def _log_fold_end(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict,
    fold_idx: int,
    class_names: list[str] | None,
    epoch_history: dict[str, list[float]] | None = None,
    best_epoch: int = 0,
    best_val_preds: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    try:
        import wandb

        if wandb.run is None:
            return

        names = class_names or [f"class_{i}" for i in range(y_prob.shape[1])]
        prefix = f"fold_{fold_idx}"

        n_unique_test = int(np.unique(y_pred).shape[0])
        log_dict: dict = {
            f"{prefix}/test_macro_f1": metrics["macro_f1"],
            f"{prefix}/test_balanced_acc": metrics["balanced_accuracy"],
            f"{prefix}/test_macro_auc": metrics["macro_auc_ovr"],
            f"{prefix}/test_weighted_f1": metrics.get("weighted_f1", float("nan")),
            f"{prefix}/best_epoch": best_epoch,
            f"{prefix}/test_n_unique_pred": n_unique_test,
            f"{prefix}/test_collapsed": int(n_unique_test < 2),
        }

        # Test — confusion matrix normalized (row %, recall emphasis)
        try:
            fig = _plot_confusion_matrix(y_true, y_pred, names, fold_idx)
            log_dict[f"{prefix}/test_confusion_matrix_normalized"] = _fig_to_wandb_image(fig)
        except Exception:
            pass

        # Test — confusion matrix raw counts
        try:
            fig = _plot_confusion_matrix_counts(
                y_true, y_pred, names, fold_idx,
                subtitle=f"Test set  (best epoch={best_epoch})",
            )
            log_dict[f"{prefix}/test_confusion_matrix_counts"] = _fig_to_wandb_image(fig)
        except Exception:
            pass

        # Val — confusion matrix at best epoch
        if best_val_preds is not None:
            vy, vp = best_val_preds
            vp_pred = vp.argmax(axis=1)
            try:
                fig = _plot_confusion_matrix_counts(
                    vy, vp_pred, names, fold_idx,
                    subtitle=f"Val set  (epoch={best_epoch})",
                )
                log_dict[f"{prefix}/val_confusion_matrix_counts"] = _fig_to_wandb_image(fig)
            except Exception:
                pass

        # ROC curves (one-vs-rest, test set)
        try:
            fig = _plot_roc_curves(y_true, y_prob, names, fold_idx)
            log_dict[f"{prefix}/roc_curves"] = _fig_to_wandb_image(fig)
        except Exception:
            pass

        # Per-class precision / recall / F1 bar chart (test set)
        try:
            fig = _plot_class_metrics(y_true, y_pred, names, fold_idx)
            log_dict[f"{prefix}/class_metrics"] = _fig_to_wandb_image(fig)
        except Exception:
            pass

        # Training curves with best-epoch marker
        if epoch_history and epoch_history.get("train_loss"):
            try:
                fig = _plot_training_curves(epoch_history, fold_idx, best_epoch=best_epoch)
                log_dict[f"{prefix}/training_curves"] = _fig_to_wandb_image(fig)
            except Exception:
                pass

        wandb.log(log_dict)
    except Exception:
        pass
