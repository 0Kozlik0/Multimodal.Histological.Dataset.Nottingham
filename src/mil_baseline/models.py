from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, D) → attention_weights (N,), bag_repr (D,)"""
        a = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        a = self.w(a).squeeze(-1)
        w = F.softmax(a, dim=0)
        bag = (w.unsqueeze(-1) * x).sum(dim=0)
        return w, bag


class ABMIL(nn.Module):
    """Multiclass ABMIL: Ilse et al. gated attention + classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        attention_hidden: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.inst = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn = GatedAttention(hidden_dim, attention_hidden)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """feats: (N, D) → logits (C,), attention (N,)"""
        h = self.inst(feats)
        w, bag = self.attn(h)
        logits = self.head(bag)
        return logits, w


class MaxPoolMIL(nn.Module):
    """Lower bound: per-instance linear then max-pool over instances."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.inst = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits_inst = self.inst(feats)
        logits, _ = logits_inst.max(dim=0)
        return logits, None


class MeanLinearMIL(nn.Module):
    """Smallest possible MIL: mean-pool patches → LayerNorm → linear classifier.

    Args:
        use_layernorm: Apply (non-affine) LayerNorm to the pooled vector.
            Mirrors the StandardScaler step in the LogisticRegression linear-probe
            baseline; without it, ResNet50 features have varying per-dim scales
            that a single linear layer can't compensate for in one fold of 27 slides.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.norm = (
            nn.LayerNorm(input_dim, elementwise_affine=False)
            if use_layernorm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, None]:
        pooled = feats.mean(dim=0)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.head(pooled), None


class MeanLinearOrdinalMIL(nn.Module):
    """MeanLinear with CORAL ordinal output (Niu et al. 2016)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_classes = num_classes
        self.norm     = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.weight   = nn.Linear(input_dim, 1, bias=False)
        # K-1 ordered biases, initialised in increasing order so cumulative probs decrease
        self.biases   = nn.Parameter(torch.linspace(-1.0, 1.0, num_classes - 1))

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, None]:
        pooled = self.norm(feats.mean(dim=0))
        pooled = self.dropout(pooled)
        z = self.weight(pooled)                # scalar
        cum = torch.sigmoid(z + self.biases)   # (K-1,) — P(y > k)
        # Convert cumulative to per-class probabilities
        eps = 1e-7
        ones = torch.ones(1, device=cum.device)
        p_geq = torch.cat([ones, cum])         # P(y >= k): ones for k=0
        p_gt  = torch.cat([cum, torch.zeros(1, device=cum.device)])  # P(y > k): 0 for last
        probs = (p_geq - p_gt).clamp(min=eps)  # (K,)
        logits = probs.log()
        return logits, None


class MeanMaxLinearMIL(nn.Module):
    """Concatenate mean + max pooling on patch features, then linear classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_mean = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.norm_max  = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.dropout   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head      = nn.Linear(input_dim * 2, num_classes)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, None]:
        pooled_mean = self.norm_mean(feats.mean(dim=0))
        pooled_max  = self.norm_max(feats.max(dim=0).values)
        pooled      = torch.cat([pooled_mean, pooled_max], dim=-1)
        pooled      = self.dropout(pooled)
        return self.head(pooled), None


class CLAM_SB(nn.Module):
    """CLAM single-branch (Lu et al., 2021): gated attention + optional instance supervision.

    Reference: https://github.com/mahmoodlab/CLAM
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        attention_hidden: int = 128,
        dropout: float = 0.25,
        k_sample: int = 8,
    ):
        super().__init__()
        self.k_sample = k_sample
        self.num_classes = num_classes
        self.inst = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn = GatedAttention(hidden_dim, attention_hidden)
        self.head = nn.Linear(hidden_dim, num_classes)
        # One binary classifier per class (in-class vs. rest)
        self.inst_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(num_classes)]
        )

    def forward_with_h(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """feats: (N, D) → bag_logits (C,), attention (N,), projected_h (N, hidden)"""
        h = self.inst(feats)
        w, bag = self.attn(h)
        logits = self.head(bag)
        return logits, w, h

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats: (N, D) → bag_logits (C,), attention (N,)"""
        logits, w, _ = self.forward_with_h(feats)
        return logits, w

    def instance_loss(
        self, h: torch.Tensor, attn: torch.Tensor, label: int
    ) -> torch.Tensor:
        """Instance-level supervision: binary CE on top-k (pos) and bottom-k (neg) patches."""
        k = min(self.k_sample, h.shape[0] // 2)
        if k < 1:
            return torch.tensor(0.0, device=h.device)
        _, top_idx = attn.topk(k, largest=True)
        _, bot_idx = attn.topk(k, largest=False)
        all_h = h[torch.cat([top_idx, bot_idx])]
        all_y = torch.cat(
            [
                torch.ones(k, dtype=torch.long, device=h.device),
                torch.zeros(k, dtype=torch.long, device=h.device),
            ]
        )
        return F.cross_entropy(self.inst_classifiers[label](all_h), all_y)


def build_mil(name: str, input_dim: int, num_classes: int, mil_cfg: dict) -> nn.Module:
    name = name.lower()
    hidden = int(mil_cfg.get("hidden_dim", 256))
    attn_h = int(mil_cfg.get("attention_hidden", 128))
    drop = float(mil_cfg.get("dropout", 0.25))
    if name == "abmil":
        return ABMIL(input_dim, num_classes, hidden, attn_h, drop)
    if name == "maxpool":
        return MaxPoolMIL(input_dim, num_classes, hidden, drop)
    if name in ("mean_linear", "mean"):
        return MeanLinearMIL(input_dim, num_classes, drop)
    if name in ("meanmax_linear", "meanmax"):
        return MeanMaxLinearMIL(input_dim, num_classes, drop)
    if name in ("mean_ordinal", "ordinal"):
        return MeanLinearOrdinalMIL(input_dim, num_classes, drop)
    if name == "clam_sb":
        k = int(mil_cfg.get("k_sample", 8))
        return CLAM_SB(input_dim, num_classes, hidden, attn_h, drop, k)
    raise ValueError(f"Unknown MIL model '{name}' — choose abmil, clam_sb, maxpool, or mean_linear")
