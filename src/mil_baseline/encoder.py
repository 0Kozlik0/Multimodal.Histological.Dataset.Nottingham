from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from mil_baseline.foundation import build_conch, build_uni2_h, build_uni_v1

EncoderName = Literal["resnet50", "uni", "uni2_h", "uni2-h", "conch"]


def imagenet_norm_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std


def patches_to_tensor(batch_rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """batch_rgb: N x H x W x 3 uint8 on CPU -> float NCHW on device (ImageNet norm, no resize)."""
    x = batch_rgb.float() / 255.0
    x = x.permute(0, 3, 1, 2).to(device)
    mean, std = imagenet_norm_tensors(device)
    return (x - mean) / std


def build_resnet50(
    weights: str, device: torch.device
) -> tuple[nn.Module, int, nn.Module]:
    w = getattr(ResNet50_Weights, weights)
    m = models.resnet50(weights=w)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    m = m.to(device).eval()
    return m, feat_dim, m


@dataclass
class PatchEncoder:
    """Frozen encoder: list of RGB uint8 HWC numpy arrays -> N x D features on CPU."""

    name: str
    model: nn.Module
    feat_dim: int
    device: torch.device
    preprocess_tensor: Callable[[torch.Tensor], torch.Tensor] | None
    is_conch: bool
    conch_preprocess: Callable[[list[np.ndarray]], torch.Tensor] | None = None

    @torch.inference_mode()
    def encode_numpy_list(self, patches: list[np.ndarray]) -> torch.Tensor:
        if self.is_conch:
            assert self.conch_preprocess is not None
            x = self.conch_preprocess(patches)
            out = self.model.encode_image(x, proj_contrast=False, normalize=False)
            return out.cpu()

        assert self.preprocess_tensor is not None
        t = torch.from_numpy(np.stack(patches, axis=0))
        x = self.preprocess_tensor(t)
        out = self.model(x)
        if out.dim() == 4:
            out = out.squeeze(-1).squeeze(-1)
        return out.cpu()


def build_patch_encoder(enc_cfg: dict, device: torch.device) -> PatchEncoder:
    raw = str(enc_cfg.get("name", "resnet50")).lower().replace("-", "_")
    if raw in ("uni_v1", "uni_v"):
        raw = "uni"

    if raw == "resnet50":
        weights = enc_cfg.get("weights", "IMAGENET1K_V2")
        ckpt = enc_cfg.get("checkpoint_path")
        if ckpt:
            weight_src = f"custom checkpoint ({ckpt})"
        else:
            weight_src = f"ImageNet pretrained ({weights})"
        print(f"[encoder] Loading ResNet50 — {weight_src} …")
        model, feat_dim, _ = build_resnet50(weights, device)
        if not ckpt:
            try:
                w_obj = getattr(ResNet50_Weights, weights)
                acc1 = w_obj.meta["_metrics"]["ImageNet-1K"]["acc@1"]
                from torch.hub import get_dir as _hub_dir
                import os as _os
                cache_file = _os.path.join(_hub_dir(), "checkpoints", _os.path.basename(w_obj.url))
                cached = "✓ cached" if _os.path.exists(cache_file) else "downloading"
                print(f"[encoder] ResNet50 weights: ImageNet acc@1={acc1}%  ({cached}: {cache_file})")
            except Exception:
                pass
        _print_encoder_summary("resnet50", feat_dim, weight_src, device)

        def _preprocess(t_nhwc: torch.Tensor) -> torch.Tensor:
            return patches_to_tensor(t_nhwc, device)

        return PatchEncoder(
            name="resnet50",
            model=model,
            feat_dim=feat_dim,
            device=device,
            preprocess_tensor=_preprocess,
            is_conch=False,
        )

    if raw == "uni":
        ckpt = enc_cfg.get("checkpoint_path")
        weight_src = f"local checkpoint ({ckpt})" if ckpt else "HuggingFace (MahmoodLab/UNI)"
        print(f"[encoder] Loading UNI (ViT-L/16) — {weight_src} …")
        model, feat_dim, pre = build_uni_v1(enc_cfg, device)
        _print_encoder_summary("uni", feat_dim, weight_src, device)
        return PatchEncoder(
            name="uni",
            model=model,
            feat_dim=feat_dim,
            device=device,
            preprocess_tensor=pre,
            is_conch=False,
        )

    if raw in ("uni2_h", "uni2"):
        ckpt = enc_cfg.get("checkpoint_path")
        weight_src = f"local checkpoint ({ckpt})" if ckpt else "HuggingFace (MahmoodLab/UNI2-h)"
        print(f"[encoder] Loading UNI2-h (ViT-H/14) — {weight_src} …")
        model, feat_dim, pre = build_uni2_h(enc_cfg, device)
        _print_encoder_summary("uni2_h", feat_dim, weight_src, device)
        return PatchEncoder(
            name="uni2_h",
            model=model,
            feat_dim=feat_dim,
            device=device,
            preprocess_tensor=pre,
            is_conch=False,
        )

    if raw == "conch":
        ckpt = enc_cfg.get("conch_checkpoint") or enc_cfg.get("checkpoint_path")
        weight_src = f"local checkpoint ({ckpt})" if ckpt else "HuggingFace (MahmoodLab/CONCH)"
        print(f"[encoder] Loading CONCH (ViT-B/16) — {weight_src} …")
        model, feat_dim, pre = build_conch(enc_cfg, device)
        _print_encoder_summary("conch", feat_dim, weight_src, device)
        return PatchEncoder(
            name="conch",
            model=model,
            feat_dim=feat_dim,
            device=device,
            preprocess_tensor=None,
            is_conch=True,
            conch_preprocess=pre,
        )

    raise ValueError(
        f"Unknown encoder name '{enc_cfg.get('name')}'. "
        "Use one of: resnet50, uni, uni2_h, conch."
    )


def _print_encoder_summary(
    name: str, feat_dim: int, weight_src: str, device: torch.device
) -> None:
    print(
        f"[encoder] ✓ {name}  feat_dim={feat_dim}  "
        f"weights={weight_src}  frozen=True  device={device}"
    )


def build_encoder(
    name: str, weights: str, device: torch.device
) -> tuple[nn.Module, int]:
    """Backward-compatible ResNet-only API for tests and simple callers."""
    enc_cfg = {"name": name, "weights": weights}
    pe = build_patch_encoder(enc_cfg, device)
    return pe.model, pe.feat_dim
