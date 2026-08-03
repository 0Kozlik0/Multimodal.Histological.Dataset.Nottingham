from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def _maybe_hf_login(enc_cfg: dict) -> None:
    token = (
        enc_cfg.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        pass


def _infer_feat_dim(
    model: nn.Module,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        z = forward_fn(dummy)
    return int(z.shape[-1])


def build_uni_v1(
    enc_cfg: dict, device: torch.device
) -> tuple[nn.Module, int, Callable[[torch.Tensor], torch.Tensor]]:
    import timm

    _maybe_hf_login(enc_cfg)
    timm_kw = {
        "dynamic_img_size": True,
        "num_classes": 0,
        "init_values": 1e-5,
    }
    ckpt = enc_cfg.get("checkpoint_path")
    if ckpt:
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            patch_size=16,
            **timm_kw,
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kw)
    model = model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.float() / 255.0
        x = x.permute(0, 3, 1, 2).to(device)
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bicubic", align_corners=False
        )
        return (x - mean) / std

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    feat_dim = _infer_feat_dim(model, forward_fn, device)
    return model, feat_dim, preprocess


def build_uni2_h(
    enc_cfg: dict, device: torch.device
) -> tuple[nn.Module, int, Callable[[torch.Tensor], torch.Tensor]]:
    import timm

    _maybe_hf_login(enc_cfg)
    timm_kw = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    ckpt = enc_cfg.get("checkpoint_path")
    if ckpt:
        model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kw)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kw
        )
    model = model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.float() / 255.0
        x = x.permute(0, 3, 1, 2).to(device)
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bicubic", align_corners=False
        )
        return (x - mean) / std

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    feat_dim = _infer_feat_dim(model, forward_fn, device)
    return model, feat_dim, preprocess


def build_conch(
    enc_cfg: dict, device: torch.device
) -> tuple[nn.Module, int, Callable[[list[np.ndarray]], torch.Tensor]]:
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as e:
        raise ImportError(
            "CONCH requires the Mahmood Lab package. Install with:\n"
            "  pip install git+https://github.com/Mahmoodlab/CONCH.git\n"
            "Request model access at https://huggingface.co/MahmoodLab/CONCH and set HF_TOKEN."
        ) from e

    _maybe_hf_login(enc_cfg)
    token = (
        enc_cfg.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    ckpt = (
        enc_cfg.get("conch_checkpoint")
        or enc_cfg.get("checkpoint_path")
        or "hf_hub:MahmoodLab/conch"
    )
    kwargs: dict = {}
    if isinstance(ckpt, str) and ckpt.startswith("hf_hub") and token:
        kwargs["hf_auth_token"] = token
    model, pil_preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", ckpt, **kwargs
    )
    model = model.to(device).eval()

    def preprocess_batch(patches: list[np.ndarray]) -> torch.Tensor:
        from PIL import Image

        ts = []
        for p in patches:
            ts.append(pil_preprocess(Image.fromarray(p)))
        return torch.stack(ts, dim=0).to(device)

    with torch.inference_mode():
        z = model.encode_image(
            preprocess_batch([np.zeros((224, 224, 3), dtype=np.uint8)]),
            proj_contrast=False,
            normalize=False,
        )
    feat_dim = int(z.shape[-1])
    return model, feat_dim, preprocess_batch
