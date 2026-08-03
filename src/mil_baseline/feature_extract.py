from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mil_baseline.encoder import PatchEncoder, build_patch_encoder
from mil_baseline.manifest import load_labels_json, require_existing_slides
from mil_baseline.patching import collect_patch_locations, open_slide
from mil_baseline.utils import load_yaml, repo_root, resolve_path, seed_everything


@torch.inference_mode()
def extract_slide_features(
    slide_path: Path,
    out_path: Path,
    cfg: dict,
    patch_encoder: PatchEncoder,
    device: torch.device,
) -> dict:
    patch_cfg = cfg["patch"]
    tissue_cfg = cfg["tissue"]
    enc_cfg = cfg["encoder"]
    level = int(patch_cfg["level"])
    ps = int(patch_cfg["size"])
    st = int(patch_cfg["stride"])
    max_n = int(patch_cfg.get("max_patches_per_slide", 8000))
    bs = int(enc_cfg.get("batch_size", 64))

    with open_slide(slide_path) as slide:
        locs = collect_patch_locations(slide, level, ps, st, tissue_cfg)
    if len(locs) == 0:
        raise RuntimeError(f"No patches passed tissue filter for {slide_path}")
    _rng_seed = cfg.get("train", {}).get("seed")
    rng = np.random.default_rng(int(_rng_seed) if _rng_seed is not None else 42)
    if len(locs) > max_n:
        idx = rng.choice(len(locs), size=max_n, replace=False)
        locs = [locs[i] for i in sorted(idx.tolist())]

    feats_chunks: list[torch.Tensor] = []
    feat_dim = int(patch_encoder.feat_dim)
    with open_slide(slide_path) as slide:
        buf: list[np.ndarray] = []
        for x0, y0 in tqdm(locs, desc=f"feats {slide_path.name}", leave=False):
            tile = slide.read_region((x0, y0), level, (ps, ps)).convert("RGB")
            buf.append(np.asarray(tile, dtype=np.uint8))
            if len(buf) >= bs:
                f = patch_encoder.encode_numpy_list(buf)
                feats_chunks.append(f.float())
                buf.clear()
        if buf:
            f = patch_encoder.encode_numpy_list(buf)
            feats_chunks.append(f.float())

    feats = torch.cat(feats_chunks, dim=0).float()
    coords = np.array(locs, dtype=np.int32)
    meta = {
        "slide_path": str(slide_path),
        "level": level,
        "patch_size": ps,
        "stride": st,
        "n_patches": int(feats.shape[0]),
        "feat_dim": feat_dim,
        "encoder": str(enc_cfg.get("name", "resnet50")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": feats,
            "coords": torch.from_numpy(coords),
            "meta": meta,
        },
        out_path,
    )
    return meta


def run_extract(config_path: Path, overwrite: bool = False) -> None:
    cfg = load_yaml(config_path)
    root = repo_root()
    slides_dir = resolve_path(root, cfg["slides_dir"])
    labels_path = resolve_path(root, cfg["labels_json"])
    feats_dir = resolve_path(root, cfg["features_dir"])
    _seed = cfg.get("train", {}).get("seed")
    if _seed is not None:
        seed_everything(int(_seed))

    df = load_labels_json(labels_path, slides_dir, int(cfg["num_classes"]))
    df = require_existing_slides(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_name = cfg["encoder"]["name"]
    print(f"\n[extract] encoder={encoder_name}  device={device}")
    print(f"[extract] slides_dir : {slides_dir}")
    print(f"[extract] features_dir: {feats_dir}")

    patch_encoder = build_patch_encoder(cfg["encoder"], device)

    # Separate cached vs. to-process so the slide bar shows real work remaining
    todo = [row for _, row in df.iterrows()
            if overwrite or not (feats_dir / f"{row['slide_id']}.pt").exists()]
    n_cached = len(df) - len(todo)
    if n_cached:
        print(f"[extract] {n_cached}/{len(df)} slides already cached — skipping.")
    print(f"[extract] Extracting features for {len(todo)} slide(s) …\n")

    slide_bar = tqdm(todo, desc="Slides", unit="slide", dynamic_ncols=True)
    for row in slide_bar:
        sid = row["slide_id"]
        out_pt = feats_dir / f"{sid}.pt"
        slide_bar.set_description(f"Slides  [{sid}]")
        meta = extract_slide_features(
            Path(row["abs_path"]), out_pt, cfg, patch_encoder, device
        )
        slide_bar.write(
            f"  ✓ {sid}  patches={meta['n_patches']}  dim={meta['feat_dim']}"
        )

    print(f"\n[extract] Done. Features saved to {feats_dir}/")
