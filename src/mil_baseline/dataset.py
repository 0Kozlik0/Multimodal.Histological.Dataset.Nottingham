from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class SlideFeatureDataset(Dataset):
    def __init__(
        self,
        slide_ids: list[str],
        labels: list[int],
        features_dir: Path,
        patient_ids: list[str] | None = None,
    ):
        self.slide_ids = slide_ids
        self.labels = labels
        self.features_dir = features_dir
        # Fall back to slide_id when patient_id not provided
        self.patient_ids = patient_ids if patient_ids is not None else slide_ids

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.slide_ids[idx]
        y = self.labels[idx]
        pid = self.patient_ids[idx]
        obj = torch.load(
            self.features_dir / f"{sid}.pt",
            map_location="cpu",
            weights_only=False,
        )
        feats = obj["features"].float()
        return {"features": feats, "y": y, "slide_id": sid, "patient_id": pid}
