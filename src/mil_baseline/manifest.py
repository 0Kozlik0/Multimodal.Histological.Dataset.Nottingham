from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SlideRecord:
    slide_id: str
    rel_path: str
    label_1indexed: int
    y: int
    abs_path: Path
    patient_id: str = ""


def load_labels_json(path: Path, slides_dir: Path, num_classes: int) -> pd.DataFrame:
    """Load labels JSON into a DataFrame.

    Expects records with keys ``image`` and ``label`` (1-indexed).  If a
    ``patient_id`` key is present it is used for patient-level CV; otherwise
    ``slide_id`` is used as the patient identity (one-slide-per-patient
    assumption) and a warning is emitted.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: list[dict] = []
    _warned_no_patient = False
    for r in rows:
        name = r["image"]
        lab = int(r["label"])
        if lab < 1 or lab > num_classes:
            raise ValueError(f"Label {lab} out of range 1..{num_classes} for {name}")
        sid = Path(name).stem
        abs_p = slides_dir / name
        if "patient_id" in r:
            pid = str(r["patient_id"])
        else:
            pid = sid
            if not _warned_no_patient:
                warnings.warn(
                    "labels JSON has no 'patient_id' field — using slide_id as "
                    "patient_id.  Patient-level CV treats each slide as a distinct "
                    "patient.  Add 'patient_id' to labels JSON for true patient-grouped splits.",
                    stacklevel=2,
                )
                _warned_no_patient = True
        out.append(
            {
                "slide_id": sid,
                "patient_id": pid,
                "filename": name,
                "label": lab,
                "y": lab - 1,
                "abs_path": abs_p,
                "exists": abs_p.exists(),
            }
        )
    return pd.DataFrame(out)


def require_existing_slides(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.loc[~df["exists"], "filename"].tolist()
    if missing:
        raise FileNotFoundError(
            "Missing slide files (place .mrxs under data/slides): "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    return df.reset_index(drop=True)
