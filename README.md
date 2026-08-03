# Multimodal Histological Dataset for the Nottingham Grading of Breast Cancer — technical validation code

Code and baselines accompanying an expert-annotated **Nottingham histological grade** whole-slide image (WSI) cohort for breast carcinoma. This repository supports a *Scientific Data*–style data descriptor: it does **not** redistribute the raw `.mrxs` slides, but provides the pipelines used to prepare labels, extract patch features, run leakage-safe cross-validation, and reproduce the technical validation experiments.


|                  |                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| **Task**         | 3-class Nottingham grade (1 / 2 / 3) at slide level; mitotic vs non-mitotic at ROI level               |
| **Cohort**       | 42 WSIs (Grade 1: 13, Grade 2: 19, Grade 3: 10)                                                        |
| **Protocol**     | Frozen patch encoders + lightweight classifiers; patient/slide-grouped CV; permutation tests           |
| **Code license** | [MIT](LICENSE)                                                                                         |
| **Repository**   | [github.com/0Kozlik0/Custom.Dataset.Nottingham](https://github.com/0Kozlik0/Custom.Dataset.Nottingham) |


> **Data availability.** Whole-slide images and the released label file are part of the dataset publication (repository / archive link to be filled upon acceptance). Point `slides_dir` / `--slides-root` at your local copy of the archive.

---

## Why this repository exists

Validating a expert-annotated WSI cohort on state-of-the-art benchmarks architectures. Here we show that:

1. **Labels and files are complete and readable** (schema, paths, OpenSlide).
2. **Slide-level grade is recoverable** under weakly supervised multiple-instance learning (MIL) with frozen encoders and a leakage-safe CV protocol.
3. **Graded morphology is recoverable at cellular scale** via an independent mitotic-figure ROI task on pathologist annotations.

Primary metrics are **macro F1** and **balanced accuracy**; AUROC is supplementary only (a collapsed single-class predictor can still look strong on AUROC)

Slide-level grades used in the published baselines are those in the released `labels.json` (**no rewriting** of grades from structure-level annotations).

---

## Repository layout

```
├── configs/                  # YAML configs for extract / train / linear-probe / mitosis
├── docker/                   # GPU Dockerfile + run helpers (OpenSlide + editable install)
├── notebooks/                # Dataset integrity notebook
├── scripts/                  # Benchmark table / ensemble helpers
├── src/
│   ├── mil_baseline/         # Patching, encoders, MIL models, CV, mitosis ROI
│   └── data_preprocessing_utils/  # Manifest prep, annotation extract
├── requirements.txt
├── requirements-conch.txt    # Extra deps for CONCH
├── pyproject.toml
└── LICENSE
```

---

## Methods and citations

Architectures and pretrained encoders used in this codebase (cite the original works when reusing these components):


| Component                         | Role in this repo                                                 | Reference                                                                                                                                                                                                                                       |
| --------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ABMIL** (gated attention)       | Attention-based MIL aggregator                                    | Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018. [PMLR](https://proceedings.mlr.press/v80/ilse18a.html)                                                                                                               |
| **CLAM-SB**                       | Attention MIL + instance supervision                              | Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide images*, Nat Biomed Eng 2021. [doi:10.1038/s41551-020-00682-w](https://doi.org/10.1038/s41551-020-00682-w)                                              |
| **MeanLinear MIL**                | Mean-pool + LayerNorm + linear head (low-capacity bag classifier) | This repository (feasibility baseline matched to small *n*)                                                                                                                                                                                     |
| **MaxPool MIL**                   | Per-instance linear → max over patches                            | Common MIL baseline; compared under the same training recipe                                                                                                                                                                                    |
| **ResNet50**                      | Frozen ImageNet patch encoder                                     | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)                                                                                                          |
| **ImageNet**                      | Pretraining data for ResNet50                                     | Deng et al., *ImageNet: A large-scale hierarchical image database*, CVPR 2009. [doi:10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)                                                                                       |
| **UNI** (ViT-L/16)                | Frozen pathology foundation encoder                               | Chen et al., *Towards a general-purpose foundation model for computational pathology*, Nat Med 2024. [doi:10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3) · [HF: MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| **UNI2-h** (ViT-H/14)             | Frozen pathology foundation encoder                               | Chen et al. (UNI family) · [HF: MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)                                                                                                                                                    |
| **CONCH** (ViT-B/16)              | Frozen pathology vision–language encoder                          | Lu et al., *A visual-language foundation model for computational pathology*, Nat Med 2024. [doi:10.1038/s41591-024-02856-4](https://doi.org/10.1038/s41591-024-02856-4) · [HF: MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)       |
| **Logistic regression** (sklearn) | Linear probe / mitosis ROI classifier                             | Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011. [jmlr.org](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)                                                                                                    |


MIL overview (problem setting): Carbonneau et al., *Multiple instance learning: A survey of problem characteristics and applications*, Pattern Recognition 2018. [doi:10.1016/j.patcog.2017.10.009](https://doi.org/10.1016/j.patcog.2017.10.009).

Foundation models (**UNI**, **UNI2-h**, **CONCH**) require accepting the model licenses on Hugging Face and setting `HF_TOKEN` (see `[.env.sample](.env.sample)`).

---

## Requirements

- **Python** ≥ 3.10  
- **PyTorch** ≥ 2.1 (see `requirements.txt`)  
- System **OpenSlide** library (for `.mrxs` reads)  
- Optional: NVIDIA GPU; Docker image under `docker/`  
- Optional: `wandb` for training logs

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # registers mil_baseline + CLI entry points
export PYTHONPATH=src             # alternative to editable install for scripts
```

OpenSlide examples: `brew install openslide` (macOS) or `apt install libopenslide-dev` (Debian/Ubuntu). Docker users: see `[docker/Dockerfile](docker/Dockerfile)`.

---

## Quick start — reproduce technical validation

Commands assume the slide archive is available (default path `.mnt/nfs-data` with year folders `2023/`, `2024/`, `2025/`). Adjust paths in configs or CLI flags as needed.

### 1. Prepare slide-level manifest

```bash
python -m src.data_preprocessing_utils.prepare_mil_dataset \
  --labels-json .mnt/nfs-data/labels.json \
  --slides-root .mnt/nfs-data \
  --allowed-years 2023 2024 2025 \
  --output-dir data/prepared_mil_dataset
```

Outputs: `labels_prepared.json`, `prepare_report.json`, and a prepared YAML config.

### 2. Dataset integrity

```bash
jupyter notebook notebooks/dataset_validity_check.ipynb
```

Checks schema, class counts, duplicates, file existence, and OpenSlide readability.

### 3. Slide-level MIL

```bash
# Extract ResNet50 patch features (once)
python -m mil_baseline.cli extract --config configs/baseline_prepared.yaml

# MeanLinear MIL — 5-fold × 3 seeds, fixed 80-epoch recipe
python -m mil_baseline.cli train-cv \
  --config configs/publication_meanlinear.yaml \
  --export-publication

# Optional: other aggregators (same features)
python -m mil_baseline.cli train-cv --config configs/publication_abmil.yaml --export-publication
python -m mil_baseline.cli train-cv --config configs/publication_clam_sb.yaml --export-publication
python -m mil_baseline.cli train-cv --config configs/publication_maxpool.yaml --export-publication

# Linear probe (no MIL training)
python -m mil_baseline.cli linear-probe --config configs/baseline_prepared.yaml
```

**Headline results** (frozen ResNet50, 5-fold × 3 seeds):


| Model                 | Macro F1      | Balanced acc. | AUROC         | perm. *p* |
| --------------------- | ------------- | ------------- | ------------- | --------- |
| MeanLinear MIL        | 0.499 ± 0.170 | 0.544 ± 0.149 | 0.708 ± 0.151 | < 0.001   |
| MaxPool MIL           | 0.363 ± 0.218 | 0.403 ± 0.217 | 0.566 ± 0.213 | 0.050     |
| ABMIL                 | 0.395 ± 0.096 | 0.439 ± 0.118 | 0.673 ± 0.139 | 0.004     |
| CLAM-SB               | 0.419 ± 0.126 | 0.453 ± 0.154 | 0.660 ± 0.143 | < 0.001   |
| Linear probe (LogReg) | 0.533 ± 0.082 | 0.586 ± 0.090 | 0.729 ± 0.133 | —         |


Aggregate multiple result dirs:

```bash
python scripts/build_benchmark_table.py \
  --results-dir results_publication_meanlinear:MeanLinear \
  --results-dir results_publication_abmil:ABMIL \
  --results-dir results_publication_clam_sb:CLAM-SB \
  --results-dir results_publication_maxpool:MaxPool \
  --linear-probe results_prepared_linear/cv_summary_linear_probe.json \
  --out publication_outputs/benchmark_table.md
```

### 4. Encoder comparison (UNI / UNI2-h / CONCH)

1. Accept HF licenses for [UNI](https://huggingface.co/MahmoodLab/UNI), [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), [CONCH](https://huggingface.co/MahmoodLab/CONCH).
2. `pip install -r requirements-conch.txt` and set `HF_TOKEN` (copy `.env.sample` → `.env`).
3. Extract + train:

```bash
set -a && source .env && set +a
export PYTHONPATH=src

python -m mil_baseline.cli extract --config configs/extract_uni.yaml
python -m mil_baseline.cli extract --config configs/extract_uni2.yaml
python -m mil_baseline.cli extract --config configs/extract_conch.yaml

python -m mil_baseline.cli train-cv --config configs/publication_meanlinear_uni.yaml --export-publication
python -m mil_baseline.cli train-cv --config configs/publication_meanlinear_uni2h.yaml --export-publication
python -m mil_baseline.cli train-cv --config configs/publication_meanlinear_conch.yaml --export-publication
python -m mil_baseline.cli train-cv --config configs/publication_meanlinear_resnet50.yaml --export-publication
```


| Encoder  | Feature dim | Source                      |
| -------- | ----------- | --------------------------- |
| ResNet50 | 2048        | torchvision `IMAGENET1K_V2` |
| UNI      | 1024        | MahmoodLab/uni              |
| UNI2-h   | 1536        | MahmoodLab/UNI2-h           |
| CONCH    | 512         | MahmoodLab/conch            |


### 5. Structure-level mitotic-figure ROI benchmark

```bash
python -m src.data_preprocessing_utils.extract_mitosis_annotations \
  --zip-path .mnt/nfs-data/2025/Anotacie_KMK_zaznacene250328.zip \
  --output-dir data/annotations/mitosis_kmk

python -m mil_baseline.cli mitosis-extract --config configs/mitosis_roi.yaml
python -m mil_baseline.cli mitosis-train-cv --config configs/mitosis_roi.yaml --export-publication
```

**Result** (frozen ResNet50 + class-balanced logistic regression, 5-fold slide-grouped CV, 380 ROIs / 12 slides): macro F1 **0.574 ± 0.037**, balanced accuracy **0.573 ± 0.034**, AUROC **0.676 ± 0.035**, permutation *p* = **0.002**.

> **Reproducer note.** MIRAX `.mrxs` files expose `openslide.bounds-x/y`. QuPath coordinates are relative to that scanned sub-region; the ROI extractor adds the offset before `read_region()`. See the technical validation report.

---

## Training protocol (summary)


| Setting    | Slide-level MIL (publication configs)                    | Mitosis ROI                                    |
| ---------- | -------------------------------------------------------- | ---------------------------------------------- |
| Encoder    | Frozen (ResNet50 / UNI / UNI2-h / CONCH)                 | Frozen ResNet50                                |
| Classifier | MeanLinear / MaxPool / ABMIL / CLAM-SB                   | Logistic regression (`C=1`, `lbfgs`, balanced) |
| Epochs     | Fixed **80** (no early stopping)                         | N/A (convex fit to convergence)                |
| Optimizer  | Adam, lr 10^{-3}, weight decay 0.01, cosine LR, EMA 0.99 | —                                              |
| CV         | 5-fold × 3 seeds, patient/slide-grouped                  | 5-fold, slide-grouped                          |
| Patch size | 256×256 (level 0), ≤8000 / slide                         | 128×128 (level 0)                              |


Full YAML configs live under `[configs/](configs/)`.

---

## License

- **Source code** in this repository is released under the [MIT License](LICENSE).  
- **Third-party model weights** (UNI, UNI2-h, CONCH, ImageNet-pretrained ResNet50) remain under their respective licenses; obtain access and cite as required by the model providers. 

---

## How to cite

```bibtex
[Will be added after publishing of the paper]
```

Please also cite the original method papers in the table above when comparing against or extending those architectures.

---

## Contact

Questions about the code or technical validation: **Patrik Kozlik** (`patrik.kozlik@gmail.com`).