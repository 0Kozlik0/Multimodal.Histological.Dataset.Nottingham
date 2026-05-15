# Baseline
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# OpenSlide must be installed on the system for .mrxs (e.g. macOS: brew install openslide)
export PYTHONPATH=src
python -m mil_baseline.cli extract --config configs/baseline.yaml
python -m mil_baseline.cli train-cv --config configs/baseline.yaml
```

# Prepare data from `.mnt/nfs-data` for MIL (manifest-only)
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m src.data_preprocessing_utils.prepare_mil_dataset \
  --labels-json .mnt/nfs-data/labels.json \
  --slides-root .mnt/nfs-data \
  --allowed-years 2023 2024 2025 \
  --output-dir data/prepared_mil_dataset
```

This writes:
- `data/prepared_mil_dataset/labels_prepared.json` (only labeled slides found under 2023/2024/2025)
- `data/prepared_mil_dataset/prepare_report.json` (found/missing/ambiguous details)
- `data/prepared_mil_dataset/baseline_prepared.yaml` (config pointing `mil_baseline` to prepared labels)

Train with prepared manifest:
```bash
export PYTHONPATH=src
python -m mil_baseline.cli extract --config configs/baseline_prepared.yaml
python -m mil_baseline.cli train-cv --config configs/baseline_prepared.yaml
```

Claude Code / large refactors: see [`CLAUDE_CODE_MIL_PROMPT.md`](CLAUDE_CODE_MIL_PROMPT.md) and [`.claude/CLAUDE.md`](.claude/CLAUDE.md).

If `extract` fails with `FileNotFoundError` for paths like `2024/slide-....mrxs`, the config `slides_dir` must point at the directory that contains those year folders (default in `configs/baseline_prepared.yaml` is `.mnt/nfs-data`). Run commands from the repository root, ensure the NFS mount exists at `.mnt/nfs-data`, or adjust `slides_dir` to your actual slide root. Re-run `prepare_mil_dataset` after changing storage layout.

Dataset validity notebook:
```bash
jupyter notebook notebooks/dataset_validity_check.ipynb
```
The notebook checks schema, class counts, duplicates, file existence, and OpenSlide readability spot checks.
OpenSlide system library is required for `.mrxs` readability checks.

# UNI2
```bash
pip install -r requirements.txt
export HF_TOKEN=...   # after HF access is granted
export PYTHONPATH=src
python -m mil_baseline.cli extract --config configs/extract_uni2.yaml
python -m mil_baseline.cli train-cv --config configs/extract_uni2.yaml
```

# CONCH
```bash
pip install -r requirements-conch.txt
export HF_TOKEN=...
python -m mil_baseline.cli extract --config configs/extract_conch.yaml
python -m mil_baseline.cli train-cv --config configs/extract_conch.yaml
```


# Features are shared between ABMIL and CLAM-SB, so extract only once
python -m mil_baseline.cli extract --config configs/baseline_prepared.yaml


# 1. Activate environment (once per shell session)
source .venv/bin/activate
export PYTHONPATH=src

# 2. Log in to wandb (once per machine — skip if WANDB_API_KEY is already set)
wandb login

# 3. ABMIL — features already extracted, so just train
python -m mil_baseline.cli train-cv \
  --config configs/baseline_prepared.yaml \
  --export-publication

# 4. CLAM-SB — reuses the same cached features, separate results dir
python -m mil_baseline.cli train-cv \
  --config configs/benchmark_clam_sb.yaml \
  --export-publication

