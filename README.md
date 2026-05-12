# Face De-Identification: Its Attacks and Defenses

A three-step pipeline that implements and evaluates face de-identification methods, CNN-based re-identification attacks, and differential-privacy defenses on the AT&T (ORL) face database.

## Overview

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `step1_deidentify.py` | Apply NP-Pix (pixelization) and NP-Blur (Gaussian blur) to all images; output quality grids and MSE/SSIM table |
| 2 | `step2_attack.py` | Train a CNN_ATT attacker (McPherson 2016) for each de-identification condition; report top-1 / top-5 accuracy |
| 3 | `step3_dp_defense.py` | Apply DP-Pix / DP-Blur (Fan 2019) defenses; measure utility vs. ε and attacker accuracy under retrain / no-retrain scenarios |

All outputs (figures, tables, checkpoints, and a report scaffold) land under `outputs/`.

## Dataset

[AT&T (ORL) Face Database](https://cam-orl.co.uk/facedatabase.html) — 40 subjects × 10 grayscale images (92 × 112 px). Downloaded automatically by the pipeline.

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU recommended for Step 2 / Step 3 retrain (CPU works but is slow)

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `opencv-python`, `scikit-image`, `numpy`, `matplotlib`, `pillow`, `tqdm`, `pytest`.

## Quick Start

Run the entire pipeline end-to-end (skips steps whose outputs already exist):

```bash
python run_all.py
```

To also train a ResNet18 attacker for a stronger-attacker comparison:

```bash
python run_all.py --with-resnet
```

To force re-running all steps even if outputs exist:

```bash
python run_all.py --force
```

## Running Individual Steps

### Step 1 — De-identification

```bash
python scripts/download_att.py    # download dataset to data/att/
python step1_deidentify.py        # pixelization b∈{2,4,8,16}, blur k∈{15,45,99}
```

Outputs: `outputs/figs/step1_pixel_grid.png`, `outputs/figs/step1_blur_grid.png`, `outputs/tables/step1_quality.csv`

### Step 2 — CNN Attack

```bash
python step2_attack.py                      # CNN_ATT (default)
python step2_attack.py --model resnet18     # stronger attacker (optional)
```

Options: `--epochs 100` (default), `--batch-size 32`, `--seed 42`.

Outputs: `outputs/tables/step2_results.csv`, `outputs/figs/step2_curves_*.png`, `outputs/ckpts/step2/`

### Step 3 — DP Defense

```bash
# Utility curves: MSE/SSIM vs ε for DP-Pix / DP-Blur
python step3_dp_defense.py utility

# Train attackers on DP-perturbed data (Fan Table 1)
python step3_dp_defense.py retrain

# Evaluate step-2 attackers on DP test sets without retraining (bonus)
python step3_dp_defense.py no-retrain

# Qualitative comparison: Orig | NP-Pix | DP-Pix | NP-Blur | DP-Blur
python step3_dp_defense.py qualitative

# Sweep m-Neighborhood parameter at fixed ε
python step3_dp_defense.py m_sweep
```

### Generate Report

```bash
python generate_report.py   # writes outputs/report.md
```

`outputs/report.md` is a Markdown scaffold pre-filled with all computed tables and figure references. Sections marked `TODO` require hand-written prose. Convert to Word with:

```bash
pandoc outputs/report.md -o report/Report.docx
```

Optionally create `team.md` in the repo root with a contribution table (see `generate_report.py` docstring) before running the report generator.

## Repository Structure

```
.
├── scripts/
│   └── download_att.py         # download AT&T dataset
├── src/
│   ├── data.py                 # ATTDataset, train/test split
│   ├── dp.py                   # DP-Pix, DP-Blur (Fan 2019 §3.3)
│   ├── image_utils.py          # block_average, pad_to_multiple, load_pgm
│   ├── metrics.py              # MSE, SSIM wrappers
│   ├── models.py               # CNN_ATT, ResNet18 builder
│   ├── obfuscate.py            # NP-Pix, NP-Blur
│   ├── scoring.py              # top-k accuracy evaluation
│   ├── seed.py                 # global RNG seeding
│   └── train.py                # train_one_model
├── tests/                      # pytest test suite
├── step1_deidentify.py
├── step2_attack.py
├── step3_dp_defense.py
├── generate_report.py
├── run_all.py                  # end-to-end runner
└── requirements.txt
```

## Running Tests

```bash
pytest
```

## References

1. R. McPherson, R. Shokri, V. Shmatikov. *Defeating Image Obfuscation with Deep Learning.* arXiv:1609.00408, 2016.
2. L. Fan. *Differential Privacy for Image Publication.* TPDP 2019.
3. L. Fan. *Image Pixelization with Differential Privacy.* DBSec 2018, Springer LNCS 10980.
4. F. S. Samaria, A. C. Harter. *Parameterisation of a stochastic model for human face identification.* 1994. (AT&T database)
5. Z. Wang et al. *Image quality assessment: from error visibility to structural similarity.* IEEE TIP 2004. (SSIM)
