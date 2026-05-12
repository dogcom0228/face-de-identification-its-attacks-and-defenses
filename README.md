# Face De-Identification: Attacks and Defenses

Implementation of a three-step pipeline on the AT&T (ORL) face database:

1. `step1_deidentify.py`: apply NP-Pix (pixelization) and NP-Blur (Gaussian blur), output quality grids and an MSE/SSIM table.
2. `step2_attack.py`: train a CNN_ATT attacker for each de-identification condition, report top-1 / top-5 accuracy.
3. `step3_dp_defense.py`: apply DP-Pix / DP-Blur defenses, measure utility vs. epsilon and attacker accuracy under retrain / no-retrain.

All outputs go under `outputs/`.

## Dataset

[AT&T (ORL) Face Database](https://cam-orl.co.uk/facedatabase.html): 40 subjects, 10 grayscale images each (92 x 112). Downloaded automatically by the pipeline.

## Requirements

- Python 3.10+
- A CUDA GPU is recommended for step 2 and step 3 retrain (CPU works but is slow).

```bash
pip install -r requirements.txt
```

Main packages: `torch`, `torchvision`, `opencv-python`, `scikit-image`, `numpy`, `matplotlib`, `pillow`, `tqdm`, `pytest`.

## Quick Start

Run the whole pipeline (skips steps whose outputs already exist):

```bash
python run_all.py
```

Also train ResNet18 as a stronger attacker:

```bash
python run_all.py --with-resnet
```

Force re-running all steps:

```bash
python run_all.py --force
```

## Running Each Step

### Step 1: De-identification

```bash
python scripts/download_att.py
python step1_deidentify.py
```

Outputs: `outputs/figs/step1_pixel_grid.png`, `outputs/figs/step1_blur_grid.png`, `outputs/tables/step1_quality.csv`.

### Step 2: CNN Attack

```bash
python step2_attack.py
python step2_attack.py --model resnet18
```

Options: `--epochs 100`, `--batch-size 32`, `--seed 42`.

Outputs: `outputs/tables/step2_results.csv`, `outputs/figs/step2_curves_*.png`, `outputs/ckpts/step2/`.

### Step 3: DP Defense

```bash
python step3_dp_defense.py utility
python step3_dp_defense.py retrain
python step3_dp_defense.py no-retrain
python step3_dp_defense.py qualitative
python step3_dp_defense.py m_sweep
```

## Repository Structure

```text
.
├── scripts/
│   └── download_att.py
├── src/
│   ├── data.py
│   ├── dp.py
│   ├── image_utils.py
│   ├── metrics.py
│   ├── models.py
│   ├── obfuscate.py
│   ├── scoring.py
│   ├── seed.py
│   └── train.py
├── tests/
├── step1_deidentify.py
├── step2_attack.py
├── step3_dp_defense.py
├── run_all.py
└── requirements.txt
```

## Running Tests

```bash
pytest
```

## References

1. R. McPherson, R. Shokri, V. Shmatikov. *Defeating Image Obfuscation with Deep Learning.* arXiv:1609.00408, 2016.
2. L. Fan. *Differential Privacy for Image Publication.* TPDP 2019.
3. L. Fan. *Image Pixelization with Differential Privacy.* DBSec 2018.
4. F. S. Samaria, A. C. Harter. *Parameterisation of a stochastic model for human face identification.* 1994.
5. Z. Wang et al. *Image quality assessment: from error visibility to structural similarity.* IEEE TIP 2004.
