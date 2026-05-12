"""Step 1: NP-Pix / NP-Blur on AT&T. Outputs comparison grids and quality CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import att_train_test_split
from src.image_utils import load_pgm
from src.metrics import mse, ssim
from src.obfuscate import np_blur, np_pix
from src.seed import set_seed

PIXEL_BS = [2, 4, 8, 16]
BLUR_KS = [15, 45, 99]


def save_grid(images: list[np.ndarray], titles: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(2.5 * len(images), 2.8))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/att")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    fig_dir = Path(args.out_dir) / "figs"
    table_dir = Path(args.out_dir) / "tables"

    sample = load_pgm(data_root, subject=0, idx=0)
    pix_imgs = [sample] + [np_pix(sample, b) for b in PIXEL_BS]
    pix_titles = ["orig"] + [f"b={b}" for b in PIXEL_BS]
    save_grid(pix_imgs, pix_titles, fig_dir / "step1_pixel_grid.png")

    blur_imgs = [sample] + [np_blur(sample, k) for k in BLUR_KS]
    blur_titles = ["orig"] + [f"k={k}" for k in BLUR_KS]
    save_grid(blur_imgs, blur_titles, fig_dir / "step1_blur_grid.png")

    train_idx, test_idx = att_train_test_split(data_root, seed=args.seed)
    all_idx = train_idx + test_idx
    rows = []
    for subject, idx in all_idx:
        orig = load_pgm(data_root, subject, idx)
        for b in PIXEL_BS:
            out = np_pix(orig, b)
            rows.append(
                {
                    "method": "pixel",
                    "param": b,
                    "subject": subject,
                    "idx": idx,
                    "mse": mse(orig, out),
                    "ssim": ssim(orig, out),
                }
            )
        for k in BLUR_KS:
            out = np_blur(orig, k)
            rows.append(
                {
                    "method": "blur",
                    "param": k,
                    "subject": subject,
                    "idx": idx,
                    "mse": mse(orig, out),
                    "ssim": ssim(orig, out),
                }
            )

    table_dir.mkdir(parents=True, exist_ok=True)
    with open(table_dir / "step1_quality.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "param", "subject", "idx", "mse", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[step1] wrote figures to {fig_dir} and table to {table_dir}/step1_quality.csv")


if __name__ == "__main__":
    main()
