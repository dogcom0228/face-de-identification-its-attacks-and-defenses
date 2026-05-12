"""Run the full pipeline. Skips steps whose outputs already exist."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def steps_for_model(model: str) -> list[tuple[list[str], list[str]]]:
    csv_suffix = "" if model == "cnn_att" else f"_{model}"
    ckpt_dir = f"outputs/ckpts/step2/{model}"
    retrain_dir = f"outputs/ckpts/step3_retrain/{model}"
    return [
        (["python", "step2_attack.py", "--model", model], [
            f"outputs/tables/step2_results{csv_suffix}.csv",
            f"{ckpt_dir}/original.pt",
            f"{ckpt_dir}/pixel_b16.pt",
            f"{ckpt_dir}/blur_k99.pt",
        ]),
        (["python", "step3_dp_defense.py", "retrain", "--model", model], [
            f"outputs/tables/step3_retrain_attack{csv_suffix}.csv",
            f"{retrain_dir}/dp_pix_eps0.1.pt",
            f"{retrain_dir}/dp_pix_eps5.0.pt",
        ]),
        (["python", "step3_dp_defense.py", "no-retrain", "--model", model], [
            f"outputs/tables/step3_no_retrain_attack{csv_suffix}.csv",
        ]),
    ]


COMMON_STEPS_HEAD: list[tuple[list[str], list[str]]] = [
    (["python", "scripts/download_att.py"], ["data/att/s01/1.pgm", "data/att/s40/10.pgm"]),
    (["python", "step1_deidentify.py"], [
        "outputs/figs/step1_pixel_grid.png",
        "outputs/figs/step1_blur_grid.png",
        "outputs/tables/step1_quality.csv",
    ]),
    (["python", "step3_dp_defense.py", "utility"], [
        "outputs/tables/step3_utility.csv",
        "outputs/figs/step3_ssim_pix.png",
    ]),
    (["python", "step3_dp_defense.py", "m_sweep"], [
        "outputs/tables/step3_m_sweep.csv",
    ]),
    (["python", "step3_dp_defense.py", "qualitative"], [
        "outputs/figs/step3_qualitative.png",
    ]),
]


def all_present(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-resnet", action="store_true")
    args = parser.parse_args()

    steps: list[tuple[list[str], list[str]]] = []
    steps.extend(COMMON_STEPS_HEAD)
    steps.extend(steps_for_model("cnn_att"))
    if args.with_resnet:
        steps.extend(steps_for_model("resnet18"))

    for cmd, outputs in steps:
        label = " ".join(cmd)
        if not args.force and outputs and all_present(outputs):
            print(f"\n=== SKIP {label} ===\n", flush=True)
            continue
        print(f"\n=== {label} ===\n", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {label}")
            return result.returncode
    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
