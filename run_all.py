"""Run full HW3 pipeline end-to-end with idempotent skipping.

Each step's expected output file(s) are checked; if all exist the step is
skipped. To force re-run, delete the relevant outputs/ files first or use
`python run_all.py --force`.

By default the pipeline runs the CNN_ATT attacker (McPherson baseline).
Use `--with-resnet` to also train ResNet18 (stronger attacker) for
side-by-side comparison.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def steps_for_model(model: str) -> list[tuple[list[str], list[str]]]:
    """Build the step2 + step3 retrain/no-retrain triple for a given model."""
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
            f"{retrain_dir}/dp_pix_eps5.0.pt",   # rightmost ε ensures 7-point sweep ran
        ]),
        (["python", "step3_dp_defense.py", "no-retrain", "--model", model], [
            f"outputs/tables/step3_no_retrain_attack{csv_suffix}.csv",
        ]),
    ]


# Steps that are model-agnostic (run once regardless of attacker arch).
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

COMMON_STEPS_TAIL: list[tuple[list[str], list[str]]] = [
    (["python", "generate_report.py"], ["outputs/report.md"]),
]


def all_present(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full HW3 pipeline. Existing outputs are skipped unless --force."
    )
    parser.add_argument("--force", action="store_true",
                        help="ignore existing outputs and re-run every step")
    parser.add_argument("--with-resnet", action="store_true",
                        help="also train ResNet18 (stronger attacker) for side-by-side comparison; "
                             "roughly doubles step2 + step3 retrain time")
    args = parser.parse_args()

    steps: list[tuple[list[str], list[str]]] = []
    steps.extend(COMMON_STEPS_HEAD)
    steps.extend(steps_for_model("cnn_att"))
    if args.with_resnet:
        steps.extend(steps_for_model("resnet18"))
    steps.extend(COMMON_STEPS_TAIL)

    for cmd, outputs in steps:
        label = " ".join(cmd)
        if not args.force and outputs and all_present(outputs):
            print(f"\n=== SKIP {label} (outputs already exist; --force to override) ===\n", flush=True)
            continue
        print(f"\n=== {label} ===\n", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {label}")
            return result.returncode
    print("\n=== ALL DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
