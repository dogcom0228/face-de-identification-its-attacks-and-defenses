"""Step 3: DP-Pix / DP-Blur defense.

Subcommands:
  python step3_dp_defense.py utility
  python step3_dp_defense.py retrain
  python step3_dp_defense.py no-retrain
  python step3_dp_defense.py qualitative
  python step3_dp_defense.py m_sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import ATTDataset, att_train_test_split
from src.dp import dp_blur, dp_pix
from src.image_utils import load_pgm
from src.metrics import mse, ssim
from src.models import build_model
from src.obfuscate import np_blur, np_pix
from src.scoring import score_model
from src.seed import set_seed
from src.train import train_one_model

EPS_UTILITY = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
EPS_ATTACK = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
M_SWEEP = [8, 16, 32, 64]
DEFAULT_B = 16
DEFAULT_B0 = 4
DEFAULT_M = 16
DEFAULT_K = 99
# Train and test use separate seeds so DataLoader shuffle on train doesn't
# advance the test rng.
TEST_SEED_OFFSET = 10_000


def cmd_utility(args) -> None:
    train_idx, test_idx = att_train_test_split(args.data_root, seed=args.seed)
    all_idx = train_idx + test_idx
    table_dir = Path(args.out_dir) / "tables"
    fig_dir = Path(args.out_dir) / "figs"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[step3 utility] sweeping eps={EPS_UTILITY}")
    for eps in EPS_UTILITY:
        rng = np.random.default_rng(args.seed)
        mse_pix_acc, ssim_pix_acc = [], []
        mse_blur_acc, ssim_blur_acc = [], []
        for subject, idx in all_idx:
            orig = load_pgm(Path(args.data_root), subject, idx)
            p = dp_pix(orig, b=DEFAULT_B, m=DEFAULT_M, eps=eps, rng=rng)
            b = dp_blur(orig, b0=DEFAULT_B0, m=DEFAULT_M, eps=eps, k=DEFAULT_K, rng=rng)
            mse_pix_acc.append(mse(orig, p))
            ssim_pix_acc.append(ssim(orig, p))
            mse_blur_acc.append(mse(orig, b))
            ssim_blur_acc.append(ssim(orig, b))
        rows.append(
            {
                "eps": eps,
                "mse_dp_pix": float(np.mean(mse_pix_acc)),
                "ssim_dp_pix": float(np.mean(ssim_pix_acc)),
                "mse_dp_blur": float(np.mean(mse_blur_acc)),
                "ssim_dp_blur": float(np.mean(ssim_blur_acc)),
            }
        )
        print(f"  eps={eps}: ssim_pix={rows[-1]['ssim_dp_pix']:.3f}  ssim_blur={rows[-1]['ssim_dp_blur']:.3f}")

    np_pix_mse, np_pix_ssim = [], []
    np_blur_mse, np_blur_ssim = [], []
    for subject, idx in all_idx:
        orig = load_pgm(Path(args.data_root), subject, idx)
        p = np_pix(orig, b=DEFAULT_B)
        b = np_blur(orig, k=DEFAULT_K)
        np_pix_mse.append(mse(orig, p))
        np_pix_ssim.append(ssim(orig, p))
        np_blur_mse.append(mse(orig, b))
        np_blur_ssim.append(ssim(orig, b))
    baseline = {
        "np_pix_mse": float(np.mean(np_pix_mse)),
        "np_pix_ssim": float(np.mean(np_pix_ssim)),
        "np_blur_mse": float(np.mean(np_blur_mse)),
        "np_blur_ssim": float(np.mean(np_blur_ssim)),
    }

    with open(table_dir / "step3_utility.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["eps", "mse_dp_pix", "ssim_dp_pix", "mse_dp_blur", "ssim_dp_blur"]
        )
        writer.writeheader()
        writer.writerows(rows)

    for metric, dp_key, np_key in [
        ("MSE", "mse_dp_pix", "np_pix_mse"),
        ("SSIM", "ssim_dp_pix", "np_pix_ssim"),
        ("MSE", "mse_dp_blur", "np_blur_mse"),
        ("SSIM", "ssim_dp_blur", "np_blur_ssim"),
    ]:
        family = "pix" if "pix" in dp_key else "blur"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["eps"] for r in rows], [r[dp_key] for r in rows], marker="o", label=f"DP-{family.title()}")
        ax.axhline(baseline[np_key], linestyle="--", color="gray", label=f"NP-{family.title()}")
        ax.set_xscale("log")
        ax.set_xlabel("eps")
        ax.set_ylabel(metric)
        ax.set_title(f"Step 3: {metric} vs eps (DP-{family.title()})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"step3_{metric.lower()}_{family}.png", dpi=160)
        plt.close(fig)

    print(f"[step3 utility] wrote {table_dir}/step3_utility.csv + 4 figs")


def _build_dp_transform(method: str, eps: float, seed: int):
    rng = np.random.default_rng(seed)

    def pix(arr):
        return dp_pix(arr, b=DEFAULT_B, m=DEFAULT_M, eps=eps, rng=rng)

    def blur(arr):
        return dp_blur(arr, b0=DEFAULT_B0, m=DEFAULT_M, eps=eps, k=DEFAULT_K, rng=rng)

    return {"dp_pix": pix, "dp_blur": blur}[method]


def cmd_retrain(args) -> None:
    train_idx, test_idx = att_train_test_split(args.data_root, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table_dir = Path(args.out_dir) / "tables"
    ckpt_dir = Path(args.out_dir) / "ckpts" / "step3_retrain" / args.model
    table_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step3 retrain] device={device}  model={args.model}  eps_grid={EPS_ATTACK}")

    rows = []
    for method in ["dp_pix", "dp_blur"]:
        for eps in EPS_ATTACK:
            label = f"{method}_eps{eps}"
            train_transform = _build_dp_transform(method, eps, seed=args.seed)
            test_transform = _build_dp_transform(method, eps, seed=args.seed + TEST_SEED_OFFSET)
            train_ds = ATTDataset(args.data_root, train_idx, transform=train_transform)
            test_ds = ATTDataset(args.data_root, test_idx, transform=test_transform)
            # num_workers must be 0: DP transforms close over an
            # np.random.Generator, so multiple workers would share state.
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            print(f"[step3 retrain] training {label} (epochs={args.epochs}, model={args.model})")
            out = train_one_model(
                train_loader,
                test_loader,
                num_classes=40,
                epochs=args.epochs,
                device=device,
                ckpt_path=ckpt_dir / f"{label}.pt",
                model_name=args.model,
            )
            last = out["history"][-1]
            rows.append(
                {
                    "method": method,
                    "eps": eps,
                    "top1": last["top1"],
                    "top5": last["top5"],
                }
            )
            print(f"[step3 retrain]   final top1={last['top1']:.4f}  top5={last['top5']:.4f}")

    csv_name = "step3_retrain_attack.csv" if args.model == "cnn_att" else f"step3_retrain_attack_{args.model}.csv"
    with open(table_dir / csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "eps", "top1", "top5"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[step3 retrain] wrote {table_dir}/{csv_name}")


def cmd_no_retrain(args) -> None:
    _, test_idx = att_train_test_split(args.data_root, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table_dir = Path(args.out_dir) / "tables"
    ckpt_root = Path(args.out_dir) / "ckpts" / "step2" / args.model
    table_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step3 no-retrain] device={device}  model={args.model}")

    pairings = [
        ("dp_pix", ckpt_root / "pixel_b16.pt"),
        ("dp_blur", ckpt_root / "blur_k99.pt"),
    ]
    rows = []
    for method, ckpt_path in pairings:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Step 2 ckpt {ckpt_path} missing. Run step2_attack.py first."
            )
        model = build_model(args.model, num_classes=40).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        for eps in EPS_ATTACK:
            test_transform = _build_dp_transform(method, eps, seed=args.seed + TEST_SEED_OFFSET)
            test_ds = ATTDataset(args.data_root, test_idx, transform=test_transform)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
            metrics = score_model(model, test_loader, device)
            rows.append(
                {
                    "method": method,
                    "eps": eps,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "attacker_ckpt": ckpt_path.name,
                }
            )
            print(f"[step3 no-retrain] {method} eps={eps}: top1={metrics['top1']:.4f}")

    csv_name = "step3_no_retrain_attack.csv" if args.model == "cnn_att" else f"step3_no_retrain_attack_{args.model}.csv"
    with open(table_dir / csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "eps", "top1", "top5", "attacker_ckpt"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[step3 no-retrain] wrote {table_dir}/{csv_name}")


def cmd_m_sweep(args) -> None:
    train_idx, test_idx = att_train_test_split(args.data_root, seed=args.seed)
    all_idx = train_idx + test_idx
    table_dir = Path(args.out_dir) / "tables"
    fig_dir = Path(args.out_dir) / "figs"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[step3 m_sweep] m in {M_SWEEP} @ eps={args.eps_m_sweep}")
    for m in M_SWEEP:
        rng = np.random.default_rng(args.seed)
        mse_pix, ssim_pix = [], []
        mse_blur, ssim_blur = [], []
        for subject, idx in all_idx:
            orig = load_pgm(Path(args.data_root), subject, idx)
            p = dp_pix(orig, b=DEFAULT_B, m=m, eps=args.eps_m_sweep, rng=rng)
            b = dp_blur(orig, b0=DEFAULT_B0, m=m, eps=args.eps_m_sweep, k=DEFAULT_K, rng=rng)
            mse_pix.append(mse(orig, p))
            ssim_pix.append(ssim(orig, p))
            mse_blur.append(mse(orig, b))
            ssim_blur.append(ssim(orig, b))
        row = {
            "m": m,
            "eps": args.eps_m_sweep,
            "mse_dp_pix": float(np.mean(mse_pix)),
            "ssim_dp_pix": float(np.mean(ssim_pix)),
            "mse_dp_blur": float(np.mean(mse_blur)),
            "ssim_dp_blur": float(np.mean(ssim_blur)),
        }
        rows.append(row)
        print(f"  m={m}: ssim_pix={row['ssim_dp_pix']:.3f}  ssim_blur={row['ssim_dp_blur']:.3f}")

    with open(table_dir / "step3_m_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["m", "eps", "mse_dp_pix", "ssim_dp_pix", "mse_dp_blur", "ssim_dp_blur"]
        )
        writer.writeheader()
        writer.writerows(rows)

    for metric, dp_key in [
        ("MSE", "mse_dp_pix"), ("SSIM", "ssim_dp_pix"),
        ("MSE", "mse_dp_blur"), ("SSIM", "ssim_dp_blur"),
    ]:
        family = "pix" if "pix" in dp_key else "blur"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["m"] for r in rows], [r[dp_key] for r in rows], marker="o",
                label=f"DP-{family.title()} (eps={args.eps_m_sweep})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(M_SWEEP)
        ax.set_xticklabels([str(m) for m in M_SWEEP])
        ax.set_xlabel("m")
        ax.set_ylabel(metric)
        ax.set_title(f"Step 3 m-sweep: {metric} vs m (DP-{family.title()})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"step3_m_{metric.lower()}_{family}.png", dpi=160)
        plt.close(fig)

    print(f"[step3 m_sweep] wrote {table_dir}/step3_m_sweep.csv + 4 figs")


def cmd_qualitative(args) -> None:
    fig_dir = Path(args.out_dir) / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subjects = [0, 9, 19, 29]
    fig, axes = plt.subplots(len(subjects), 5, figsize=(10, 8))
    col_titles = ["Orig", "NP-Pix", "DP-Pix", "NP-Blur", "DP-Blur"]

    for row_i, subject in enumerate(subjects):
        orig = load_pgm(Path(args.data_root), subject, 0)
        imgs = [
            orig,
            np_pix(orig, b=DEFAULT_B),
            dp_pix(orig, b=DEFAULT_B, m=DEFAULT_M, eps=0.5, rng=rng),
            np_blur(orig, k=DEFAULT_K),
            dp_blur(orig, b0=DEFAULT_B0, m=DEFAULT_M, eps=0.5, k=DEFAULT_K, rng=rng),
        ]
        for col_i, img in enumerate(imgs):
            ax = axes[row_i, col_i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "step3_qualitative.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[step3 qualitative] wrote {fig_dir}/step3_qualitative.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["utility", "retrain", "no-retrain", "qualitative", "m_sweep"]:
        p = sub.add_parser(name)
        p.add_argument("--data-root", default="data/att")
        p.add_argument("--out-dir", default="outputs")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--model", default="cnn_att", choices=["cnn_att", "resnet18"])
        if name == "m_sweep":
            p.add_argument("--eps-m-sweep", type=float, default=0.5)

    args = parser.parse_args()
    set_seed(args.seed)
    dispatch = {
        "utility": cmd_utility,
        "retrain": cmd_retrain,
        "no-retrain": cmd_no_retrain,
        "qualitative": cmd_qualitative,
        "m_sweep": cmd_m_sweep,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
