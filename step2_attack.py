"""Step 2: train attackers on (original, NP-Pix x 4, NP-Blur x 3)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data import ATTDataset, att_train_test_split
from src.obfuscate import np_blur, np_pix
from src.seed import set_seed
from src.train import train_one_model

CONDITIONS = (
    [("original", None)]
    + [("pixel_b", b) for b in [2, 4, 8, 16]]
    + [("blur_k", k) for k in [15, 45, 99]]
)


def build_transform(name: str, param: int | None):
    if name == "original":
        return None
    if name == "pixel_b":
        return lambda arr: np_pix(arr, b=param)
    if name == "blur_k":
        return lambda arr: np_blur(arr, k=param)
    raise ValueError(name)


def condition_label(name: str, param: int | None) -> str:
    return name if param is None else f"{name}{param}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/att")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="cnn_att", choices=["cnn_att", "resnet18"])
    args = parser.parse_args()
    set_seed(args.seed)

    train_idx, test_idx = att_train_test_split(args.data_root, seed=args.seed)
    fig_dir = Path(args.out_dir) / "figs"
    table_dir = Path(args.out_dir) / "tables"
    ckpt_dir = Path(args.out_dir) / "ckpts" / "step2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = ckpt_dir / args.model
    print(f"[step2] device={device}  model={args.model}")

    results = []
    curves = {}
    for name, param in CONDITIONS:
        label = condition_label(name, param)
        transform = build_transform(name, param)
        train_ds = ATTDataset(args.data_root, train_idx, transform=transform)
        test_ds = ATTDataset(args.data_root, test_idx, transform=transform)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        )

        print(f"[step2] training {label} (epochs={args.epochs}, model={args.model})")
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
        results.append(
            {
                "condition": label,
                "top1": last["top1"],
                "top5": last["top5"],
                "baseline_top1": 1 / 40,
                "baseline_top5": 5 / 40,
            }
        )
        curves[label] = [(h["epoch"], h["top1"]) for h in out["history"]]
        print(f"[step2]   final top1={last['top1']:.4f}  top5={last['top5']:.4f}")

    table_dir.mkdir(parents=True, exist_ok=True)
    results_csv = "step2_results.csv" if args.model == "cnn_att" else f"step2_results_{args.model}.csv"
    with open(table_dir / results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "top1", "top5", "baseline_top1", "baseline_top5"]
        )
        writer.writeheader()
        writer.writerows(results)

    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_suffix = "" if args.model == "cnn_att" else f"_{args.model}"
    for group, members in [
        ("pixel", ["original", "pixel_b2", "pixel_b4", "pixel_b8", "pixel_b16"]),
        ("blur", ["original", "blur_k15", "blur_k45", "blur_k99"]),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label in members:
            xs, ys = zip(*curves[label])
            ax.plot(xs, ys, label=label, linewidth=1.5)
        ax.axhline(1 / 40, linestyle="--", color="gray", label="random")
        ax.set_xlabel("epoch")
        ax.set_ylabel("top-1 accuracy")
        ax.set_title(f"Step 2: {group} attack accuracy vs epoch ({args.model})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(fig_dir / f"step2_curves_{group}{curve_suffix}.png", dpi=160)
        plt.close(fig)

    print(f"[step2] wrote {table_dir}/{results_csv} and step2_curves_*{curve_suffix}.png")


if __name__ == "__main__":
    main()
