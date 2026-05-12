"""Training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import build_model
from src.scoring import score_model


def _build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )


def _build_scheduler(optim: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: 1.0 / (1.0 + 1e-7 * step)
    )


def train_one_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 40,
    epochs: int = 100,
    lr: float = 0.01,
    device: torch.device | None = None,
    ckpt_path: Path | None = None,
    model_name: str = "cnn_att",
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes).to(device)
    optim = _build_optimizer(model, lr)
    sched = _build_scheduler(optim)
    loss_fn = nn.NLLLoss()

    history = []
    use_cuda = device.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train(True)
        running_loss = torch.zeros((), device=device)
        n_samples = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            sched.step()
            running_loss += loss.detach() * y.numel()
            n_samples += y.numel()

        metrics = score_model(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(running_loss.item()) / max(1, n_samples)
        history.append(metrics)

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "history": history},
            ckpt_path,
        )
    return {"model": model, "history": history}
