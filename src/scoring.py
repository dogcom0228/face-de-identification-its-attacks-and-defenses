"""Top-1 / Top-5 accuracy and model scoring."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


@torch.no_grad()
def score_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Run inference, return top-1 / top-5."""
    model.train(False)
    use_cuda = device.type == "cuda"
    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        all_logits.append(model(x))
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {
        "top1": topk_accuracy(logits, labels, k=1),
        "top5": topk_accuracy(logits, labels, k=5),
    }
