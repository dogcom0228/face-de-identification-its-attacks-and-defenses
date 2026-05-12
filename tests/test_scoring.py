import torch

from src.scoring import topk_accuracy


def test_topk_perfect_predictions():
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 5.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    assert topk_accuracy(logits, labels, k=1) == 1.0
    assert topk_accuracy(logits, labels, k=2) == 1.0


def test_topk_random_predictions():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    labels = torch.tensor([0])  # 真實 label 但 logit 最小
    assert topk_accuracy(logits, labels, k=1) == 0.0
    assert topk_accuracy(logits, labels, k=3) == 1.0


def test_topk_mixed():
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],  # top-1 = 0 ✓
            [1.0, 3.0, 0.0],  # top-1 = 1, label=0 ✗，但 top-2 ✓
        ]
    )
    labels = torch.tensor([0, 0])
    assert topk_accuracy(logits, labels, k=1) == 0.5
    assert topk_accuracy(logits, labels, k=2) == 1.0
