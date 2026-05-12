import numpy as np
import torch

from src.data import ATTDataset, att_train_test_split


def test_split_sizes_and_stratified():
    train_idx, test_idx = att_train_test_split("data/att", seed=42)
    assert len(train_idx) == 320
    assert len(test_idx) == 80
    assert set(train_idx).isdisjoint(test_idx)
    for subject in range(40):
        in_train = sum(1 for (s, _) in train_idx if s == subject)
        in_test = sum(1 for (s, _) in test_idx if s == subject)
        assert in_train == 8, f"subject {subject} train={in_train}"
        assert in_test == 2, f"subject {subject} test={in_test}"


def test_split_deterministic():
    a = att_train_test_split("data/att", seed=42)
    b = att_train_test_split("data/att", seed=42)
    assert a == b


def test_dataset_returns_normalized_grayscale():
    train_idx, _ = att_train_test_split("data/att", seed=42)
    ds = ATTDataset("data/att", train_idx)
    img, label = ds[0]
    assert img.shape == (1, 112, 92)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert 0 <= int(label) < 40


def test_dataset_transform_applied():
    train_idx, _ = att_train_test_split("data/att", seed=42)

    def zero_out(arr: np.ndarray) -> np.ndarray:
        return np.zeros_like(arr)

    ds = ATTDataset("data/att", train_idx, transform=zero_out)
    img, _ = ds[0]
    assert torch.all(img == 0)
