"""AT&T (ORL) face dataset: 40 subjects, 10 images of 92x112 grayscale each."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_SUBJECTS = 40
IMAGES_PER_SUBJECT = 10


def att_train_test_split(
    root: str | Path, seed: int = 42
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    train, test = [], []
    for s in range(NUM_SUBJECTS):
        order = list(range(IMAGES_PER_SUBJECT))
        rng.shuffle(order)
        for i in order[:8]:
            train.append((s, i))
        for i in order[8:]:
            test.append((s, i))
    return train, test


class ATTDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        indices: list[tuple[int, int]],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.root = Path(root)
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        subject, img_idx = self.indices[idx]
        path = self.root / f"s{subject + 1:02d}" / f"{img_idx + 1}.pgm"
        arr = np.asarray(Image.open(path), dtype=np.uint8)
        if self.transform is not None:
            arr = self.transform(arr)
        arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, subject
