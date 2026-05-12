"""Shared image utilities: block average, padding, PGM loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def pad_to_multiple(img: np.ndarray, b: int) -> np.ndarray:
    H, W = img.shape
    pad_h = (-H) % b
    pad_w = (-W) % b
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")


def block_average(padded: np.ndarray, b: int) -> np.ndarray:
    # cv2.INTER_AREA does not equal a true block average for integer
    # downscales > 2, so compute it with reshape + mean.
    H, W = padded.shape
    return (
        padded.astype(np.float64)
        .reshape(H // b, b, W // b, b)
        .mean(axis=(1, 3))
    )


def load_pgm(root: Path, subject: int, idx: int) -> np.ndarray:
    return np.asarray(
        Image.open(root / f"s{subject + 1:02d}" / f"{idx + 1}.pgm"),
        dtype=np.uint8,
    )
