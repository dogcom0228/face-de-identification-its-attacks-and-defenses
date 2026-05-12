"""Shared image utilities (block average, padding, PGM loader).

Extracted from src/obfuscate.py, src/dp.py, step1_deidentify.py, step3_dp_defense.py
to remove duplication (DRY).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def pad_to_multiple(img: np.ndarray, b: int) -> np.ndarray:
    """Pad with edge replication so each side is a multiple of b."""
    H, W = img.shape
    pad_h = (-H) % b
    pad_w = (-W) % b
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")


def block_average(padded: np.ndarray, b: int) -> np.ndarray:
    """True b×b block average. padded.shape must be multiples of b.

    NOTE: We deliberately do NOT use cv2.resize(..., INTER_AREA) — for integer
    downscale factors > 2 OpenCV's INTER_AREA does NOT compute the exact block
    average (it uses a different sampling kernel). Reshape + mean is exact.
    """
    H, W = padded.shape
    return (
        padded.astype(np.float64)
        .reshape(H // b, b, W // b, b)
        .mean(axis=(1, 3))
    )


def load_pgm(root: Path, subject: int, idx: int) -> np.ndarray:
    """Load AT&T PGM image as uint8 (H, W) grayscale.

    subject is 0-indexed (0..39 → s01..s40), idx is 0-indexed (0..9 → 1..10).
    """
    return np.asarray(
        Image.open(root / f"s{subject + 1:02d}" / f"{idx + 1}.pgm"),
        dtype=np.uint8,
    )
