"""Grayscale image quality metrics: MSE and SSIM (vs original)."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Grayscale SSIM with data_range=255."""
    return float(structural_similarity(a, b, data_range=255))
