"""NP-Pix (pixelization) and NP-Blur (Gaussian blur)."""

from __future__ import annotations

import cv2
import numpy as np

from src.image_utils import block_average, pad_to_multiple


def np_pix(img: np.ndarray, b: int) -> np.ndarray:
    if b == 1:
        return img.copy()
    H, W = img.shape
    padded = pad_to_multiple(img, b)
    small = block_average(padded, b)
    big = cv2.resize(
        small.astype(np.float32),
        (padded.shape[1], padded.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.clip(big[:H, :W], 0, 255).astype(np.uint8)


def np_blur(img: np.ndarray, k: int) -> np.ndarray:
    if k % 2 == 0:
        raise ValueError(f"kernel size k={k} must be odd")
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)
