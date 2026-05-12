"""Non-private face obfuscation: pixelization (NP-Pix) and Gaussian blur (NP-Blur)."""

from __future__ import annotations

import cv2
import numpy as np

from src.image_utils import block_average, pad_to_multiple


def np_pix(img: np.ndarray, b: int) -> np.ndarray:
    """Pixelize: b x b block-average then nearest-neighbor upsample. McPherson §5.3."""
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
    """Gaussian blur with odd kernel size k. sigma=0 lets OpenCV derive sigma from k."""
    if k % 2 == 0:
        raise ValueError(f"kernel size k={k} must be odd")
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)
