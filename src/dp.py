"""DP-Pix and DP-Blur (Fan 2019)."""

from __future__ import annotations

import cv2
import numpy as np

from src.image_utils import block_average, pad_to_multiple


def laplace_scale(b: int, m: int, eps: float) -> float:
    if b <= 0:
        raise ValueError(f"b must be > 0, got {b}")
    if m <= 0:
        raise ValueError(f"m must be > 0, got {m}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    return 255.0 * m / (b ** 2 * eps)


def dp_pix(
    img: np.ndarray,
    b: int = 16,
    m: int = 16,
    eps: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    H, W = img.shape
    padded = pad_to_multiple(img, b)
    small = block_average(padded, b)
    scale = laplace_scale(b=b, m=m, eps=eps)
    noisy = small + rng.laplace(0.0, scale, small.shape)
    big = cv2.resize(
        noisy.astype(np.float32),
        (padded.shape[1], padded.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.clip(big[:H, :W], 0, 255).astype(np.uint8)


def dp_blur(
    img: np.ndarray,
    b0: int = 4,
    m: int = 16,
    eps: float = 0.5,
    k: int = 99,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if k % 2 == 0:
        raise ValueError(f"kernel size k={k} must be odd")
    rng = rng or np.random.default_rng()
    H, W = img.shape
    padded = pad_to_multiple(img, b0)
    small = block_average(padded, b0)
    scale = laplace_scale(b=b0, m=m, eps=eps)
    noisy = small + rng.laplace(0.0, scale, small.shape)
    big = cv2.resize(
        noisy.astype(np.float32),
        (padded.shape[1], padded.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    big = np.clip(big[:H, :W], 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(big, (k, k), sigmaX=0)
