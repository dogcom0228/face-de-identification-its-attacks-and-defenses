import numpy as np

from src.metrics import mse, ssim


def test_mse_self_is_zero():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    assert mse(img, img) == 0.0


def test_mse_symmetric():
    a = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    b = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    assert abs(mse(a, b) - mse(b, a)) < 1e-6


def test_ssim_self_is_one():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    assert abs(ssim(img, img) - 1.0) < 1e-6


def test_ssim_grayscale_handles_2d():
    a = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    b = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    val = ssim(a, b)
    assert -1.0 <= val <= 1.0
