import numpy as np

from src.obfuscate import np_blur, np_pix


def test_np_pix_shape_and_dtype():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    out = np_pix(img, b=4)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_np_pix_block_constant_when_b_divides_dims():
    img = np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ],
        dtype=np.uint8,
    )
    out = np_pix(img, b=2)
    assert out[0, 0] == out[0, 1] == out[1, 0] == out[1, 1]
    assert out[2, 2] == out[2, 3] == out[3, 2] == out[3, 3]


def test_np_pix_b1_is_identity():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    out = np_pix(img, b=1)
    assert np.array_equal(out, img)


def test_np_blur_shape_and_dtype():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    out = np_blur(img, k=15)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_np_blur_reduces_variance():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    blurred = np_blur(img, k=99)
    assert blurred.var() < img.var()
