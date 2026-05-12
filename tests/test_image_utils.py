from pathlib import Path

import numpy as np
import pytest

from src.image_utils import block_average, load_pgm, pad_to_multiple


def test_load_pgm_shape_dtype_range():
    img = load_pgm(Path("data/att"), subject=0, idx=0)
    assert img.shape == (112, 92)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_pad_to_multiple_already_aligned():
    img = np.zeros((8, 8), dtype=np.uint8)
    out = pad_to_multiple(img, b=4)
    assert out.shape == (8, 8)


def test_pad_to_multiple_unaligned():
    img = np.zeros((10, 9), dtype=np.uint8)
    out = pad_to_multiple(img, b=4)
    # (10,9) -> next multiples of 4 are (12,12)
    assert out.shape == (12, 12)


def test_pad_to_multiple_edge_replication():
    img = np.arange(20, dtype=np.uint8).reshape(4, 5)
    out = pad_to_multiple(img, b=4)
    assert out.shape == (4, 8)
    # padded cols should replicate the last col (= img[:, 4])
    np.testing.assert_array_equal(out[:, 5], img[:, 4])
    np.testing.assert_array_equal(out[:, 6], img[:, 4])
    np.testing.assert_array_equal(out[:, 7], img[:, 4])


def test_block_average_exact_arithmetic():
    img = np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ],
        dtype=np.uint8,
    )
    out = block_average(img, b=2)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx((10 + 20 + 50 + 60) / 4)
    assert out[0, 1] == pytest.approx((30 + 40 + 70 + 80) / 4)
    assert out[1, 0] == pytest.approx((90 + 100 + 130 + 140) / 4)
    assert out[1, 1] == pytest.approx((110 + 120 + 150 + 160) / 4)


def test_block_average_b1_is_identity():
    img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    out = block_average(img, b=1)
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img.astype(np.float64))
