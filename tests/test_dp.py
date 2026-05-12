import numpy as np
import pytest

from src.dp import dp_blur, dp_pix, laplace_scale


def test_laplace_scale_default_dppix():
    # Fan defaults m=16, b=16, eps=0.5 -> 255*16/(256*0.5) = 31.875.
    assert abs(laplace_scale(b=16, m=16, eps=0.5) - 31.875) < 1e-6


def test_laplace_scale_inversely_proportional_to_eps():
    s1 = laplace_scale(b=16, m=16, eps=0.5)
    s2 = laplace_scale(b=16, m=16, eps=1.0)
    assert abs(s1 - 2 * s2) < 1e-6


def test_dp_pix_shape_and_dtype():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    out = dp_pix(img, b=16, m=16, eps=0.5, rng=np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_dp_pix_large_eps_approximates_np_pix():
    # At very large eps the Laplace scale -> 0, so dp_pix should match np_pix.
    from src.obfuscate import np_pix

    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    out = dp_pix(img, b=16, m=16, eps=1e8, rng=np.random.default_rng(0))
    ref = np_pix(img, b=16)
    assert np.max(np.abs(out.astype(int) - ref.astype(int))) <= 1


def test_dp_blur_shape_and_dtype():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    out = dp_blur(img, b0=4, m=16, eps=0.5, k=99, rng=np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_dp_blur_uses_b0_not_b():
    # b0 is DP-Blur's internal cell width; changing it must change the output.
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    a = dp_blur(img, b0=4, m=16, eps=0.5, k=15, rng=np.random.default_rng(0))
    b = dp_blur(img, b0=8, m=16, eps=0.5, k=15, rng=np.random.default_rng(0))
    assert not np.array_equal(a, b)


def test_laplace_scale_raises_when_b_zero():
    with pytest.raises(ValueError, match="b"):
        laplace_scale(b=0, m=16, eps=0.5)


def test_laplace_scale_raises_when_m_zero():
    with pytest.raises(ValueError, match="m"):
        laplace_scale(b=16, m=0, eps=0.5)


def test_laplace_scale_raises_when_eps_nonpositive():
    with pytest.raises(ValueError, match="eps"):
        laplace_scale(b=16, m=16, eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        laplace_scale(b=16, m=16, eps=-1.0)


def test_dp_blur_raises_when_k_even():
    img = np.random.randint(0, 256, (112, 92), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd"):
        dp_blur(img, b0=4, m=16, eps=0.5, k=4, rng=np.random.default_rng(0))
