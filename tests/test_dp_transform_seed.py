"""Regression test for audit §D-C3: train/test rng must be independent."""

import sys
from pathlib import Path

import numpy as np

# step3_dp_defense.py is at repo root; tests/ is a subdir, so adjust sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step3_dp_defense import TEST_SEED_OFFSET, _build_dp_transform


def test_build_dp_transform_same_seed_reproducible():
    img = np.random.RandomState(0).randint(0, 256, (112, 92), dtype=np.uint8)
    t1 = _build_dp_transform("dp_pix", eps=0.5, seed=42)
    t2 = _build_dp_transform("dp_pix", eps=0.5, seed=42)
    out1 = t1(img)
    out2 = t2(img)
    np.testing.assert_array_equal(out1, out2)


def test_build_dp_transform_different_seed_produces_different_noise():
    img = np.random.RandomState(0).randint(0, 256, (112, 92), dtype=np.uint8)
    train_t = _build_dp_transform("dp_pix", eps=0.5, seed=42)
    test_t = _build_dp_transform("dp_pix", eps=0.5, seed=42 + TEST_SEED_OFFSET)
    train_out = train_t(img)
    test_out = test_t(img)
    assert not np.array_equal(train_out, test_out)
