import numpy as np
import torch

from src.seed import set_seed


def test_set_seed_makes_numpy_deterministic():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_set_seed_makes_torch_deterministic():
    set_seed(7)
    a = torch.rand(5)
    set_seed(7)
    b = torch.rand(5)
    assert torch.allclose(a, b)
