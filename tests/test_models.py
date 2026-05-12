import pytest
import torch

from src.models import CNN_ATT, ResNet18_ATT, build_model


def test_forward_output_shape():
    model = CNN_ATT(num_classes=40)
    x = torch.randn(4, 1, 112, 92)
    out = model(x)
    assert out.shape == (4, 40)


def test_logsoftmax_output_sums_to_one():
    model = CNN_ATT(num_classes=40)
    x = torch.randn(2, 1, 112, 92)
    out = model(x).exp()
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-4)


def test_flatten_dim_is_8064():
    # Conv/pool feature map matches paper: 128 * 9 * 7 = 8064.
    # Height: 112 -> 56 -> 28 -> 9; Width: 92 -> 46 -> 23 -> 7.
    model = CNN_ATT(num_classes=40)
    x = torch.randn(1, 1, 112, 92)
    feat = model.features(x)
    assert feat.shape == (1, 128, 9, 7)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model", num_classes=40)


def test_build_model_known_names_instantiate():
    cnn = build_model("cnn_att", num_classes=40)
    resnet = build_model("resnet18", num_classes=40)
    assert isinstance(cnn, CNN_ATT)
    assert isinstance(resnet, ResNet18_ATT)


def test_resnet18_att_forward_output_shape():
    model = ResNet18_ATT(num_classes=40)
    x = torch.randn(2, 1, 112, 92)
    out = model(x)
    assert out.shape == (2, 40)


def test_resnet18_att_logsoftmax_sums_to_one():
    model = ResNet18_ATT(num_classes=40)
    x = torch.randn(2, 1, 112, 92)
    sums = model(x).exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-4)
