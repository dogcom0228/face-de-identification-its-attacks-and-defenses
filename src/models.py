"""Attack models: CNN_ATT (McPherson) and ResNet18 baseline."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class CNN_ATT(nn.Module):
    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 9, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet18_ATT(nn.Module):
    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, num_classes),
            nn.LogSoftmax(dim=1),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


MODEL_REGISTRY = {
    "cnn_att": CNN_ATT,
    "resnet18": ResNet18_ATT,
}


def build_model(name: str, num_classes: int = 40) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Known: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes)
