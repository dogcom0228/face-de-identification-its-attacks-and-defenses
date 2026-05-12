"""Attack models for AT&T (40-class face ID).

- CNN_ATT: McPherson 2016 Appendix A.3, the canonical paper baseline.
- ResNet18_ATT: torchvision ResNet18 adapted for 1-channel 92×112 input,
  used as a "stronger attacker" reference. Note: ResNet18 has ~11M params
  on a 320-image training set — overfitting is expected. This is itself
  an interesting datapoint for the discussion.

Use `build_model(name)` to instantiate by string key.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class CNN_ATT(nn.Module):
    """Input (B, 1, 112, 92). Output (B, num_classes) log-probabilities.

    Conv channels 32->64->128, all 3x3 padding=1, LeakyReLU(0.01).
    MaxPool 2,2,3 — the final 3x3 is critical (not 2x2).
    Flatten -> 8064 -> 1024 -> Dropout(0.5) -> num_classes -> LogSoftmax.
    """

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
    """ResNet18 adapted for AT&T: 1-channel input + 40-class LogSoftmax head.

    Input (B, 1, 112, 92). Output (B, num_classes) log-probabilities.

    Adaptation from torchvision.models.resnet18:
      - conv1: 3-channel → 1-channel input (re-init from scratch, no pretrained).
      - fc: 1000-class ImageNet head → Linear(512, num_classes) + LogSoftmax.

    Trained from scratch (no ImageNet pretrain) because AT&T is grayscale faces
    very different from ImageNet RGB natural images.
    """

    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        # Replace conv1: in_channels 3 → 1, keep 7x7 stride 2.
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace fc with classifier + LogSoftmax (match CNN_ATT API).
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
    """Factory: name → instantiated model. Raises ValueError on unknown name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Known: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes)
