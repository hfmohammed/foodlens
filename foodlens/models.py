import torch
from torch import nn

from .classes import NUM_CLASSES


class FoodCNN(nn.Module):
    """3 x [conv3x3 same + ReLU + maxpool2] with 32/64/128 channels -> GAP -> Linear(10). Returns raw logits."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        layers, c = [], 3
        for out in (32, 64, 128):
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def build_mobilenet(num_classes: int = NUM_CLASSES, pretrained: bool = True, unfreeze_last: bool = False):
    """MobileNetV3-Small; classifier replaced by Linear(576, 10). Backbone frozen except optionally the last block."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier = nn.Linear(576, num_classes)
    for p in m.features.parameters():
        p.requires_grad = False
    if unfreeze_last:
        for p in m.features[-1].parameters():
            p.requires_grad = True
    return m


def build_model(arch: str, **kw):
    return FoodCNN() if arch == "cnn" else build_mobilenet(**kw)


def set_train_mode(model: nn.Module, arch: str, unfreeze_last: bool = False):
    """Keep frozen BatchNorm layers in eval mode so their running stats are not updated."""
    model.train()
    if arch == "mobilenet":
        model.features.eval()
        if unfreeze_last:
            model.features[-1].train()
