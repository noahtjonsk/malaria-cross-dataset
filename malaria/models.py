"""The three architectures RQ1 compares, built the same way.

Each network starts from torchvision's ImageNet weights and has its 1000-class
ImageNet head replaced by a single output (the logit for "parasitised"). Every
layer is then fine-tuned: nothing is frozen. Fine-tuning the whole network,
rather than training only a new head on frozen ImageNet features, is what the
supervisor asked for ("do not forget to unfreeze the weights") and what the
malaria CNN studies this project compares against did; `build_model` asserts it.

| arch         | parameters | torchvision weights |
|--------------|-----------:|---------------------|
| mobilenet_v2 |   ~3.5 M   | IMAGENET1K_V1       |
| resnet50     |  ~25.6 M   | IMAGENET1K_V1       |
| vgg16        | ~138.4 M   | IMAGENET1K_V1       |

IMAGENET1K_V1 for all three, so the pretraining recipe is the same generation
for every architecture (ResNet-50 and MobileNetV2 also have V2 weights, trained
with a heavier recipe that VGG-16 has no counterpart for).
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

ARCHS = ("vgg16", "resnet50", "mobilenet_v2")


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    """An ImageNet-pretrained network with a 1-logit head, all layers trainable."""
    if arch == "vgg16":
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier[6] = nn.Linear(net.classifier[6].in_features, 1)
    elif arch == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif arch == "mobilenet_v2":
        net = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    else:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")

    for p in net.parameters():
        p.requires_grad_(True)
    check_all_trainable(net)
    return net


def check_all_trainable(net: nn.Module) -> None:
    """Raise if any parameter is frozen (a continuous-analysis check)."""
    frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
    if frozen:
        raise AssertionError(f"{len(frozen)} frozen parameters, e.g. {frozen[:3]}")


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
