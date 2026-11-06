"""Grad-CAM: where in a crop a model's evidence for "parasitised" sits.

An appendix check added at the supervisor's suggestion (the proposal left
saliency out of scope). It does not test what a model relies on; RQ2 does that
by changing the input. It shows where the evidence is located, so the RQ2
reading (crop format and background matter) can be checked against it.

Grad-CAM (Selvaraju et al., 2017) weights each channel of the last
convolutional block by the mean gradient of the logit with respect to it, sums
the channels and keeps the positive part. The map is as coarse as that block:
14 x 14 for VGG-16, 7 x 7 for ResNet-50 and MobileNetV2, upsampled to the
224 x 224 input. A small ring covers about one map cell, so the map can say
"cell or background" more reliably than "parasite or not".

The measure is the share of a map's mass inside a region (cell, parasite),
read next to that region's share of the image area: a map spread evenly over
the crop would put exactly the area share there.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import paths
from .background import cell_mask, foreground
from .imagestats import BLACK_LEVEL


def target_layer(net: nn.Module, arch: str) -> nn.Module:
    """The last convolutional block, whose output Grad-CAM is computed on.

    VGG-16 uses the ReLU after its last convolution (features[29]) rather than
    the convolution itself, because that ReLU works in place and would
    overwrite a hooked convolution output.
    """
    if arch == "vgg16":
        return net.features[29]
    if arch == "resnet50":
        return net.layer4
    if arch == "mobilenet_v2":
        return net.features[-1]
    raise ValueError(f"unknown arch {arch!r}")


def gradcam(net: nn.Module, layer: nn.Module, x: torch.Tensor) -> tuple:
    """(maps, probabilities) for a batch: maps are N x 224 x 224, non-negative.

    In eval mode the samples of a batch do not interact, so one backward pass
    of the summed logits gives every sample its own gradients.
    """
    store = {}

    def keep(_module, _inputs, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(keep)
    try:
        net.zero_grad(set_to_none=True)
        logit = net(x)[:, 0]
        logit.sum().backward()
    finally:
        handle.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam[:, 0].numpy(), torch.sigmoid(logit).detach().numpy()


def to_input_grid(mask: np.ndarray, size: int = paths.MODEL_INPUT) -> np.ndarray:
    """A crop-sized mask on the model's input grid: black-padded to a square and
    resized like data.load_input, with nearest-neighbour so it stays binary."""
    h, w = mask.shape
    side = max(h, w)
    out = np.zeros((side, side), np.uint8)
    r0, c0 = (side - h) // 2, (side - w) // 2
    out[r0:r0 + h, c0:c0 + w] = mask
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def cell_region(rgb: np.ndarray, on_black: bool, seed: np.ndarray | None = None) -> np.ndarray:
    """The cell's pixels in a crop.

    Crops on black (NIH, background removed) are their non-black pixels. Raw
    test crops get the same Otsu cell mask that build_masked_crops.py used to
    remove their background, so "cell" means the same thing in both variants.
    """
    if on_black:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) >= BLACK_LEVEL
    kept, _ = cell_mask(foreground(rgb, "otsu-gray"), seed)
    return kept if seed is None else kept | seed


def heat_share(cam: np.ndarray, region: np.ndarray) -> float:
    """Share of the map's mass inside `region`; NaN for an all-zero map.

    Summed in float64: in float32 the part can round to slightly more than the whole.
    """
    cam = cam.astype(np.float64)
    total = cam.sum()
    return float(cam[region].sum() / total) if total > 0 else float("nan")
