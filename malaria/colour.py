"""Colour spaces and colour transforms for the stain comparison.

RGB mixes three things the EDA needs apart: which colour a pixel is, how strong
that colour is, and how bright it is. The other spaces separate them:

HSV      hue (which colour), saturation (how strong), value (how bright)
CIE Lab  lightness L apart from two colour-opponent axes, a (green-red) and
         b (blue-yellow); the space the Reinhard stain transfer works in
YCbCr    luma Y apart from blue- and red-difference chroma
Giemsa   colour deconvolution with scikit-image's `bex_from_rgb` matrix
         (methyl blue + eosin): how much of each stain a pixel absorbed. This
         is the Giemsa counterpart of the H&E deconvolution the stain
         standardisation literature is built on.
H&E      the same deconvolution with the H&E-DAB matrix (`rgb2hed`), included
         because that literature reports in it; the stains are not Giemsa's,
         so it is a reference point rather than a measurement of these slides.

Every statistic is taken over tissue pixels only (gray >= BLACK_LEVEL), for the
reason given in imagestats.py: black padding has zero hue and saturation and
would dilute every average on a masked crop.

The transforms map a test cell's colours toward an NIH reference, so their
effect can be looked at before any of them is used in RQ2.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import color

from .imagestats import BLACK_LEVEL

SPACES = {
    "RGB": ["R", "G", "B"],
    "HSV": ["H", "S", "V"],
    "CIE Lab": ["L", "a", "b"],
    "YCbCr": ["Y", "Cb", "Cr"],
    "Giemsa (methyl blue, eosin)": ["giemsa_blue", "giemsa_eosin"],
    "H&E (haematoxylin, eosin)": ["hed_H", "hed_E"],
}
CHANNELS = [c for chans in SPACES.values() for c in chans]


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that are cell rather than black padding, as imagestats defines it."""
    tissue = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) >= BLACK_LEVEL
    return tissue if int(tissue.sum()) >= 100 else np.ones_like(tissue)


def channel_maps(rgb: np.ndarray) -> dict:
    """Every channel in CHANNELS as a float map. Hue is in degrees."""
    f = rgb.astype(np.float64) / 255.0
    hsv, lab, ycc = color.rgb2hsv(f), color.rgb2lab(f), color.rgb2ycbcr(f)
    bex = color.separate_stains(f, color.bex_from_rgb)
    hed = color.rgb2hed(f)
    return {
        "R": rgb[..., 0].astype(float), "G": rgb[..., 1].astype(float),
        "B": rgb[..., 2].astype(float),
        "H": hsv[..., 0] * 360.0, "S": hsv[..., 1], "V": hsv[..., 2],
        "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2],
        "Y": ycc[..., 0], "Cb": ycc[..., 1], "Cr": ycc[..., 2],
        "giemsa_blue": bex[..., 0], "giemsa_eosin": bex[..., 1],
        "hed_H": hed[..., 0], "hed_E": hed[..., 1],
    }


def tissue_channel_means(rgb: np.ndarray) -> dict:
    """Mean of every channel over tissue pixels.

    Hue is a circular mean weighted by saturation, because a pale, nearly grey
    pixel has an essentially arbitrary hue. Across cells, though, hue stays hard
    to summarise: NIH cells are either pink (hue near 330-360) or lavender (near
    240-280), so a median over a mix of patients jumps between the two depending
    on which patients were sampled. Compare hue as a signed angle from a
    reference, and lean on Lab a/b and the stain channels for the stain axis.
    """
    mask = tissue_mask(rgb)
    maps = channel_maps(rgb)
    out = {}
    for name, m in maps.items():
        v = m[mask]
        if name == "H":
            ang, wgt = np.deg2rad(v), maps["S"][mask]
            if wgt.sum() <= 0:
                wgt = np.ones_like(wgt)
            out[name] = float(np.rad2deg(np.arctan2((wgt * np.sin(ang)).sum(),
                                                    (wgt * np.cos(ang)).sum())) % 360)
        else:
            out[name] = float(v.mean())
    return out


# ------------------------------------------------------------------ transforms
def lab_moments(images) -> tuple:
    """Pooled tissue-pixel mean and std of L, a, b over a list of RGB crops."""
    pix = np.concatenate([color.rgb2lab(im.astype(np.float64) / 255.0)[tissue_mask(im)]
                          for im in images])
    return pix.mean(axis=0), pix.std(axis=0)


def reinhard(rgb: np.ndarray, target_mean, target_std) -> np.ndarray:
    """Reinhard et al. (2001) colour transfer on tissue pixels.

    Each Lab channel of the cell is standardised and rescaled to the target
    mean and std; padding stays black.
    """
    mask = tissue_mask(rgb)
    lab = color.rgb2lab(rgb.astype(np.float64) / 255.0)
    src = lab[mask]
    mu, sd = src.mean(axis=0), src.std(axis=0)
    sd[sd == 0] = 1.0
    lab[mask] = (src - mu) / sd * np.asarray(target_std) + np.asarray(target_mean)
    out = (np.clip(color.lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8)
    out[~mask] = 0
    return out


def hsv_adjust(rgb: np.ndarray, hue_shift_deg: float = 0.0,
               sat_scale: float = 1.0, val_scale: float = 1.0) -> np.ndarray:
    """Rotate hue and scale saturation and value on tissue pixels."""
    mask = tissue_mask(rgb)
    hsv = color.rgb2hsv(rgb.astype(np.float64) / 255.0)
    hsv[..., 0] = (hsv[..., 0] + hue_shift_deg / 360.0) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 1)
    out = (color.hsv2rgb(hsv) * 255).round().astype(np.uint8)
    out[~mask] = 0
    return out


def reference_quantiles(images, n_quantiles: int = 256) -> np.ndarray:
    """Per-channel RGB quantiles of pooled tissue pixels: the matching target."""
    pix = np.concatenate([im[tissue_mask(im)] for im in images]).astype(float)
    q = np.linspace(0, 100, n_quantiles)
    return np.stack([np.percentile(pix[:, c], q) for c in range(3)])


def match_to_reference(rgb: np.ndarray, ref_quantiles: np.ndarray) -> np.ndarray:
    """Histogram matching per RGB channel, tissue pixels to tissue pixels.

    skimage.exposure.match_histograms matches whole images, so the black padding
    of a masked crop would take part; here only tissue is matched.
    """
    mask = tissue_mask(rgb)
    out = rgb.copy()
    q = np.linspace(0, 100, ref_quantiles.shape[1])
    for c in range(3):
        src = rgb[..., c][mask].astype(float)
        src_q = np.percentile(src, q)
        # strictly increasing x for np.interp
        src_q = src_q + np.arange(len(q)) * 1e-6
        out[..., c][mask] = np.clip(np.interp(src, src_q, ref_quantiles[c]),
                                    0, 255).round().astype(np.uint8)
    out[~mask] = 0
    return out
