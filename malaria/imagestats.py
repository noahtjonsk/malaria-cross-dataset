"""Per-image statistics for the EDA.

The features are grouped to match the axes along which the three datasets are
claimed to differ: size, illumination, staining colour, sharpness, and crop
format. Each is cheap enough to compute over all 115k cells and is written to
CSV once, so figures and quoted numbers can be regenerated without touching the
images again.

A note on sharpness. Laplacian variance is the standard no-reference blur
measure, but on raw images it is dominated by pixel dimensions and JPEG
quantisation, so a large sharp slide and a small sharp crop are not comparable.
Every scale-dependent statistic here is therefore computed after resizing to
MODEL_INPUT (224), which is what the network actually sees. `lapvar_native` is
kept alongside it as a footnote, not as the headline number.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm

from . import paths

# Pixels darker than this count as background padding rather than tissue. NIH
# cells are segmented onto pure black, so the measure separates a masked crop
# from an unmasked one; real stained tissue essentially never goes this dark.
BLACK_LEVEL = 15

FEATURES = [
    "w", "h", "area", "aspect", "upscale_factor",
    "gray_mean", "gray_std", "gray_p5", "gray_p50", "gray_p95",
    "r_mean", "g_mean", "b_mean", "rb_diff", "sat_mean", "hue_mean",
    "lapvar224", "lapvar_native",
    "black_frac", "black_frac_outer", "black_frac_center",
    "gray_mean_center", "gray_std_center", "lapvar224_center",
    "r_mean_center", "g_mean_center", "b_mean_center",
    "rb_diff_center", "sat_mean_center", "hue_mean_center",
]

# Features measured on tissue only. NIH cells are segmented onto black, so
# roughly a quarter of every NIH crop is padding. Whole-crop statistics
# therefore describe that padding as much as the cell, and comparing them
# across datasets compares crop format rather than imaging. The centre
# variants are the honest cross-dataset comparison; the whole-crop ones remain
# the right description of what the network is actually fed.
#
# Two masks are used. The gray/sharpness variants use the central disc, which
# is geometry-based. The colour variants use the tissue mask
# (gray >= BLACK_LEVEL), because a black padding pixel contributes S=0, H=0
# and R-B=0 and so dilutes every colour average on a masked crop.
CENTER_FEATURES = ["gray_mean_center", "gray_std_center", "lapvar224_center",
                   "r_mean_center", "g_mean_center", "b_mean_center",
                   "rb_diff_center", "sat_mean_center", "hue_mean_center"]


def _radial_masks(size: int):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c
    return r > 0.875, r < 0.5   # outer ring, central disc


_OUTER, _CENTER = _radial_masks(paths.MODEL_INPUT)


def image_stats(path) -> dict:
    """Every feature in FEATURES for a single image."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = bgr.shape[:2]

    gray_native = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    n = paths.MODEL_INPUT
    small = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    b, g, r = (small[..., i].astype(np.float32).mean() for i in range(3))
    dark = gray < BLACK_LEVEL

    # Tissue mask for the colour features. Black padding contributes S=0, H=0
    # and R-B=0, which dilutes every colour average on a masked NIH crop while
    # leaving the unmasked test sets untouched: the same trap the _center
    # variants fix for brightness and sharpness. Fall back to the whole crop
    # if the mask degenerates (an almost entirely black image).
    tissue = ~dark
    if int(tissue.sum()) < 100:
        tissue = np.ones_like(dark)
    bt, gt, rt = (small[..., i].astype(np.float32)[tissue].mean()
                  for i in range(3))

    return {
        "w": w, "h": h, "area": w * h, "aspect": w / h,
        # how much interpolation this cell needs to reach the network input;
        # the resize is to a square, so the SHORT side is the one stretched
        "upscale_factor": n / min(w, h),

        "gray_mean": float(gray.mean()), "gray_std": float(gray.std()),
        "gray_p5": float(np.percentile(gray, 5)),
        "gray_p50": float(np.percentile(gray, 50)),
        "gray_p95": float(np.percentile(gray, 95)),

        "r_mean": float(r), "g_mean": float(g), "b_mean": float(b),
        "rb_diff": float(r - b),                  # Giemsa purple axis
        "sat_mean": float(hsv[..., 1].mean()),
        "hue_mean": float(hsv[..., 0].mean()),

        "lapvar224": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "lapvar_native": float(cv2.Laplacian(gray_native, cv2.CV_64F).var()),

        "black_frac": float(dark.mean()),
        "black_frac_outer": float(dark[_OUTER].mean()),
        "black_frac_center": float(dark[_CENTER].mean()),

        # tissue-only versions of the three scale/format-sensitive measures
        "gray_mean_center": float(gray[_CENTER].mean()),
        "gray_std_center": float(gray[_CENTER].std()),
        "lapvar224_center": float(cv2.Laplacian(gray, cv2.CV_64F)[_CENTER].var()),

        # tissue-only colour, masked on gray >= BLACK_LEVEL
        "r_mean_center": float(rt), "g_mean_center": float(gt),
        "b_mean_center": float(bt),
        "rb_diff_center": float(rt - bt),
        "sat_mean_center": float(hsv[..., 1][tissue].mean()),
        "hue_mean_center": float(hsv[..., 0][tissue].mean()),
    }


def stats_for_manifest(manifest: pd.DataFrame, root=None,
                       progress: bool = True) -> pd.DataFrame:
    """Compute `image_stats` for every row, returned aligned to the manifest."""
    root = paths.ROOT if root is None else root
    it = manifest["path"]
    if progress:
        it = _tqdm(it, total=len(manifest), desc="image stats")
    recs = [image_stats(root / p) for p in it]
    return pd.DataFrame(recs, index=manifest.index)


def attach_stats(manifest: pd.DataFrame, **kw) -> pd.DataFrame:
    """Manifest columns plus every feature, one row per cell."""
    return pd.concat([manifest, stats_for_manifest(manifest, **kw)], axis=1)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with a pooled standard deviation.

    Used to rank *which* image property separates two datasets most, so the RQ2
    degradations can be aimed at the factors that actually differ rather than
    chosen arbitrarily.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                     / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def domain_gap_table(stats: pd.DataFrame, reference: str = "nih",
                     group_col: str = "domain",
                     features=None) -> pd.DataFrame:
    """Cohen's d for every feature, reference domain vs each other domain."""
    features = list(FEATURES if features is None else features)
    ref = stats[stats[group_col] == reference]
    out = {}
    for name, grp in stats[stats[group_col] != reference].groupby(group_col,
                                                                 observed=True):
        out[str(name)] = {f: cohens_d(np.asarray(grp[f]), np.asarray(ref[f]))
                          for f in features}
    table = pd.DataFrame(out)
    table["max_abs_d"] = table.abs().max(axis=1)
    return table.sort_values("max_abs_d", ascending=False)
