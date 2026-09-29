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
from scipy.stats import rankdata
from tqdm.auto import tqdm as _tqdm

from . import paths

# Pixels darker than this count as background padding rather than tissue. NIH
# cells are segmented onto pure black, so the measure separates a masked crop
# from an unmasked one; real stained tissue essentially never goes this dark.
BLACK_LEVEL = 15

# The brightness and sharpness measures are taken on the tissue interior: the
# tissue mask eroded by this many pixels. Erosion matters for the Laplacian,
# which reads a 3x3 neighbourhood, so a pixel sitting on the tissue boundary
# still sees padding and returns the mask edge rather than the optics. Three
# pixels clears that kernel with room to spare.
TISSUE_ERODE_PX = 3
_ERODE_KERNEL = np.ones((2 * TISSUE_ERODE_PX + 1,) * 2, np.uint8)

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
# Two windows, for two different failure modes. Colour uses the tissue mask
# (gray >= BLACK_LEVEL) over the whole crop, because a black padding pixel
# contributes S=0, H=0 and R-B=0 and so dilutes every colour average on a masked
# crop. Brightness and sharpness use the central disc intersected with that mask
# eroded by TISSUE_ERODE_PX: the disc keeps the sampled region comparable across
# datasets, the mask removes padding where a crop is small enough for it to
# reach inside the disc, and the erosion keeps a neighbourhood filter off the
# boundary.
CENTER_FEATURES = ["gray_mean_center", "gray_std_center", "lapvar224_center",
                   "r_mean_center", "g_mean_center", "b_mean_center",
                   "rb_diff_center", "sat_mean_center", "hue_mean_center"]


def _radial_masks(size: int):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / c
    return r > 0.875, r < 0.5   # outer ring, central disc


_OUTER, _CENTER = _radial_masks(paths.MODEL_INPUT)


def interior_mask(gray: np.ndarray) -> np.ndarray:
    """The window the `_center` features are measured on.

    The central disc intersected with the tissue mask, eroded so a
    neighbourhood filter never reads across the padding boundary. Exposed
    rather than inlined because the RQ2 sweep in section J has to measure
    degraded crops the same way; if it used a different window its curve would
    not be on the same scale as the reference lines it is compared against.
    """
    tissue = gray >= BLACK_LEVEL
    if int(tissue.sum()) < 100:
        tissue = np.ones_like(tissue)
    eroded = cv2.erode(tissue.astype(np.uint8), _ERODE_KERNEL).astype(bool)
    interior = eroded & _CENTER
    for fallback in (eroded, tissue):          # tiny or oddly shaped crops
        if int(interior.sum()) >= 100:
            break
        interior = fallback
    return interior


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

    # Window for the brightness and sharpness measures: the central disc
    # intersected with the eroded tissue mask. Both halves are needed.
    #
    # The disc keeps the comparison like-for-like. Without it the window on an
    # unmasked crop is the entire rectangle, neighbouring cells and background
    # included, so NIH's single segmented cell would be compared against a whole
    # field of view and the contrast difference would be image content rather
    # than imaging.
    #
    # The eroded tissue mask removes the padding. NIH cells vary in size, so on
    # about 7% of crops the padding reaches inside the disc, and the Laplacian
    # then returns the mask edge: those crops score a median near 197 against
    # about 6 for the rest, which inflates NIH's spread enough to make Cohen's d
    # report no sharpness difference where one exists. Erosion is what makes it
    # safe for a neighbourhood filter, which would otherwise read padding from a
    # pixel sitting on the boundary.
    interior = interior_mask(gray)
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

        # tissue-only versions of the three scale/format-sensitive measures.
        # The `_center` suffix is kept for continuity with earlier runs; the
        # window is the eroded tissue mask, not a central disc.
        "gray_mean_center": float(gray[interior].mean()),
        "gray_std_center": float(gray[interior].std()),
        "lapvar224_center": float(
            cv2.Laplacian(gray, cv2.CV_64F)[interior].var()),

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


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-based effect size: |2A - 1|, where A = P(a > b) over random pairs.

    0 means the two samples are indistinguishable on this feature; 1 means a
    single value tells them apart every time. Reported alongside `cohens_d`
    because d is unreliable on several of these features: Laplacian variance is
    heavily skewed, so one extreme crop inflates the pooled standard deviation
    and d reports "no difference" where the distributions barely overlap. A
    rank measure counts that crop once. It also needs no assumption about
    distribution shape and no per-feature transformation, so rows of the gap
    table stay comparable with each other, which a mix of logged and unlogged
    features would not.

    Numerically this is |2 * AUC - 1| for a one-feature classifier, the same
    quantity section H already reports for the domain classifier.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    ranks = rankdata(np.concatenate([a, b]))
    auc = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return float(abs(2 * auc - 1))


def domain_gap_table(stats: pd.DataFrame, reference: str = "nih",
                     group_col: str = "domain",
                     features=None, measure=None) -> pd.DataFrame:
    """Effect size for every feature, reference domain vs each other domain.

    `measure` defaults to `cohens_d`; pass `separation` for the rank-based
    version. Section H reports both, because they disagree about which feature
    separates the domains most and the disagreement is itself a finding.
    """
    measure = cohens_d if measure is None else measure
    features = list(FEATURES if features is None else features)
    ref = stats[stats[group_col] == reference]
    out = {}
    for name, grp in stats[stats[group_col] != reference].groupby(group_col,
                                                                 observed=True):
        out[str(name)] = {f: measure(np.asarray(grp[f]), np.asarray(ref[f]))
                          for f in features}
    table = pd.DataFrame(out)
    table["max_abs"] = table.abs().max(axis=1)
    return table.sort_values("max_abs", ascending=False)
