"""Background removal: putting a test-set crop into the NIH format.

NIH cells come segmented onto black, while BBBC041 and MP-IDB crops keep the
surrounding plasma and neighbouring cells. The EDA ranked that as the largest
difference between the training and test inputs, so a cross-dataset drop
measured on unmodified crops mixes crop format with imaging. This module finds
the cell in a crop, keeps it, and sets everything else to 0.

Two families of method are compared rather than one assumed:

rembg:<model>  general-purpose salient-object models (U^2-Net, IS-Net,
               BiRefNet). They were trained on everyday photographs, not blood
               smears, so none is used before scripts/validate_background.py has
               scored it against the hand-drawn NIH outlines and checked that it
               keeps MP-IDB parasite pixels. rembg returns a soft mask; it is cut
               at 0.5 unless the name carries a threshold, as in
               rembg:u2net@0.1, because on pale cells the model is rarely that
               sure of anything.
otsu-gray      Otsu threshold on brightness: a stained red cell is darker than
               the plasma around it.
otsu-sat       Otsu threshold on HSV saturation: plasma is close to grey.

rembg's `bria-rmbg` model is deliberately not offered: its weights are licensed
for non-commercial use only, which a thesis repository should not depend on.

Every method yields a foreground map; `cell_mask` then keeps only the part
belonging to the cell of interest -- the component under the crop centre, or,
for MP-IDB, the component(s) touching the annotated parasite -- and fills its
holes, since a pale central pallor or a ring parasite is still part of the cell.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np
from scipy import ndimage as ndi

from .crops import square_padded_box

REMBG_MODELS = ("u2net", "isnet-general-use", "birefnet-general")
METHODS = ("otsu-gray", "otsu-sat") + tuple(f"rembg:{m}" for m in REMBG_MODELS)
# Measured on this machine's CPU, per crop: Otsu 0.002 s, U^2-Net 0.57 s,
# IS-Net 1.5 s, BiRefNet 31 s. BiRefNet would need about 750 CPU-hours for the
# 87k test crops, so it stays available by name but is left out of the default
# comparison.
DEFAULT_METHODS = tuple(m for m in METHODS if m != "rembg:birefnet-general")

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@functools.lru_cache(maxsize=None)
def _rembg_session(model: str):
    from rembg import new_session   # optional dependency, imported on first use
    return new_session(model)


def foreground(rgb: np.ndarray, method: str) -> np.ndarray:
    """Boolean foreground map for an RGB uint8 crop, before any clean-up."""
    if method.startswith("rembg:"):
        from rembg import remove
        model, _, threshold = method.split(":", 1)[1].partition("@")
        cut = round(255 * (float(threshold) if threshold else 0.5))
        alpha = np.asarray(remove(rgb, session=_rembg_session(model), only_mask=True))
        if alpha.ndim == 3:
            alpha = alpha[..., -1]
        return alpha >= max(1, cut)
    if method == "otsu-gray":
        channel, invert = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), True
    elif method == "otsu-sat":
        channel, invert = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1], False
    else:
        raise ValueError(f"unknown background method: {method!r}; "
                         f"choose from {METHODS}")
    channel = cv2.GaussianBlur(channel, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(channel, 0, 255, mode + cv2.THRESH_OTSU)
    return th > 0


def cell_mask(fg: np.ndarray, seed: np.ndarray | None = None):
    """Keep the cell of interest in a foreground map.

    `seed` marks pixels known to belong to the cell (the MP-IDB parasite mask).
    Without one the crop centre is used, since every crop is centred on its
    annotation. Returns (mask, status), status one of:

    seed     component(s) overlapping the seed were kept
    centre   the component under the crop centre was kept
    anchor   no component overlapped the seed; the one under its centroid was kept
    nearest  nothing covered the anchor, so the closest component was kept. It
             may be a neighbouring cell, so summarise_results.py repeats RQ2
             without these cells as a robustness check.
    empty    the method found no foreground at all; the mask is empty
    """
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _KERNEL)
    fg = ndi.binary_fill_holes(fg)
    n, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    h, w = fg.shape
    if n <= 1:
        return np.zeros((h, w), bool), "empty"

    if seed is not None and seed.any():
        keep = np.unique(labels[seed & (labels > 0)])
        if keep.size:
            return ndi.binary_fill_holes(np.isin(labels, keep)), "seed"
        anchor = np.argwhere(seed).mean(axis=0).round().astype(int)
    else:
        anchor = np.array([h // 2, w // 2])

    r, c = anchor
    if labels[r, c] > 0:
        return labels == labels[r, c], "centre" if seed is None else "anchor"
    # nearest foreground pixel to the anchor
    _, (ir, ic) = ndi.distance_transform_edt(labels == 0, return_indices=True)
    return labels == labels[ir[r, c], ic[r, c]], "nearest"


def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set everything outside `mask` to 0, the NIH padding value."""
    out = rgb.copy()
    out[~mask] = 0
    return out


def tight_box(mask: np.ndarray):
    """The square cut tight around `mask`, as a CropBox, or None if it is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return square_padded_box(int(rows[0]), int(cols[0]), int(rows[-1]) + 1,
                             int(cols[-1]) + 1, *mask.shape[:2], pad_frac=0.0)


def to_nih_format(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The kept cell on black, re-cut to the square tight around it.

    Segmentation runs on the padded crop, because the surrounding plasma is what
    lets a method find the cell edge. cell_images crops, however, are cut tight
    to the cell, so the result is trimmed to the square around the kept mask
    before it is compared with them; otherwise the padding alone would double the
    black fraction.
    """
    box = tight_box(mask)
    out = apply_mask(rgb, mask)
    return out if box is None else box.apply(out)


def split_touching(fg: np.ndarray, cell_side: int) -> np.ndarray:
    """Label image in which touching cells are cut apart.

    An Otsu threshold leaves cells that touch as one component. The distance
    transform of the cleaned foreground peaks once inside each cell, so a
    watershed seeded at peaks at least 0.35 cell widths apart cuts a clump along
    the narrow necks where cells meet. Returns 0 for background and one positive
    label per cell.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _KERNEL)
    fg = ndi.binary_fill_holes(fg)
    if not fg.any():
        return np.zeros(fg.shape, np.int32)
    dist = ndi.distance_transform_edt(fg)
    components, _ = ndi.label(fg)
    peaks = peak_local_max(dist, min_distance=max(3, int(0.35 * cell_side)),
                           labels=components, exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=fg).astype(np.int32)


def host_cell_box(rgb: np.ndarray, parasites: np.ndarray, parasite_box,
                  cell_side: int, pad_frac: float, max_cells: float = 2.2):
    """The crop window around the red blood cell hosting one parasite.

    MP-IDB annotates parasites, not the cells they sit in, so a crop cut around
    the parasite mask is a close-up that leaves the cell edge outside, unlike the
    whole-cell crops of NIH and BBBC041. This finds the host cell instead. A square
    window three cell widths across is centred on the parasite; the cells in it
    are segmented with the Otsu brightness threshold (the method validated on the
    NIH outlines) and touching cells are cut apart (`split_touching`); the cell
    region(s) holding at least a fifth of the parasite's pixels are kept; and the
    crop is the padded square around them and the parasite together. A kept region
    wider than `max_cells` cell widths is taken for touching cells that did not
    split.

    `rgb` is the whole image, `parasites` its boolean parasite mask (all
    parasites), `parasite_box` this parasite's (r0, c0, r1, c1).

    Returns (CropBox, outcome, host_side). outcome is "host_cell", or a fallback
    when the window has no foreground ("fallback_empty"), the parasite lies in no
    cell region ("fallback_missed"), or the kept region is wider than `max_cells`
    cell widths, i.e. the split failed ("fallback_clump"). A fallback crop is a square of
    max(cell_side, parasite side) centred on the parasite.
    """
    h, w = rgb.shape[:2]
    r0, c0, r1, c1 = (int(v) for v in parasite_box)
    p_side = max(r1 - r0, c1 - c0)
    cr, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0

    def centred(side, pad):
        half = side / 2.0
        return square_padded_box(int(round(cr - half)), int(round(cc - half)),
                                 int(round(cr + half)), int(round(cc + half)),
                                 h, w, pad_frac=pad)

    win = centred(max(3 * cell_side, int(round(1.5 * p_side))), 0.0)
    seed = np.zeros((win.height, win.width), bool)
    sr0, sc0 = max(r0, win.r0), max(c0, win.c0)
    sr1, sc1 = min(r1, win.r1), min(c1, win.c1)
    seed[sr0 - win.r0:sr1 - win.r0, sc0 - win.c0:sc1 - win.c0] = (
        parasites[sr0:sr1, sc0:sc1])

    labels = split_touching(foreground(win.apply(rgb), "otsu-gray"), cell_side)
    kept = np.zeros(labels.shape, bool)
    overlap = np.bincount(labels[seed], minlength=int(labels.max()) + 1)[1:]
    if overlap.sum() > 0:
        chosen = np.flatnonzero(overlap >= max(1, 0.2 * overlap.sum())) + 1
        kept = ndi.binary_fill_holes(np.isin(labels, chosen))
    rows = np.flatnonzero(kept.any(axis=1))
    cols = np.flatnonzero(kept.any(axis=0))
    host_side = (0 if rows.size == 0 else
                 int(max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)))

    if rows.size == 0:
        outcome = "fallback_empty" if labels.max() == 0 else "fallback_missed"
    elif host_side > max_cells * cell_side:
        outcome = "fallback_clump"
    else:
        box = square_padded_box(min(win.r0 + int(rows[0]), r0),
                                min(win.c0 + int(cols[0]), c0),
                                max(win.r0 + int(rows[-1]) + 1, r1),
                                max(win.c0 + int(cols[-1]) + 1, c1),
                                h, w, pad_frac=pad_frac)
        return box, "host_cell", host_side
    return centred(max(cell_side, p_side), pad_frac), outcome, host_side


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * int((a & b).sum()) / total


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = int((a | b).sum())
    return 1.0 if union == 0 else int((a & b).sum()) / union
