"""The cropping rule.

NIH ships pre-cropped single cells; BBBC041 and MP-IDB do not. Cutting cells out
of the two test sets is therefore a preprocessing step introduced by this project
rather than a property of the source data, so the rule is fixed in advance, applied
identically to every cell in both test sets, and documented here for the write-up.

THE RULE
--------
Given a cell's bounding box (r0, c0, r1, c1) in a source image:

1. Take the box centre and the longer of its two sides, s.
2. Form a square of side s about that centre, so no cell is anisotropically
   distorted when it is later resized to a square network input. The four
   edges are rounded to integer pixels independently, so the window is square
   to within 1 px: about 40% of crops come out one pixel taller or wider than
   they are wide or tall. At a 224 px input that is a distortion of under 1%.
3. Grow the square by CROP_PAD_FRAC (10%) of s on every side, to keep a little
   of the surrounding field rather than cutting flush to the annotation.
4. Clip to the image bounds. Cells near an edge therefore come out smaller and
   possibly non-square; they are kept and flagged `at_border=True` rather than
   dropped, because dropping them would silently remove parasites from the
   test set.
5. Write the patch to PNG at its native size. Resizing to the network input
   happens at load time, so the size distribution stays measurable and the
   input resolution stays changeable later.

WHERE THE BOXES COME FROM
-------------------------
BBBC041  the `bounding_box` field of each object in training.json / test.json.
MP-IDB   connected components (8-connectivity, area >= MIN_MASK_AREA) of the
         binary parasite mask in gt/. This reproduces the dataset authors' own
         parasite decomposition: on every one of the 144 images that ship crops,
         the component count equals the shipped crop count.

WHAT THE RULE DELIBERATELY DOES NOT DO
--------------------------------------
It does not remove the background. NIH cells are segmented onto black, and
MP-IDB's shipped Falciparum crops are too, but BBBC041 provides bounding boxes
only and so cannot be masked at all. Masking MP-IDB alone would make Falciparum
crops resemble the NIH training data while Vivax crops did not, which would
confound the per-species comparison in RQ3 with crop format. Both test sets
therefore get plain unmasked rectangles. The resulting NIH-masked /
test-unmasked difference is real, is a property of the sources, and is measured
and reported rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .paths import CROP_PAD_FRAC, MIN_MASK_AREA


@dataclass(frozen=True)
class CropBox:
    """A resolved crop window in source-image pixel coordinates."""
    r0: int
    c0: int
    r1: int
    c1: int
    at_border: bool

    @property
    def height(self) -> int:
        return self.r1 - self.r0

    @property
    def width(self) -> int:
        return self.c1 - self.c0

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[self.r0:self.r1, self.c0:self.c1]


def square_padded_box(r0: int, c0: int, r1: int, c1: int,
                      img_h: int, img_w: int,
                      pad_frac: float = CROP_PAD_FRAC) -> CropBox:
    """Steps 1-4 of the rule. Coordinates are half-open: [r0, r1) x [c0, c1)."""
    side = max(r1 - r0, c1 - c0)
    half = (side * (1.0 + 2.0 * pad_frac)) / 2.0
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0

    want_r0, want_r1 = int(round(cr - half)), int(round(cr + half))
    want_c0, want_c1 = int(round(cc - half)), int(round(cc + half))

    got_r0, got_r1 = max(0, want_r0), min(img_h, want_r1)
    got_c0, got_c1 = max(0, want_c0), min(img_w, want_c1)

    at_border = (got_r0, got_r1, got_c0, got_c1) != (want_r0, want_r1, want_c0, want_c1)
    return CropBox(got_r0, got_c0, got_r1, got_c1, at_border)


def mask_components(mask: np.ndarray, min_area: int = MIN_MASK_AREA):
    """Parasite bounding boxes from an MP-IDB gt mask, ordered left to right.

    Left-to-right order matches how the dataset authors indexed their own crops,
    which is what lets the expert stage labels be carried across (see manifests).

    Yields (r0, c0, r1, c1, area) tuples with half-open coordinates.
    """
    binary = (mask > 127).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comps = []
    for i in range(1, n):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        c0 = int(stats[i, cv2.CC_STAT_LEFT])
        r0 = int(stats[i, cv2.CC_STAT_TOP])
        c1 = c0 + int(stats[i, cv2.CC_STAT_WIDTH])
        r1 = r0 + int(stats[i, cv2.CC_STAT_HEIGHT])
        comps.append((float(centroids[i][0]), (r0, c0, r1, c1, area)))

    comps.sort(key=lambda t: t[0])  # by centroid x
    return [c for _, c in comps]


def read_gray_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return m
