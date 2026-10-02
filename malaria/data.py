"""Cells as model input: one torch Dataset over any manifest.

Every stage after the manifests reads cells through these tables (manifests.py),
and the models do too: a Dataset is built from manifest rows (`path`,
`label_binary`), so the training cells, the NIH hold-out set and every test set
variant (raw, background removed, colour matched) go through the same code.

Input rule, identical for every dataset:

1. Pad the crop to a square with black. NIH crops are cut tight to the cell and
   are not square; stretching them would change the cell's shape, and black is
   the padding NIH cells already sit on. Raw test crops are already square
   (crops.square_padded_box), so padding leaves them unchanged.
2. Resize to paths.MODEL_INPUT (224), the ImageNet input size the pretrained
   weights expect.
3. Normalise with the ImageNet channel mean and std, for the same reason.

Training augmentation is geometric only: horizontal and vertical flips and
rotations by multiples of 90 degrees. A cell turned upside down is still the
same cell, and none of these change colour, brightness or the black frame.
There is deliberately no colour or brightness jitter: RQ2 matches test-cell
colour to NIH at test time, and a model that had already been taught to ignore
colour in training would leave less of the drop for that step to recover, by an
amount set by an arbitrary jitter strength. (The proposal listed "minor
brightness adjustments"; this is the one departure, decided 29 Sept 2026.)
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import paths
from .manifests import load_manifest
from .splits import check_no_patient_leakage

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------ which cells
# Test-set variants. "raw" is the crop as cut from the source photograph (RQ1);
# the others are the RQ2 steps, built by build_masked_crops.py (background
# removed) and build_colour_crops.py (then colour matched to NIH train cells).
VARIANT_MANIFESTS = {
    "raw": "{dataset}_cells.csv",
    "masked": "masked_otsu-gray_cells.csv",
    "reinhard": "colour_reinhard_cells.csv",
    "histmatch": "colour_histmatch_cells.csv",
}
VARIANTS = tuple(VARIANT_MANIFESTS)
VARIANT_SUFFIX = {"raw": "", "masked": "_masked", "reinhard": "_reinhard",
                  "histmatch": "_histmatch"}
TEST_DATASETS = ("bbbc041", "mpidb_wholecell")
NIH_SPLIT_SIZES = {"train": 19203, "val": 4178, "test": 4177}


def variant_manifest_files(variant: str) -> list:
    """The manifest files one variant's cells are read from (raw includes NIH)."""
    files = {VARIANT_MANIFESTS[variant].format(dataset=d) for d in TEST_DATASETS}
    if variant == "raw":
        files |= {"nih_cells.csv", "nih_split.csv"}
    return sorted(files)


def nih_cells(split: str | None = None) -> pd.DataFrame:
    """NIH cell_images rows with their pinned patient-grouped split.

    Checks, before any model sees a cell, that the split file covers every NIH
    cell, that no patient crosses splits, and that the split sizes are the
    pinned ones.
    """
    cells = load_manifest("nih_cells.csv")
    pinned = pd.read_csv(paths.MANIFESTS / "nih_split.csv", dtype={"cell_id": "string"})
    cells = cells.merge(pinned[["cell_id", "split"]], on="cell_id", how="left",
                        validate="one_to_one")
    if cells["split"].isna().any():
        raise AssertionError(f"{int(cells['split'].isna().sum())} NIH cells have no split")
    check_no_patient_leakage(cells)
    sizes = cells["split"].value_counts().to_dict()
    if sizes != NIH_SPLIT_SIZES:
        raise AssertionError(f"NIH split sizes {sizes} != pinned {NIH_SPLIT_SIZES}")
    return cells if split is None else cells[cells["split"] == split].reset_index(drop=True)


def test_cells(dataset: str, variant: str = "raw") -> pd.DataFrame:
    """Primary-metric cells of one external test set in one RQ2 variant.

    Only `eval_group == "primary"` rows: BBBC041 difficult and leukocyte cells
    are excluded from every metric (manifests.py explains why).
    """
    if dataset not in TEST_DATASETS:
        raise ValueError(f"unknown test set {dataset!r}")
    m = load_manifest(VARIANT_MANIFESTS[variant].format(dataset=dataset))
    m = m[(m["dataset"] == dataset + VARIANT_SUFFIX[variant]) & (m["eval_group"] == "primary")]
    if not len(m):
        raise AssertionError(f"no primary {dataset} cells in the {variant} manifest")
    return m.reset_index(drop=True)


# ------------------------------------------------------------ one image
def pad_to_square(rgb: np.ndarray) -> np.ndarray:
    """Centre the crop on a black square as wide as its longer side."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=rgb.dtype)
    r0, c0 = (side - h) // 2, (side - w) // 2
    out[r0:r0 + h, c0:c0 + w] = rgb
    return out


def load_input(path, size: int = paths.MODEL_INPUT) -> np.ndarray:
    """Read a crop as RGB uint8, padded to a square and resized to `size`."""
    bgr = cv2.imread(str(paths.ROOT / path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = pad_to_square(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flips and a random multiple-of-90-degree rotation."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    if rng.random() < 0.5:
        rgb = rgb[::-1, :]
    return np.rot90(rgb, k=int(rng.integers(4)))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class CellDataset(Dataset):
    """Manifest rows -> (image tensor, label, row index).

    The row index lets predictions be joined back to the manifest (cell_id,
    source_image, stage, species) without trusting the loader's ordering.
    """

    def __init__(self, cells: pd.DataFrame, train: bool = False, seed: int = 0):
        if cells["label_binary"].isna().any():
            raise ValueError("unlabelled rows (e.g. BBBC041 difficult/leukocyte) "
                             "must be filtered out before building a dataset")
        self.paths = cells["path"].astype(str).tolist()
        self.labels = cells["label_binary"].astype(int).to_numpy()
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rgb = load_input(self.paths[i])
        if self.train:
            # Seeded from torch, which gives every DataLoader worker its own
            # seed, so workers do not repeat each other's flips and rotations.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            rgb = augment(rgb, rng)
        return to_tensor(rgb), torch.tensor(self.labels[i], dtype=torch.float32), i
