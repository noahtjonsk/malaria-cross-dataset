"""The NIH train / validation / test split.

Every cross-dataset drop this project reports is measured against the NIH
held-out score, so that baseline has to be honest. NIH filenames encode the
patient the slide came from (the C### prefix): 200 patients contribute between
65 and 702 cells each, 150 of them with parasitised cells and 50 with uninfected
cells only. A random split therefore puts cells from the same slide,
stained in the same session and photographed under the same lighting, on both
sides of the split, and the held-out score measures memorisation of a patient
rather than recognition of a parasite.

The split is grouped by patient, computed once, and written to CSV so that
MobileNetV2, ResNet-50 and VGG-16 all train on identical rows -- which RQ1
requires, since it attributes performance differences to architecture.

Because the per-patient cell counts are so skewed, a random grouped split gives
uneven class balance across folds. `patient_grouped_split` therefore assigns
patients greedily, largest first, always to whichever split is furthest below
its target size, which keeps the proportions close without ever splitting a
patient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths


def patient_grouped_split(manifest: pd.DataFrame,
                          fractions=(0.70, 0.15, 0.15),
                          names=("train", "val", "test"),
                          seed: int = 42) -> pd.Series:
    """Assign each NIH cell to a split without splitting any patient.

    Returns a Series of split names aligned to `manifest`.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    rng = np.random.default_rng(seed)
    counts = manifest.groupby("patient_id", observed=True).size()
    # Shuffle first so ties among equally-sized patients break randomly, then
    # sort stably so the shuffle survives as the tie-break order.
    counts = counts.sample(frac=1.0, random_state=int(rng.integers(1_000_000)))
    counts = counts.sort_values(ascending=False, kind="mergesort")  # stable sort

    total = int(counts.sum())
    targets = {n: f * total for n, f in zip(names, fractions)}
    filled = {n: 0 for n in names}
    assign: dict[str, str] = {}

    for patient, k in counts.items():
        # deficit relative to target, as a fraction of the target, so the
        # small splits are not starved by the large one
        pick = max(names, key=lambda n: (targets[n] - filled[n]) / targets[n])
        assign[str(patient)] = pick
        filled[pick] += int(k)

    return pd.Series(manifest["patient_id"].astype(str).map(assign),
                     index=manifest.index, name="split")


def build_nih_split(manifest: pd.DataFrame, **kw) -> pd.DataFrame:
    """Add a `split` column to the NIH rows and write it to disk."""
    nih = pd.DataFrame(manifest[manifest["dataset"] == "nih"]).copy()
    nih["split"] = patient_grouped_split(nih, **kw)

    paths.MANIFESTS.mkdir(parents=True, exist_ok=True)
    nih.loc[:, ["cell_id", "patient_id", "label_binary", "split"]].to_csv(
        paths.MANIFESTS / "nih_split.csv", index=False)
    return nih


def split_summary(nih: pd.DataFrame) -> pd.DataFrame:
    g = nih.groupby("split", observed=True)
    return pd.DataFrame({
        "cells": g.size(),
        "patients": g["patient_id"].nunique(),
        "share_of_cells": g.size() / len(nih),
        "parasitised_rate": g["label_binary"].mean(),
    }).sort_values("cells", ascending=False)


def inherit_nih_split(cells: pd.DataFrame) -> pd.Series:
    """The split for cells recut from the NIH photographs.

    NIH-NLM-ThinBloodSmearsPf photographs the same patients as cell_images, so a
    recut cell must land on the same side of the split as its patient already
    does in nih_split.csv; assigning it afresh would leak patients across splits.
    Raises if a patient has no entry, rather than silently dropping its cells.
    """
    pinned = pd.read_csv(paths.MANIFESTS / "nih_split.csv",
                         dtype={"patient_id": "string"})
    per_patient = pinned.drop_duplicates("patient_id").set_index("patient_id")["split"]
    split = cells["patient_id"].map(per_patient)
    missing = sorted(set(cells.loc[split.isna(), "patient_id"].astype(str)))
    if missing:
        raise KeyError(f"patients with no entry in nih_split.csv: {missing}")
    return split.rename("split")


def check_no_patient_leakage(nih: pd.DataFrame) -> None:
    """Raise if any patient appears in more than one split."""
    per = nih.groupby("patient_id", observed=True)["split"].nunique()
    bad = pd.Series(per)[pd.Series(per) > 1]
    if len(bad):
        raise AssertionError(f"patients in >1 split: {list(bad.index)}")
