"""Colour-match the background-removed test crops to NIH: the second RQ2 step.

    python scripts/build_colour_crops.py                     # both methods, every cell
    python scripts/build_colour_crops.py --method reinhard --limit 50   # trial run

Input is the output of build_masked_crops.py (`masked_otsu-gray_cells.csv`):
test cells already in the NIH format, background set to black. Each crop's
tissue pixels are then mapped toward NIH colour with one of two methods from
malaria/colour.py, black padding untouched:

reinhard   Reinhard et al. (2001): match the mean and std of L, a and b.
histmatch  per-channel RGB histogram matching to the pooled NIH quantiles.

Both are reported in RQ2; neither is chosen on test results, since picking
the better one by its test score would be a form of leakage.

The NIH reference comes from the *train* split only (a fixed random sample of
REFERENCE_CELLS cells, both classes), so no NIH hold-out cell influences the
test inputs it is later compared with. The reference statistics are written to
outputs/tables/colour_reference.json.

Writes
    data/crops/colour/<method>/<path under data/crops/masked/otsu-gray>
    data/manifests/colour_<method>_cells.csv   dataset "<dataset>_<method>"
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.colour import (lab_moments, match_to_reference,  # noqa: E402
                            reference_quantiles, reinhard)
from malaria.data import nih_cells  # noqa: E402
from malaria.manifests import COLUMNS, _frame, load_manifest, tqdm  # noqa: E402

METHODS = ("reinhard", "histmatch")
SOURCE = "masked_otsu-gray_cells.csv"
REFERENCE_CELLS = 1000
MASKED_ROOT = "data/crops/masked/otsu-gray"

_REF = None   # set once per worker process


def _read_rgb(rel: str) -> np.ndarray:
    bgr = cv2.imread(str(paths.ROOT / rel), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rel)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def nih_reference(seed: int = 0) -> dict:
    """Lab moments and RGB quantiles of pooled tissue pixels of NIH train cells."""
    train = nih_cells("train").sample(REFERENCE_CELLS, random_state=seed)
    images = [_read_rgb(p) for p in train["path"]]
    mean, std = lab_moments(images)
    return {"n_cells": REFERENCE_CELLS, "seed": seed,
            "lab_mean": mean.tolist(), "lab_std": std.tolist(),
            "rgb_quantiles": reference_quantiles(images).tolist()}


def _init(ref: dict) -> None:
    global _REF
    _REF = ref


def _process(job) -> None:
    src, dst, method = job
    rgb = _read_rgb(src)
    if method == "reinhard":
        out = reinhard(rgb, _REF["lab_mean"], _REF["lab_std"])
    else:
        out = match_to_reference(rgb, np.asarray(_REF["rgb_quantiles"]))
    path = paths.ROOT / dst
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)):
        raise IOError(f"cv2.imwrite failed for {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="at most this many crops per source split, for a trial run")
    args = ap.parse_args()

    ref = nih_reference()
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    (paths.TABLES / "colour_reference.json").write_text(json.dumps(ref))
    print(f"NIH reference from {ref['n_cells']} train cells: "
          f"Lab mean {np.round(ref['lab_mean'], 2)}, std {np.round(ref['lab_std'], 2)}")

    cells = load_manifest(SOURCE)
    if args.limit:
        cells = pd.concat([g.sample(min(len(g), args.limit), random_state=0)
                           for _, g in cells.groupby("source_split")])
    bad = ~cells["path"].str.startswith(MASKED_ROOT + "/")
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} rows outside {MASKED_ROOT}")

    for method in args.method:
        out_rel = [f"data/crops/colour/{method}/{p[len(MASKED_ROOT) + 1:]}"
                   for p in cells["path"]]
        jobs = list(zip(cells["path"], out_rel, [method] * len(cells)))
        print(f"{method}: {len(jobs):,} crops on {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(ref,)) as ex:
            list(tqdm(ex.map(_process, jobs, chunksize=32), total=len(jobs)))

        out = cells.copy()
        out["dataset"] = out["dataset"].str.replace("_masked$", f"_{method}", regex=True)
        out["path"] = out_rel
        out = _frame(out[COLUMNS].to_dict("records"))
        name = f"colour_{method}" + ("_trial" if args.limit else "") + "_cells.csv"
        out.to_csv(paths.MANIFESTS / name, index=False)
        assert len(out) == len(cells), "row count changed"
        print(f"wrote {paths.MANIFESTS / name}")


if __name__ == "__main__":
    main()
