"""Put the test-set crops into the NIH format with a validated background method.

    python scripts/build_masked_crops.py --method otsu-gray              # the RQ2 test crops
    python scripts/build_masked_crops.py --method otsu-gray --limit 50   # trial run
    python scripts/build_masked_crops.py --method rembg:u2net --uninfected-per-split 3000
        # every parasitised cell, 3000 uninfected per split: enough for the EDA

For every BBBC041 and MP-IDB crop in the manifests: find the cell
(malaria/background.py), keep it, set everything else to 0, and cut the result
to the square tight around the kept cell, which is how cell_images crops are
cut. MP-IDB crops are seeded by their annotated parasite, and its pixels are
added back to the kept mask after their retention has been recorded, so no
method can delete the thing being classified.

Pick the method with scripts/validate_background.py first.

Writes
    data/crops/masked/<method>/<original path under data/crops>
    data/manifests/masked_<method>_cells.csv   dataset "<dataset>_masked", shared schema
    outputs/tables/masked_<method>_status.csv  status, kept fraction, parasite retention

In the manifest, r0/c0/r1/c1 are the tight square in source-image coordinates
and pad_frac is 0.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.background import (METHODS, cell_mask, foreground,  # noqa: E402
                                tight_box, to_nih_format)
from malaria.manifests import (COLUMNS, MpidbParasiteSeeds, _frame,  # noqa: E402
                               load_manifest, tqdm)

_SEEDS = None   # one per worker process


def slug(method: str) -> str:
    return method.replace(":", "_")


def _process(job) -> dict:
    global _SEEDS
    cell_id, rel, out_rel, method, species, source_image, window = job
    bgr = cv2.imread(str(paths.ROOT / rel), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rel)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    seed = None
    if species is not None:
        if _SEEDS is None:
            _SEEDS = MpidbParasiteSeeds()
        seed = _SEEDS(species, source_image, cell_id, window)

    kept, status = cell_mask(foreground(rgb, method), seed)
    retention = np.nan
    if seed is not None and seed.any():
        retention = float((kept & seed).sum() / seed.sum())
        kept = kept | seed

    box = tight_box(kept)
    out = to_nih_format(bgr, kept)
    dst = paths.ROOT / out_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), out):
        raise IOError(f"cv2.imwrite failed for {dst}")
    h, w = kept.shape
    return {"cell_id": cell_id, "status": status, "kept_frac": float(kept.mean()),
            "parasite_retention": retention,
            "dr0": box.r0 if box else 0, "dc0": box.c0 if box else 0,
            "dr1": box.r1 if box else h, "dc1": box.c1 if box else w}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    # Every RQ evaluates whole cells, so MP-IDB defaults to the host-cell framing;
    # "mpidb" (parasite close-ups) is kept for the EDA only.
    ap.add_argument("--datasets", nargs="+", default=["bbbc041", "mpidb_wholecell"],
                    choices=["bbbc041", "mpidb", "mpidb_wholecell"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="at most this many crops per source split, for a trial run")
    ap.add_argument("--uninfected-per-split", type=int, default=None,
                    help="keep every parasitised cell but only this many uninfected "
                         "cells per source split, primary set only (EDA sample)")
    args = ap.parse_args()

    # rembg gives each onnxruntime session OMP_NUM_THREADS threads when it is
    # set; without it every worker would claim every core.
    os.environ.setdefault("OMP_NUM_THREADS",
                          str(max(1, (os.cpu_count() or 1) // args.workers)))

    frames = []
    for ds in args.datasets:
        m = load_manifest(f"{ds}_cells.csv")
        if args.uninfected_per_split is not None:
            # U^2-Net takes about 0.57 s a crop on CPU, so the 80k uninfected
            # site_a cells alone are 13 CPU-hours; the EDA needs every parasite
            # but only a sample of the negatives.
            m = m[m["eval_group"] == "primary"]
            neg = m[m["label_binary"] == 0]
            if len(neg):
                neg = pd.concat([g.sample(min(len(g), args.uninfected_per_split),
                                          random_state=0)
                                 for _, g in neg.groupby("source_split")])
            m = pd.concat([m[m["label_binary"] == 1], neg])
        if args.limit:
            m = pd.concat([g.sample(min(len(g), args.limit), random_state=0)
                           for _, g in m.groupby("source_split")])
        frames.append(m)
    cells = pd.concat(frames, ignore_index=True)

    crops_root = paths.CROPS.relative_to(paths.ROOT).as_posix()
    masked_root = (paths.CROPS_MASKED / slug(args.method)).relative_to(paths.ROOT).as_posix()
    out_rel = [f"{masked_root}/{p[len(crops_root) + 1:]}"
               for p in cells["path"]]
    jobs = [(r.cell_id, r.path, o, args.method,
             str(r.species) if r.dataset in ("mpidb", "mpidb_wholecell") else None,
             r.source_image,
             (int(r.r0), int(r.c0), int(r.r1), int(r.c1)))
            for r, o in zip(cells.itertuples(index=False), out_rel)]

    print(f"{args.method}: {len(jobs):,} crops on {args.workers} workers "
          f"(OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']})")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        status = pd.DataFrame(list(tqdm(ex.map(_process, jobs, chunksize=8),
                                        total=len(jobs))))

    # ex.map keeps job order, so status lines up with cells row by row; cell_id
    # alone is not unique (mpidb and mpidb_wholecell share their cell_ids).
    assert (status["cell_id"].to_numpy() == cells["cell_id"].to_numpy()).all()
    st = status.set_index("cell_id")
    masked = cells.copy()
    masked["dataset"] = masked["dataset"].astype(str) + "_masked"
    masked["path"] = out_rel
    r0, c0 = masked["r0"].astype(int).to_numpy(), masked["c0"].astype(int).to_numpy()
    masked["r0"], masked["r1"] = r0 + st["dr0"].to_numpy(), r0 + st["dr1"].to_numpy()
    masked["c0"], masked["c1"] = c0 + st["dc0"].to_numpy(), c0 + st["dc1"].to_numpy()
    masked["pad_frac"] = 0.0
    masked = _frame(masked[COLUMNS].to_dict("records"))

    name = slug(args.method) + ("_trial" if args.limit is not None else
                                "_sample" if args.uninfected_per_split is not None
                                else "")
    masked.to_csv(paths.MANIFESTS / f"masked_{name}_cells.csv", index=False)
    report = cells[["dataset", "source_split", "cell_id", "label_binary",
                    "eval_group"]].reset_index(drop=True)
    report = pd.concat([report, st.drop(columns=["dr0", "dc0", "dr1", "dc1"])
                        .reset_index(drop=True)], axis=1)
    report.to_csv(paths.TABLES / f"masked_{name}_status.csv", index=False)

    print(pd.crosstab([report.dataset, report.source_split], report.status).to_string())
    print(report.groupby(["dataset", "source_split"])
          [["kept_frac", "parasite_retention"]].median().round(3).to_string())
    print(f"wrote {paths.MANIFESTS / f'masked_{name}_cells.csv'}")


if __name__ == "__main__":
    main()
