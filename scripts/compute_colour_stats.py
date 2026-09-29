"""Tissue-pixel means of every colour channel for a sample of cells per domain.

    python scripts/compute_colour_stats.py --masked rembg_u2net_sample
    python scripts/compute_colour_stats.py --per-domain 400 --workers 8

The channels, and why each one is included, are documented in malaria/colour.py.

The test sets are read in the NIH format built by build_masked_crops.py, because
colour is averaged over tissue pixels and an unmasked crop has no padding to
exclude: its "tissue" would include the plasma and the neighbouring cells.
Without --masked the raw test crops are used, and the `format` column
says so.

Writes outputs/tables/colour_channel_stats.csv, one row per sampled cell.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.colour import CHANNELS, tissue_channel_means  # noqa: E402
from malaria.manifests import load_manifest  # noqa: E402

KEEP = ["domain", "format", "dataset", "source_split", "cell_id", "path",
        "label_binary", "stage", "species", "patient_id", "source_image"]


def _means(rel: str) -> dict:
    return tissue_channel_means(plots.load_rgb(rel, paths.MODEL_INPUT))


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to n cells, split evenly over the binary label where both occur."""
    groups = [g for _, g in df.groupby("label_binary")]
    per = max(1, n // max(1, len(groups)))
    return pd.concat([g.sample(min(per, len(g)), random_state=seed) for g in groups])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=400)
    ap.add_argument("--masked", default=None,
                    help="read the test sets from masked_<name>_cells.csv, "
                         "e.g. rembg_u2net_sample")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    poly = load_manifest("nihpoly_cells.csv")
    frames = [load_manifest("nih_cells.csv").assign(format="cell_images"),
              poly[poly.source_split == "polygon_masked"].assign(format="nih_format")]
    if args.masked:
        frames.append(load_manifest(f"masked_{args.masked}_cells.csv")
                      .assign(format="nih_format"))
    else:
        frames += [load_manifest("bbbc041_cells.csv").assign(format="raw"),
                   load_manifest("mpidb_cells.csv").assign(format="raw")]
    cells = plots.add_domain(pd.concat(frames, ignore_index=True))
    cells = cells[cells["eval_group"] == "primary"]

    picked = pd.concat([sample(g, args.per_domain, args.seed)
                        for _, g in cells.groupby("domain", observed=True)],
                       ignore_index=True)
    print(f"{len(picked):,} cells over {picked['domain'].nunique()} domains")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(_means, picked["path"].tolist(), chunksize=16))

    out = pd.concat([picked[KEEP].reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    dst = paths.TABLES / "colour_channel_stats.csv"
    out.to_csv(dst, index=False)
    print(out.groupby(["domain", "label_binary"], observed=True)[CHANNELS]
          .median().round(3).to_string())
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
