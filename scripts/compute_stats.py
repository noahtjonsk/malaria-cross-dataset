"""Compute per-image statistics for every cell and cache them to CSV.

Roughly 115k images, so this runs once and everything downstream reads the CSV.

    python scripts/compute_stats.py
    python scripts/compute_stats.py --workers 8
    python scripts/compute_stats.py --manifest nihpoly_cells.csv   # -> cell_stats_nihpoly.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.imagestats import FEATURES, image_stats  # noqa: E402
from malaria.manifests import load_manifest  # noqa: E402


def _stats_for_path(rel: str) -> dict:
    return image_stats(paths.ROOT / rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunksize", type=int, default=256)
    ap.add_argument("--manifest", default="all_cells.csv",
                    help="manifest in data/manifests to compute statistics for")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"computing {len(FEATURES)} features for {len(manifest):,} cells "
          f"on {args.workers} workers ...")

    paths_list = manifest["path"].tolist()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(_stats_for_path, paths_list,
                           chunksize=args.chunksize))

    stats = pd.concat([manifest.reset_index(drop=True),
                       pd.DataFrame(recs)], axis=1)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    suffix = Path(args.manifest).stem.replace("_cells", "")
    out = paths.TABLES / ("cell_stats.csv" if args.manifest == "all_cells.csv"
                          else f"cell_stats_{suffix}.csv")
    stats.to_csv(out, index=False)

    print(f"wrote {out}  ({len(stats):,} rows x {stats.shape[1]} cols)")
    print(stats.groupby(["dataset", "source_split"], observed=True)
          [["w", "gray_mean", "lapvar224", "black_frac"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
