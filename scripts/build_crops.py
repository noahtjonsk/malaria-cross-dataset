"""Build the per-cell manifests and cache the cropped cells for both test sets.

Run once (it takes a while -- roughly 85k crops are written); afterwards it is
cheap, because crops that already exist on disk are not re-encoded.

    python scripts/build_crops.py            # build, skipping existing crops
    python scripts/build_crops.py --rebuild  # re-encode every crop
    python scripts/build_crops.py --dry-run  # manifests only, write nothing
    python scripts/build_crops.py --nihpoly  # NIH cells recut from the Polygon
                                             # Set photographs (add --nihpoly-point
                                             # for the Point Set as well)
    python scripts/build_crops.py --mpidb-wholecell  # MP-IDB framed on the host
                                                     # red cell, not the parasite
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                  # noqa: E402

from malaria import paths            # noqa: E402
from malaria.manifests import (build_all, build_mpidb_wholecell_manifest,  # noqa: E402
                               build_nihpoly_manifest, load_manifest)
from malaria.splits import check_no_patient_leakage, inherit_nih_split  # noqa: E402


def build_nihpoly(args) -> None:
    sets = ("polygon", "point") if args.nihpoly_point else ("polygon",)
    paths.ensure_dirs()
    cells = build_nihpoly_manifest(sets, write_crops=not args.dry_run,
                                   skip_existing=not args.rebuild)
    cells["split"] = inherit_nih_split(cells)
    check_no_patient_leakage(cells)
    cells.drop(columns="split").to_csv(paths.MANIFESTS / "nihpoly_cells.csv",
                                       index=False)
    print(f"\nnihpoly manifest rows: {len(cells):,}")
    print(cells.groupby(["source_split", "eval_group", "label_binary"],
                        dropna=False).size().to_string())
    print("\ncells per inherited split (polygon_raw):")
    raw = cells[cells.source_split == cells.source_split.iloc[0]]
    print(raw.groupby(["split", "label_binary"], dropna=False).size().to_string())
    print(f"\nwritten to {paths.MANIFESTS / 'nihpoly_cells.csv'}")


def build_mpidb_wholecell(args) -> None:
    paths.ensure_dirs()
    cells, framing = build_mpidb_wholecell_manifest(write_crops=not args.dry_run,
                                                    skip_existing=not args.rebuild)
    old = load_manifest("mpidb_cells.csv")
    both = old[["cell_id", "stage"]].merge(cells[["cell_id", "stage"]], on="cell_id",
                                           how="outer", suffixes=("_old", "_new"),
                                           indicator=True)
    if not (both["_merge"] == "both").all():
        raise AssertionError("whole-cell and parasite framings cover different cells")
    if not (both["stage_old"].astype(str) == both["stage_new"].astype(str)).all():
        raise AssertionError("stage labels changed between framings")

    cells.to_csv(paths.MANIFESTS / "mpidb_wholecell_cells.csv", index=False)
    framing.to_csv(paths.TABLES / "mpidb_wholecell_framing.csv", index=False)

    old_side = (old["c1"].astype(int) - old["c0"].astype(int)).groupby(old["species"]).median()
    summary = framing.groupby("species").agg(
        cells=("cell_id", "size"),
        host_cell_found=("framing", lambda s: float((s == "host_cell").mean())),
        parasite_side=("parasite_side", "median"),
        crop_side_now=("crop_side", "median"))
    summary["crop_side_before"] = old_side
    print()
    print(f"mpidb_wholecell manifest rows: {len(cells):,} "
          "(same cells and stage labels as mpidb_cells.csv)")
    print(summary.round(3).to_string())
    print()
    print("framing outcomes:")
    print(pd.crosstab(framing["species"], framing["framing"]).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-encode crops that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="build manifests without writing any crop files")
    ap.add_argument("--nihpoly", action="store_true",
                    help="only recut the NIH cells from NIH-NLM-ThinBloodSmearsPf")
    ap.add_argument("--nihpoly-point", action="store_true",
                    help="with --nihpoly, include the Point Set as well")
    ap.add_argument("--mpidb-wholecell", action="store_true",
                    help="only build MP-IDB crops framed on the host red cell")
    args = ap.parse_args()

    if args.mpidb_wholecell:
        build_mpidb_wholecell(args)
        return

    if args.nihpoly:
        build_nihpoly(args)
        return

    cells, audit = build_all(write_crops=not args.dry_run,
                             skip_existing=not args.rebuild)

    print(f"\nmanifest rows: {len(cells):,}")
    print(cells.groupby(["dataset", "source_split"]).size().to_string())
    print("\nMP-IDB stage label provenance:")
    print(audit["stage_source"].value_counts().to_string())
    print(f"\nwritten to {paths.MANIFESTS}")


if __name__ == "__main__":
    main()
