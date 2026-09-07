"""Build the per-cell manifests and cache the cropped cells for both test sets.

Run once (it takes a while -- roughly 85k crops are written); afterwards it is
cheap, because crops that already exist on disk are not re-encoded.

    python scripts/build_crops.py            # build, skipping existing crops
    python scripts/build_crops.py --rebuild  # re-encode every crop
    python scripts/build_crops.py --dry-run  # manifests only, write nothing
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malaria import paths            # noqa: E402
from malaria.manifests import build_all   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-encode crops that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="build manifests without writing any crop files")
    args = ap.parse_args()

    cells, audit = build_all(write_crops=not args.dry_run,
                             skip_existing=not args.rebuild)

    print(f"\nmanifest rows: {len(cells):,}")
    print(cells.groupby(["dataset", "source_split"]).size().to_string())
    print("\nMP-IDB stage label provenance:")
    print(audit["stage_source"].value_counts().to_string())
    print(f"\nwritten to {paths.MANIFESTS}")


if __name__ == "__main__":
    main()
