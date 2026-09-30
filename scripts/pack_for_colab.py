"""Zip exactly the files the Colab training notebook needs.

    python scripts/pack_for_colab.py --variants raw          # RQ1: colab/data_raw.zip + code.zip
    python scripts/pack_for_colab.py --variants masked reinhard histmatch   # RQ2 sets, uploaded later
    python scripts/pack_for_colab.py --dry-run               # list counts and size only

Colab cannot read this laptop's disk, so the cells go up to Google Drive as
one archive per group of variants. Together they hold only what train.py and evaluate.py read: the NIH
cell_images, every primary test cell in each RQ2 variant that has been built,
and the manifests that point at them. Paths inside the zip are the repository
paths, so unzipping it in the repository root on Colab reproduces the layout
malaria/paths.py expects.

PNGs are already compressed, so the archive stores them without recompression
(faster, and almost no larger).

code.zip is the package, scripts, notebooks and tracked manifests, for when
the notebook is not cloning the code from GitHub.
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malaria import paths  # noqa: E402
from malaria.data import (TEST_DATASETS, VARIANT_MANIFESTS, nih_cells,  # noqa: E402
                          test_cells)

OUT = paths.COLAB
CODE = ["malaria/*.py", "scripts/*.py", "notebooks/10_train_colab.ipynb",
        "requirements.txt", "data/manifests/nih_split.csv",
        "data/manifests/mpidb_stage_audit.csv"]


def data_files(variants) -> list:
    files = list(nih_cells()["path"]) if "raw" in variants else []
    manifests = ["nih_cells.csv"]
    for variant, pattern in VARIANT_MANIFESTS.items():
        if variant not in variants:
            continue
        if variant != "raw" and not (paths.MANIFESTS / pattern).exists():
            print(f"  skipping {variant}: {pattern} not built")
            continue
        for d in TEST_DATASETS:
            files += list(test_cells(d, variant)["path"])
            manifests.append(pattern.format(dataset=d))
    files += [f"data/manifests/{m}" for m in sorted(set(manifests))]
    missing = [f for f in files if not (paths.ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[:3]}")
    return sorted(set(files))


def write_zip(dst: Path, files: list, compress: bool) -> None:
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(dst, "w", mode) as z:
        for f in files:
            z.write(paths.ROOT / f, f)
    print(f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB, {len(files):,} files)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_MANIFESTS),
                    choices=list(VARIANT_MANIFESTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = data_files(args.variants)
    size = sum((paths.ROOT / f).stat().st_size for f in data)
    print(f"data: {len(data):,} files, {size / 1e9:.2f} GB")
    code = sorted({p.relative_to(paths.ROOT).as_posix()
                   for pat in CODE for p in paths.ROOT.glob(pat)})
    print(f"code: {len(code)} files")
    if args.dry_run:
        return
    OUT.mkdir(exist_ok=True)
    write_zip(OUT / f"data_{'_'.join(args.variants)}.zip", data, compress=False)
    write_zip(OUT / "code.zip", code, compress=True)


if __name__ == "__main__":
    main()
