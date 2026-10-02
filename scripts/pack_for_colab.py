"""Zip exactly the files the Colab training notebook needs.

    python scripts/pack_for_colab.py                         # code.zip, data_raw.zip, data_rq2.zip
    python scripts/pack_for_colab.py --variants raw          # RQ1 only: code.zip + data_raw.zip
    python scripts/pack_for_colab.py --code-only             # code.zip after a code change
    python scripts/pack_for_colab.py --dry-run               # list counts and size only

Colab cannot read this laptop's disk, so the cells go up to Google Drive as
one archive per group of variants. Together they hold only what train.py and evaluate.py read: the NIH
cell_images, every primary test cell in each RQ2 variant that has been built,
and the manifests that point at them. Paths inside the zip are the repository
paths, so unzipping it in the repository root on Colab reproduces the layout
malaria/paths.py expects.

PNGs are already compressed, so the archive stores them without recompression
(faster, and almost no larger).

The archive names are fixed (data_raw.zip for the raw crops, data_rq2.zip
for the RQ2 variants), so the notebook never has to guess them. Every zip holds
PROVENANCE.txt with the git commit and the hash of each manifest it contains;
evaluate.py records the same hashes, and compare_models.py checks them.

code.zip is the package, scripts, notebooks and tracked manifests, for when
the notebook is not cloning the code from GitHub.
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malaria import paths, provenance  # noqa: E402
from malaria.data import (TEST_DATASETS, VARIANTS, nih_cells, test_cells,  # noqa: E402
                          variant_manifest_files)

OUT = paths.COLAB
CODE = ["malaria/*.py", "scripts/*.py", "notebooks/10_train_colab.ipynb",
        "requirements.txt", "data/manifests/nih_split.csv",
        "data/manifests/mpidb_stage_audit.csv",
        "outputs/tables/masked_otsu-gray_status.csv"]   # RQ2 robustness check


ZIP_NAMES = {"raw": "data_raw.zip", "rq2": "data_rq2.zip"}


def data_files(variants) -> tuple:
    """(crop and manifest paths, manifest file names) for a group of variants."""
    files = list(nih_cells()["path"]) if "raw" in variants else []
    manifests = {"nih_cells.csv"}
    for variant in variants:
        names = variant_manifest_files(variant)
        missing = [n for n in names if not (paths.MANIFESTS / n).exists()]
        if missing:
            raise SystemExit(f"{variant} is not built yet: {missing}")
        for d in TEST_DATASETS:
            files += list(test_cells(d, variant)["path"])
        manifests |= set(names)
    files += [f"data/manifests/{m}" for m in sorted(manifests)]
    missing = [f for f in files if not (paths.ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[:3]}")
    return sorted(set(files)), sorted(manifests)


def provenance_text(manifests: list) -> str:
    lines = [f"commit {provenance.git_commit()}"]
    lines += [f"manifest {n} {h}" for n, h in provenance.manifest_hashes(manifests).items()]
    return "".join(line + "\n" for line in lines)


def write_zip(dst: Path, files: list, compress: bool, manifests: list) -> None:
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(dst, "w", mode) as z:
        for f in files:
            z.write(paths.ROOT / f, f)
        z.writestr(provenance.PROVENANCE_FILE.name, provenance_text(manifests))
    print(f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB, {len(files):,} files)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--code-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = {} if args.code_only else {
        g: [v for v in args.variants if (v == "raw") == (g == "raw")] for g in ZIP_NAMES}
    groups = {g: vs for g, vs in groups.items() if vs}
    packs = {g: data_files(vs) for g, vs in groups.items()}
    for g, (files, _) in packs.items():
        size = sum((paths.ROOT / f).stat().st_size for f in files)
        print(f"{ZIP_NAMES[g]}: {groups[g]}, {len(files):,} files, {size / 1e9:.2f} GB")
    code = sorted({p.relative_to(paths.ROOT).as_posix()
                   for pat in CODE for p in paths.ROOT.glob(pat)})
    print(f"code: {len(code)} files")
    if args.dry_run:
        return
    OUT.mkdir(exist_ok=True)
    for g, (files, manifests) in packs.items():
        write_zip(OUT / ZIP_NAMES[g], files, compress=False, manifests=manifests)
    write_zip(OUT / "code.zip", code, compress=True,
              manifests=["nih_split.csv", "mpidb_stage_audit.csv"])


if __name__ == "__main__":
    main()
