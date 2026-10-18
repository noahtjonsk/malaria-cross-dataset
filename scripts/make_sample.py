"""Build the committed sample in tests/fixtures/sample/ that `make all` runs on.

    python scripts/make_sample.py          # needs the full manifests and crops

The source datasets are several GB and are not in the repository, so the
Midterm Review checks run the baseline pipeline on a small, fixed sample of
the same cells instead (paths.SAMPLE, MALARIA_SAMPLE=1):

    NIH          data.SAMPLE_PER_CLASS cells per split and class, drawn from the
                 pinned patient-grouped split (nih_split.csv), so no patient
                 crosses splits in the sample either
    BBBC041      PER_CLASS parasitised and PER_CLASS uninfected primary cells per site
    MP-IDB       PER_CLASS P. falciparum and PER_CLASS other-species whole-cell crops

Parasitised cells are over-represented in the external sets (BBBC041 is only
2.7-5.1% parasitised), so that every reported set has both classes where it
can; the sample checks that the code runs, not the size of any effect.

Writes, under tests/fixtures/sample/
    crops/<dataset>/<cell_id>.png   copies of the crops, unchanged
    manifests/                      nih_cells.csv, nih_split.csv, bbbc041_cells.csv,
                                    mpidb_wholecell_cells.csv (same schema, paths
                                    pointing at the copies)
    mpidb_wholecell_framing.csv     the framing rows of the sampled MP-IDB cells,
                                    read by summarise_results.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.data import SAMPLE_PER_CLASS, nih_cells, test_cells  # noqa: E402
from malaria.manifests import COLUMNS  # noqa: E402

if paths.SAMPLE:
    raise SystemExit("run without MALARIA_SAMPLE: the sample is drawn from the full data")

PER_CLASS = 30
SEED = 0
OUT_MANIFESTS = paths.SAMPLE_DIR / "manifests"
OUT_CROPS = paths.SAMPLE_DIR / "crops"
OUT_FRAMING = paths.SAMPLE_DIR / "mpidb_wholecell_framing.csv"


def draw(cells: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """n rows per value of `by` (fewer where a group is smaller), seeded."""
    return (cells.sample(frac=1, random_state=SEED).groupby(by).head(n)
            .sort_values("cell_id").reset_index(drop=True))


def copy_crops(cells: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Copy each crop into the sample's crops/<dataset>/ and repoint `path` at it."""
    out = OUT_CROPS / dataset
    out.mkdir(parents=True, exist_ok=True)
    new = []
    for cell_id, path in zip(cells["cell_id"], cells["path"]):
        dst = out / f"{cell_id}.png"
        shutil.copyfile(paths.ROOT / path, dst)
        new.append(dst.relative_to(paths.ROOT).as_posix())
    return cells.assign(path=new)


def main() -> None:
    readme = paths.SAMPLE_DIR / "README.md"          # hand-written, kept across rebuilds
    readme = readme.read_text(encoding="utf-8") if readme.exists() else None
    if paths.SAMPLE_DIR.exists():
        shutil.rmtree(paths.SAMPLE_DIR)
    OUT_MANIFESTS.mkdir(parents=True)

    nih = pd.concat([draw(nih_cells(s), "label_binary", n) for s, n in SAMPLE_PER_CLASS.items()])
    nih = copy_crops(nih, "nih")
    nih[COLUMNS].to_csv(OUT_MANIFESTS / "nih_cells.csv", index=False)
    nih[["cell_id", "patient_id", "label_binary", "split"]].to_csv(
        OUT_MANIFESTS / "nih_split.csv", index=False)

    bbbc = test_cells("bbbc041")
    bbbc = draw(bbbc.assign(group=bbbc["source_split"] + bbbc["label_binary"].astype(str)),
                "group", PER_CLASS)
    copy_crops(bbbc, "bbbc041")[COLUMNS].to_csv(OUT_MANIFESTS / "bbbc041_cells.csv", index=False)

    mpidb = test_cells("mpidb_wholecell")
    mpidb = draw(mpidb.assign(group=mpidb["species"] == "Falciparum"), "group", PER_CLASS)
    mpidb = copy_crops(mpidb, "mpidb_wholecell")
    mpidb[COLUMNS].to_csv(OUT_MANIFESTS / "mpidb_wholecell_cells.csv", index=False)

    framing = pd.read_csv(paths.TABLES / "mpidb_wholecell_framing.csv")
    framing[framing["cell_id"].isin(mpidb["cell_id"])].to_csv(OUT_FRAMING, index=False)
    if readme:
        (paths.SAMPLE_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"NIH {nih.groupby('split').size().to_dict()}, "
          f"BBBC041 {bbbc.groupby(['source_split', 'label_binary']).size().to_dict()}, "
          f"MP-IDB {mpidb.groupby('species').size().to_dict()}")
    size = sum(f.stat().st_size for f in OUT_CROPS.rglob("*.png")) / 1e6
    print(f"{len(list(OUT_CROPS.rglob('*.png')))} crops, {size:.1f} MB -> {paths.SAMPLE_DIR}")


if __name__ == "__main__":
    main()
