"""Figures for the D3 Methodology Overview (docs/methodology/figures/).

    python scripts/make_mo_figures.py

Copies the three EDA figures the MO cites (from notebooks/03_eda_short.ipynb)
and draws one new one: the same test cells in each RQ2 variant (raw,
background removed, Reinhard, histogram matching), so a reader can see what
each step does to a cell before reading the results. Needs
build_masked_crops.py and build_colour_crops.py to have been run.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.data import load_input, test_cells  # noqa: E402

OUT = paths.MO_FIGURES
COPY = ["short_1_example_cells.png", "short_3_crop_format.png",
        "short_4_brightness_colour.png"]
VARIANTS = [("raw", "raw crop (RQ1)"), ("masked", "background removed"),
            ("reinhard", "+ Reinhard"), ("histmatch", "+ histogram matching")]
ROWS = [("bbbc041", "site_a", 1, "BBBC041 site_a, parasitised"),
        ("bbbc041", "site_b", 1, "BBBC041 site_b, parasitised"),
        ("bbbc041", "site_a", 0, "BBBC041 site_a, uninfected"),
        ("mpidb_wholecell", "Falciparum", 1, "MP-IDB P. falciparum"),
        ("mpidb_wholecell", "Vivax", 1, "MP-IDB P. vivax")]


def variants_figure(seed: int = 3) -> None:
    sets = {(d, v): test_cells(d, v).set_index("cell_id")
            for d in ("bbbc041", "mpidb_wholecell") for v, _ in VARIANTS}
    plots.set_style()
    fig, axes = plt.subplots(len(ROWS), len(VARIANTS), figsize=(6.4, 1.45 * len(ROWS)))
    for r, (d, split, label, title) in enumerate(ROWS):
        raw = sets[(d, "raw")]
        pick = raw[(raw["source_split"] == split) & (raw["label_binary"] == label)]
        cid = pick.sample(1, random_state=seed).index[0]
        for c, (v, name) in enumerate(VARIANTS):
            ax = axes[r, c]
            ax.imshow(load_input(sets[(d, v)].loc[cid, "path"], 160))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_variants.png")
    plt.close(fig)
    print(f"wrote {OUT / 'rq2_variants.png'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy(paths.FIGURES / name, OUT / name)
    print(f"copied {len(COPY)} EDA figures")
    variants_figure()


if __name__ == "__main__":
    main()
