"""Figures for the D3 Methodology Overview (docs/methodology/figures/).

    python scripts/make_mo_figures.py

Copies the three EDA figures the MO cites (from notebooks/03_eda_short.ipynb)
and draws three: the same test cells in each RQ2 variant (raw, background
removed, Reinhard, histogram matching), so a reader can see what each step
does to a cell before reading the results; training against NIH validation
loss per epoch for the three architectures (from the training logs); and the
pipeline grouped by research question, with planned stages dashed. Needs
build_masked_crops.py and build_colour_crops.py to have been run.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.data import load_input, test_cells  # noqa: E402
from malaria.models import ARCHS  # noqa: E402

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
    """The same test cells as raw crops and after each RQ2 step (Figure 5)."""
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


LABELS = {"vgg16": "VGG-16", "resnet50": "ResNet-50", "mobilenet_v2": "MobileNetV2"}
SERIES = ("#2a78d6", "#eb6834")             # categorical slots 1-2 (dataviz palette)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"


def training_curves() -> None:
    """Training vs NIH validation loss per epoch, one panel per architecture."""
    plots.set_style()
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(9, 2.8), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        log = json.loads((paths.TABLES / f"{paths.run_tag(arch, 0)}_training_log.json").read_text())
        h = log["history"]
        best = min(h, key=lambda r: r["val"]["loss"])
        assert abs(best["val"]["loss"] - log["best_val_loss"]) < 1e-12 and len(h) == log["epoch"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train"]["loss"] for r in h], marker="o", ms=4, lw=2, color=SERIES[0],
                label="training (augmented)")
        ax.plot(ep, [r["val"]["loss"] for r in h], marker="o", ms=4, lw=2, color=SERIES[1],
                label="NIH validation")
        ax.axvline(best["epoch"], color=MUTED, ls=":", lw=1)
        ax.set_title(f"{LABELS[arch]}: best epoch {best['epoch']} of {len(h)}")
        ax.set_xlabel("epoch")
        ax.set_xticks(ep)
    axes[0].set_ylabel("binary cross-entropy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")
    plt.close(fig)
    print(f"wrote {OUT / 'training_curves.png'}")


STAGES = [  # band title, [(letter, name, detail, planned)]
    ("Data preparation (A-D)", [
        ("A", "Raw data", "NIH: 27,558 cells,\n200 patients\nBBBC041: 1,328 smears\nMP-IDB: 210 smears", False),
        ("B", "Cell crops", "one manifest schema\nsquare, +10% padding\nMP-IDB on host cell\nbuild_crops.py", False),
        ("C", "Patient split", "NIH 70/15/15,\ngrouped by patient\nnih_split.csv\nsplits.py", False),
        ("D", "EDA", "tissue-only brightness,\ncolour, crop format\n03_eda_short.ipynb", False)]),
    ("RQ1: model comparison (E, G)", [
        ("E", "Fine-tune on NIH", "VGG-16, ResNet-50,\nMobileNetV2; all layers,\nearly stopping; train.py", False),
        ("G", "Frozen inference", "NIH hold-out, site_a,\nsite_b, MP-IDB; 0.5\nthreshold; evaluate.py", False),
        ("E", "Seeds 1-2", "same settings,\nthree runs per model", True)]),
    ("RQ2: test-input ablation (F, G)", [
        ("F1", "Background removed", "Otsu on brightness\nbuild_masked_crops.py", False),
        ("F2", "Colour matched", "Reinhard or\nhistogram matching\nbuild_colour_crops.py", False),
        ("G", "Frozen re-scoring", "the same three models,\nno retraining\nevaluate.py --variants", False)]),
    ("Evaluation and RQ3 error analysis (H)", [
        ("H", "Metrics", "drop, share recovered, F1\nimage-resampled 95% CIs\nsummarise_results.py", False),
        ("H3", "Species and stage", "sensitivity by species,\nmisses by life stage", False),
        ("H", "Across seeds", "mean and spread\nover three seeds", True)]),
]


def pipeline_figure() -> None:
    """The pipeline grouped by research question; solid = implemented, dashed = planned."""
    width, box_h, title_h, pad, gap = 10.0, 1.55, 0.45, 0.18, 0.55
    band_h = box_h + 2 * pad
    height = len(STAGES) * (band_h + title_h) + (len(STAGES) - 1) * gap + 0.45
    fig, ax = plt.subplots(figsize=(7.4, 7.4 * height / width))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(0.05, height - 0.2, "solid box = implemented    dashed grey box = planned",
            fontsize=7.5, color=INK_2, va="center")
    top = height - 0.45
    for i, (title, boxes) in enumerate(STAGES):
        ax.text(0.05, top - title_h / 2, title, fontsize=9.5, weight="bold", color=INK, va="center")
        y0 = top - title_h - band_h
        ax.add_patch(Rectangle((0.02, y0), width - 0.04, band_h, fill=False, ls=(0, (4, 3)),
                               lw=0.8, ec=MUTED))
        n, between = len(boxes), 0.45
        w = (width - 0.04 - 2 * pad - (n - 1) * between) / n
        prev = None
        for j, (letter, name, detail, planned) in enumerate(boxes):
            x, y = 0.02 + pad + j * (w + between), y0 + pad
            ax.add_patch(FancyBboxPatch((x, y), w, box_h, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="#f1f1ef" if planned else "white",
                                        ec=MUTED if planned else INK, lw=1.0,
                                        ls=(0, (3, 2)) if planned else "-"))
            ink = MUTED if planned else INK
            ax.text(x + 0.12, y + box_h - 0.2, name, fontsize=8, weight="bold", color=ink, va="center")
            ax.text(x + w - 0.1, y + box_h - 0.2, letter, fontsize=8.5, weight="bold",
                    color=MUTED if planned else SERIES[0], ha="right", va="center")
            ax.text(x + 0.12, y + box_h - 0.42, detail, fontsize=6.3, color=MUTED if planned else INK_2,
                    va="top", linespacing=1.35)
            ax.text(x + 0.12, y + 0.14, "planned" if planned else "implemented", fontsize=6.3,
                    style="italic", color=INK_2, va="center")
            if prev is not None and not planned:
                ax.add_patch(FancyArrowPatch((prev, y + box_h / 2), (x, y + box_h / 2),
                                             arrowstyle="-|>", mutation_scale=9, lw=1, color=INK_2))
            prev = x + w
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((width / 2, y0 - 0.02), (width / 2, y0 - gap + 0.02),
                                         arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK_2))
        top = y0 - gap
    fig.savefig(OUT / "pipeline_v3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'pipeline_v3.png'}")


def main() -> None:
    """Copy the EDA figures and draw the Methodology Overview's own figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy(paths.FIGURES / name, OUT / name)
    print(f"copied {len(COPY)} EDA figures")
    variants_figure()
    training_curves()
    pipeline_figure()


if __name__ == "__main__":
    main()
