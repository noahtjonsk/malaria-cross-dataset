"""Figure helpers shared by the EDA notebook.

Kept out of the notebook so that every figure uses the same palette, the same
domain ordering and the same save path, and so the notebook reads as analysis
rather than as matplotlib boilerplate.
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths

# One colour per domain, used consistently across every figure.
DOMAIN_ORDER = ["nih", "bbbc041/site_a", "bbbc041/site_b",
                "mpidb/Falciparum", "mpidb/Malariae",
                "mpidb/Ovale", "mpidb/Vivax"]
DOMAIN_COLORS = {
    "nih": "#1f77b4",
    "bbbc041/site_a": "#ff7f0e", "bbbc041/site_b": "#d62728",
    "mpidb/Falciparum": "#2ca02c", "mpidb/Malariae": "#8c564b",
    "mpidb/Ovale": "#9467bd", "mpidb/Vivax": "#17becf",
}

STAGE_ORDER = ["ring", "trophozoite", "schizont", "gametocyte", "unknown"]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "legend.frameon": False,
    })


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    """A single `domain` column: the finest grouping the EDA compares across.

    BBBC041 splits into its two acquisition batches and MP-IDB into its four
    species, because both differ internally by as much as the datasets differ
    from each other -- which is the point of most of these figures.
    """
    df = df.copy()
    domain = np.where(df["dataset"] == "nih", "nih",
                      df["dataset"].astype(str) + "/"
                      + df["source_split"].astype(str))
    present = [d for d in DOMAIN_ORDER if d in set(domain)]
    df["domain"] = pd.Categorical(domain, categories=present, ordered=True)
    if bool(df["domain"].isna().any()):
        missing = sorted(set(domain) - set(DOMAIN_ORDER))
        raise ValueError(f"domains not listed in DOMAIN_ORDER: {missing}; "
                         "add them there so no cell silently drops out")
    return df


def domains_in(df: pd.DataFrame):
    return [d for d in DOMAIN_ORDER if d in set(df["domain"].astype(str))]


def save(fig, name: str) -> None:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / f"{name}.png")
    print(f"  saved outputs/figures/{name}.png")


def feature_panels(stats: pd.DataFrame, features, titles=None, xlabels=None,
                   log=(), ncols: int = 3, bins: int = 60, density: bool = True):
    """A grid of per-domain overlaid histograms, one panel per feature."""
    features = list(features)
    titles = titles or {}
    xlabels = xlabels or {}
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    doms = domains_in(stats)
    for ax, feat in zip(axes, features):
        vals = stats[feat].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if feat in log:
            finite = finite[finite > 0]
            edges = np.logspace(np.log10(finite.min()),
                                np.log10(finite.max()), bins)
            ax.set_xscale("log")
        else:
            lo, hi = np.percentile(finite, [0.2, 99.8])
            edges = np.linspace(lo, hi, bins)
        for d in doms:
            v = stats.loc[stats["domain"].astype(str) == d, feat]
            v = v.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if feat in log:
                v = v[v > 0]
            if len(v) == 0:
                continue
            ax.hist(v, bins=edges, density=density, histtype="step",
                    lw=1.6, label=d, color=DOMAIN_COLORS.get(d))
        ax.set_title(titles.get(feat, feat))
        ax.set_xlabel(xlabels.get(feat, feat))
        ax.set_ylabel("density" if density else "cells")
    for ax in axes[len(features):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig, axes


def load_rgb(path, size: int | None = None) -> np.ndarray:
    bgr = cv2.imread(str(paths.ROOT / path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if size:
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def contact_sheet(rows, n: int = 8, size: int | None = None, seed: int = 0,
                  title: str = "", sample: bool = True):
    """Cells per group, rendered at the network input size.

    `rows` is a list of (label, dataframe). Images are shown at `size` so the
    comparison is between what the network sees, not between raw file sizes.
    With `sample=True` (the default) `n` random rows are drawn from each group;
    with `sample=False` the first `n` rows are shown in the order given, for
    sheets whose order carries meaning (e.g. the smallest crops).
    """
    size = size or paths.MODEL_INPUT
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(rows), n,
                             figsize=(1.25 * n, 1.35 * len(rows)))
    # reshape, not atleast_2d: with n == 1 and several rows, atleast_2d gives
    # shape (1, m) and axes[1, 0] raises IndexError
    axes = np.asarray(axes).reshape(len(rows), n)
    for ri, (label, sub) in enumerate(rows):
        take = (sub.sample(min(n, len(sub)), random_state=int(rng.integers(1_000_000)))
                if sample else sub.iloc[:n])
        for ci in range(n):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if ci < len(take):
                ax.imshow(load_rgb(take.iloc[ci]["path"], size))
            else:
                ax.set_visible(False)
            if ci == 0:
                ax.set_ylabel(f"{label}\n(n={len(sub):,})", rotation=0,
                              ha="right", va="center", fontsize=8, labelpad=8)
    if title:
        fig.suptitle(title, y=1.005)
    fig.tight_layout()
    return fig
