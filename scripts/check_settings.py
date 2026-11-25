"""Checks behind three fixed settings that the Technical Report justifies (Table 4).

    python scripts/check_settings.py                      # all three checks
    python scripts/check_settings.py --parts patience     # one or more of: patience bootstrap reference

1. patience: early-stopping patience 3 (train.SETTINGS). From the nine training logs, would
   a patience of 1 or 2 have stopped a run before its best validation epoch?
2. bootstrap: N_BOOT = 2,000 (metrics.N_BOOT). The seed-0 analysis is rerun with K independent
   random streams (metrics.BOOT_SEED = 0..K-1): how far do the 95% drop limits, the RQ3 species-gap
   limits and the Bonferroni-adjusted RQ1 comparisons move, and does any reported decision change?
   Needs outputs/predictions/<model>_s0/ (seeds 1-2 are on Drive only).
3. reference: REFERENCE_CELLS = 1,000 (build_colour_crops.py). The NIH colour reference from five
   independent 1,000-cell samples against the reference from all NIH training cells, next to the
   NIH-BBBC041 colour gap the reference is used to close. Reads every NIH training crop (minutes).

Writes outputs/tables/settings_checks.md with the parts that were run.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from malaria import metrics, paths  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
OUT = TABLES / "settings_checks.md"
ARCHS = ("vgg16", "resnet50", "mobilenet_v2")


def check_patience() -> list:
    """Per run: epochs run, best epoch, and where patience 1 or 2 would have stopped."""
    lines = ["## Early-stopping patience (train.SETTINGS patience = 3)", "",
             "| Run | Epochs run | Best epoch | Longest wait before a new best | Patience 2 keeps epoch | Stops before best |",
             "|---|---|---|---|---|---|"]
    early = 0
    for a in ARCHS:
        for s in (0, 1, 2):
            log = json.loads((TABLES / f"{a}_s{s}_training_log.json").read_text())
            vl = [h["val"]["loss"] for h in log["history"]]
            best, wait, longest, best_ep = float("inf"), 0, 0, 0
            for i, v in enumerate(vl, 1):
                if v < best:
                    longest, best, wait, best_ep = max(longest, wait), v, 0, i
                else:
                    wait += 1
            b, wait2, keep2 = float("inf"), 0, 0
            for i, v in enumerate(vl, 1):
                if v < b:
                    b, wait2, keep2 = v, 0, i
                else:
                    wait2 += 1
                if wait2 >= 2:
                    break
            early += keep2 < best_ep
            lines.append(f"| {a}_s{s} | {len(vl)} | {best_ep} | {longest} | {keep2} | {'yes' if keep2 < best_ep else 'no'} |")
    lines += ["", f"With patience 2, {early} of 9 runs would have stopped before their best epoch.", ""]
    return lines


def check_bootstrap(k: int) -> list:
    """Rerun the seed-0 bootstrap with k independent streams; report movement and decision changes."""
    from malaria.results import load
    tags = [f"{a}_s0" for a in ARCHS]
    cells = [("bbbc041/site_a", "sensitivity"), ("bbbc041/site_b", "sensitivity"),
             ("mpidb/falciparum", "sensitivity"), ("mpidb/other", "sensitivity"),
             ("bbbc041/site_a", "specificity"), ("bbbc041/site_b", "specificity"),
             ("bbbc041/site_a", "auc"), ("bbbc041/site_b", "auc")]
    alpha_adj = metrics.ALPHA / (len(cells) * 3)          # as compare_models.py
    sets = {t: load(t, "raw") for t in tags}
    drops, gaps, comps = {}, {}, {}
    seed0 = metrics.BOOT_SEED
    try:
        for stream in range(k):
            metrics.BOOT_SEED = stream
            draws = {}
            for t in tags:
                ref = metrics.reference_draws(sets[t]["nih_test"])
                boot = {n: metrics.bootstrap(sets[t][n], n) for n in sets[t] if n != "nih_test"}
                for n, m in cells:
                    draws[(t, n, m)] = metrics.drop_draws(ref, boot[n], m)[1]
                    drops.setdefault((t, n, m), []).append(metrics.interval(draws[(t, n, m)]))
                gap = 100 * (boot["mpidb/falciparum"]["sensitivity"][1] - boot["mpidb/other"]["sensitivity"][1])
                gaps.setdefault(t, []).append(metrics.interval(gap))
            for a, b in combinations(tags, 2):
                for n, m in cells:
                    lo, hi = metrics.interval(draws[(a, n, m)] - draws[(b, n, m)], alpha_adj)
                    comps.setdefault((a, b, n, m), []).append((lo, hi))
            print(f"bootstrap stream {stream} done", flush=True)
    finally:
        metrics.BOOT_SEED = seed0

    def move(ivs):
        x = np.array(ivs, dtype=float)
        return float(np.nanmax(np.nanmax(x, 0) - np.nanmin(x, 0)))

    def calls(ivs, rule):
        return {rule(lo, hi) for lo, hi in ivs}

    excl = lambda lo, hi: lo > 0 or hi < 0          # noqa: E731
    above = lambda lo, _hi: lo > 0                  # noqa: E731
    d_move = [move(v) for v in drops.values()]
    width = np.median([hi - lo for v in drops.values() for lo, hi in v])
    sig = [sum(excl(*v[i]) for v in comps.values()) for i in range(k)]
    return ["## Bootstrap resamples (metrics.N_BOOT = 2,000)", "",
            f"Seed-0 runs, {k} independent random streams (metrics.BOOT_SEED = 0 to {k - 1}).", "",
            "| Interval | Count | Largest movement of a limit (pp) | Decision changed between streams |",
            "|---|---|---|---|",
            f"| 95% drop interval (median width {width:.1f} pp) | {len(drops)} | {max(d_move):.2f} | "
            f"{sum(len(calls(v, above)) > 1 for v in drops.values())} (above zero: share-recovered rule) |",
            f"| 95% RQ3 species gap | {len(gaps)} | {max(move(v) for v in gaps.values()):.2f} | "
            f"{sum(len(calls(v, excl)) > 1 for v in gaps.values())} (excludes zero) |",
            f"| {100 * (1 - alpha_adj):.2f}% RQ1 paired comparison | {len(comps)} | "
            f"{max(move(v) for v in comps.values()):.2f} | {sum(len(calls(v, excl)) > 1 for v in comps.values())} "
            f"(significant in each stream: {', '.join(map(str, sig))}) |", ""]


def check_reference(seeds=range(5)) -> list:
    """1,000-cell NIH colour references against the reference from all NIH training cells."""
    import cv2
    from skimage import color
    from malaria.colour import tissue_mask
    from malaria.data import nih_cells

    q = np.linspace(0, 100, 256)

    def stats(rel):
        bgr = cv2.imread(str(paths.ROOT / rel), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rel)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = tissue_mask(rgb)
        lab = color.rgb2lab(rgb.astype(np.float64) / 255.0)[m]
        return lab.mean(0), lab.std(0), np.stack([np.bincount(rgb[..., c][m], minlength=256) for c in range(3)])

    def quantiles(hist):   # exact integer-histogram quantiles (pixels are 0-255)
        return np.array([[np.searchsorted(np.cumsum(h) / h.sum(), x / 100) for x in q] for h in hist], float)

    train = nih_cells("train").reset_index(drop=True)
    print(f"reading {len(train)} NIH training crops", flush=True)
    st = [stats(p) for p in train["path"]]
    means, stds, hists = (np.array([s[i] for s in st]) for i in range(3))
    full = (means.mean(0), stds.mean(0), quantiles(hists.sum(0)))
    d_mean, d_std, d_q = [], [], []
    for seed in seeds:
        idx = train.sample(1000, random_state=seed).index.to_numpy()   # as build_colour_crops.nih_reference
        d_mean.append(np.abs(means[idx].mean(0) - full[0]))
        d_std.append(np.abs(stds[idx].mean(0) - full[1]))
        d_q.append(np.abs(quantiles(hists[idx].sum(0))[:, 3:253] - full[2][:, 3:253]).max())
    cs = pd.read_csv(TABLES / "colour_channel_stats.csv").groupby("domain")[["L", "a", "b"]].median()
    gap = (cs.drop(index="nih") - cs.loc["nih"]).abs()
    lines = ["## NIH colour reference (build_colour_crops.REFERENCE_CELLS = 1,000)", "",
             f"Five independent 1,000-cell samples of the {len(train):,} NIH training cells against all of them.", "",
             "| Quantity | Largest difference over the five samples |", "|---|---|",
             "| Lab mean (L, a, b) | " + ", ".join(f"{v:.2f}" for v in np.max(d_mean, 0)) + " |",
             "| Lab within-cell std (L, a, b) | " + ", ".join(f"{v:.2f}" for v in np.max(d_std, 0)) + " |",
             f"| RGB quantiles 1-99% (grey levels, 0-255) | {max(d_q):.0f} |", "",
             "For scale, |median Lab of each domain minus NIH| (`colour_channel_stats.csv`):", "",
             "| Domain | L | a | b |", "|---|---|---|---|"]
    lines += [f"| {d} | {r.L:.2f} | {r.a:.2f} | {r.b:.2f} |" for d, r in gap.iterrows()]
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["patience", "bootstrap", "reference"],
                    choices=["patience", "bootstrap", "reference"])
    ap.add_argument("--streams", type=int, default=5)
    args = ap.parse_args()
    sections = {"patience": check_patience, "bootstrap": lambda: check_bootstrap(args.streams),
                "reference": check_reference}
    blocks = ["\n".join(sections[p]()) for p in ("patience", "bootstrap", "reference") if p in args.parts]
    text = "# Checks behind fixed settings (`scripts/check_settings.py`)\n\n" + "\n".join(blocks)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
