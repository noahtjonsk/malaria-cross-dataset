"""Turn one model's prediction files into the RQ1, RQ2 and RQ3 tables.

    python scripts/summarise_results.py --model vgg16_s0
    python scripts/summarise_results.py --model mobilenet_v2_s0_smoke        # after --smoke runs
    python scripts/summarise_results.py --model vgg16_s0 --variants raw      # RQ1 and RQ3 only

Reads outputs/predictions/<model>/<variant>_<set>.csv (written by evaluate.py)
and applies the definitions in malaria/metrics.py: fixed 0.5 threshold, drop in
percentage points from the NIH hold-out set, 95% intervals from resampling
source images (NIH hold-out: patients). A missing prediction file for any
requested variant stops the run, so no table is silently left incomplete.

Test sets reported (lit review, Table 2)
    nih_test           reference
    bbbc041/site_a     sensitivity, specificity, AUC
    bbbc041/site_b     sensitivity, specificity, AUC
    mpidb/falciparum   sensitivity only (P. falciparum, the NIH species)
    mpidb/other        sensitivity only (P. malariae, ovale and vivax pooled)

Writes outputs/tables/ (every table starts with model, arch and seed columns)
    rq1_<model>.csv    value, interval, drop and drop interval per set and metric
    rq2_<model>.csv    the same per RQ2 variant, plus the share of the drop recovered
    rq2_<model>_without_nearest.csv
                       RQ2 again without the BBBC041 cells whose mask fell back to
                       the nearest component (robustness check)
    rq3_<model>.csv    sensitivity by MP-IDB species group and by BBBC041 life stage,
                       with each stage's share of all missed parasites
and outputs/figures/R_<model>.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.data import VARIANTS  # noqa: E402
from malaria.metrics import (THRESHOLD, bootstrap, interval,  # noqa: E402
                             reference_draws, share_recovered, summarise)
from malaria.results import SET_COLOURS, load  # noqa: E402

MASK_STATUS = paths.TABLES / "masked_otsu-gray_status.csv"


def rq3_table(raw: dict) -> pd.DataFrame:
    """Sensitivity per MP-IDB species group and per BBBC041 life stage.

    The species groups are resampled exactly as in the RQ1 table (same set
    names), so their intervals agree, and independently of each other, so the
    interval on their difference is the one for two independent groups.
    """
    rows, sens = [], {}
    for name in ("mpidb/falciparum", "mpidb/other"):
        sens[name] = pt, draws = bootstrap(raw[name], name)["sensitivity"]
        lo, hi = interval(draws)
        rows.append({"group": name, "n_parasitised": len(raw[name]),
                     "sensitivity": pt, "ci_low": lo, "ci_high": hi})
    (fp, fd), (op, od) = sens["mpidb/falciparum"], sens["mpidb/other"]
    lo, hi = interval(100 * (fd - od))
    rows.append({"group": "falciparum minus other (pp)", "sensitivity": 100 * (fp - op),
                 "ci_low": lo, "ci_high": hi})

    pos = pd.concat([raw["bbbc041/site_a"], raw["bbbc041/site_b"]])
    pos = pos[pos["label_binary"] == 1].assign(missed=lambda d: d["prob"] < THRESHOLD)
    site_misses = pos.groupby("source_split")["missed"].sum()
    for (site, stage), g in pos.groupby(["source_split", "stage"]):
        name = f"bbbc041/{site}/{stage}"
        pt, draws = bootstrap(g, name)["sensitivity"]
        lo, hi = interval(draws)
        m = int(g["missed"].sum())
        rows.append({"group": name, "n_parasitised": len(g),
                     "sensitivity": pt, "ci_low": lo, "ci_high": hi,
                     "missed": m, "miss_rate": m / len(g),
                     "share_of_site_misses": m / max(int(site_misses[site]), 1)})
    return pd.DataFrame(rows)


def without_nearest(sets: dict) -> dict:
    """Drop BBBC041 cells whose background mask fell back to the nearest
    component (background.cell_mask), which may be a neighbouring cell."""
    st = pd.read_csv(MASK_STATUS, usecols=["dataset", "cell_id", "status"])
    nearest = set(st.loc[(st["dataset"] == "bbbc041") & (st["status"] == "nearest"), "cell_id"])
    return {k: s[~s["cell_id"].isin(nearest)] if k.startswith("bbbc041") else s
            for k, s in sets.items()}


def rq2_table(variant_sets: dict, nih_test: pd.DataFrame, ref: dict, keep=None) -> pd.DataFrame:
    """Every variant against the same NIH reference, plus the share of the drop
    recovered. `keep` filters the test sets identically in every variant."""
    frames = []
    for v, sets in variant_sets.items():
        sets = {k: s for k, s in sets.items() if k != "nih_test"}
        t = summarise({"nih_test": nih_test, **(keep(sets) if keep else sets)}, ref=ref)
        frames.append(t[t["test_set"] != "nih_test"].assign(variant=v))
    t = pd.concat(frames, ignore_index=True)
    holdout = {m: pt for m, (pt, _) in ref.items()}
    by_key = t.set_index(["variant", "test_set", "metric"])["value"]

    def value(variant, r):
        return by_key.get((variant, r.test_set, r.metric), np.nan)

    total, step = [], []
    for r in t.itertuples():
        h, raw = holdout[r.metric], value("raw", r)
        s_mask = share_recovered(h, raw, value("masked", r))
        s_total = share_recovered(h, raw, r.value)
        if r.variant == "raw":
            total.append(np.nan)
            step.append(np.nan)
        elif r.variant == "masked":
            total.append(s_total)
            step.append(s_mask)
        else:                     # a colour step, applied after background removal
            total.append(s_total)
            step.append(s_total - s_mask)
    t["share_recovered_total"], t["share_recovered_step"] = total, step
    return t[["variant", *[c for c in t.columns if c != "variant"]]]


VARIANT_MARKERS = {"raw": ("o", "raw crop (RQ1)"), "masked": ("s", "background removed"),
                   "reinhard": ("^", "+ Reinhard"), "histmatch": ("D", "+ histogram matching")}


def figure(model: str, rq1: pd.DataFrame, rq2: pd.DataFrame | None = None) -> None:
    """One row per test set, one panel per metric; RQ2 variants as extra markers.

    Every panel uses the same row order, so a row is the same test set in all
    three panels even where a metric is undefined (specificity on MP-IDB).
    """
    plots.set_style()
    t = rq1.assign(variant="raw") if rq2 is None else pd.concat(
        [rq1[rq1["test_set"] == "nih_test"].assign(variant="raw"), rq2])
    sets = list(SET_COLOURS)
    variants = [v for v in VARIANT_MARKERS if v in set(t["variant"])]
    offset = {v: (i - (len(variants) - 1) / 2) * 0.16 for i, v in enumerate(variants)}
    metrics = ["sensitivity", "specificity", "auc"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, metric in zip(axes, metrics):
        for r in t[t["metric"] == metric].itertuples():
            y = sets.index(r.test_set) + offset[r.variant]
            c = SET_COLOURS[r.test_set]
            ax.errorbar(100 * r.value, y, xerr=[[100 * (r.value - r.ci_low)],
                                                [100 * (r.ci_high - r.value)]],
                        fmt=VARIANT_MARKERS[r.variant][0], color=c, ecolor=c,
                        capsize=2, markersize=5, mfc=c if r.variant == "raw" else "white")
        ax.set_yticks(range(len(sets)), sets)
        ax.set_ylim(len(sets) - 0.5, -0.5)
        ax.set_xlim(0, 101)
        ax.set_xlabel("AUC (x100)" if metric == "auc" else f"{metric} (%)")
        ax.set_title(metric)
    if len(variants) > 1:
        handles = [plt.Line2D([], [], marker=VARIANT_MARKERS[v][0], ls="", color="0.3",
                              mfc="0.3" if v == "raw" else "white", label=VARIANT_MARKERS[v][1])
                   for v in variants]
        fig.legend(handles=handles, loc="lower center", ncol=len(variants), fontsize=8,
                   bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"{model}: fixed 0.5 threshold, 95% intervals from resampling source images "
                 "(NIH hold-out: patients)",
                 fontsize=9, y=1.02)
    plots.save(fig, f"R_{model}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run tag, e.g. resnet50_s0")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS,
                    help="all must have prediction files; raw is always included")
    args = ap.parse_args()
    variants = ["raw", *[v for v in VARIANTS if v in args.variants and v != "raw"]]

    arch, seed, _ = paths.parse_run_tag(args.model)
    ident = {"model": args.model, "arch": arch, "seed": seed}

    def save(t: pd.DataFrame, name: str) -> None:
        t = pd.concat([pd.DataFrame(ident, index=t.index), t], axis=1)
        t.to_csv(paths.TABLES / name, index=False)
        print(f"wrote {name}")

    variant_sets = {v: load(args.model, v) for v in variants}   # raises if any is missing
    raw = variant_sets["raw"]
    if "masked" in variants and not MASK_STATUS.exists():
        raise SystemExit(f"{MASK_STATUS} is needed for the robustness check "
                         "(it is packed into code.zip by pack_for_colab.py)")
    paths.TABLES.mkdir(parents=True, exist_ok=True)

    ref = reference_draws(raw["nih_test"])
    rq1 = summarise(raw, ref=ref)
    save(rq1, f"rq1_{args.model}.csv")
    print(rq1.round(3).to_string(index=False))

    rq2 = None
    if len(variants) > 1:
        rq2 = rq2_table(variant_sets, raw["nih_test"], ref)
        save(rq2, f"rq2_{args.model}.csv")
        if "masked" in variants:
            rob = rq2_table(variant_sets, raw["nih_test"], ref, keep=without_nearest)
            save(rob, f"rq2_{args.model}_without_nearest.csv")

    save(rq3_table(raw), f"rq3_{args.model}.csv")
    figure(args.model, rq1, rq2)


if __name__ == "__main__":
    main()
