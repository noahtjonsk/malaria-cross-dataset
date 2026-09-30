"""Turn one model's prediction files into the RQ1, RQ2 and RQ3 tables.

    python scripts/summarise_results.py --model vgg16_s0
    python scripts/summarise_results.py --model mobilenet_v2_s0_smoke        # after --smoke runs

Reads outputs/predictions/<model>/<variant>_<set>.csv (written by evaluate.py)
and applies the definitions in malaria/metrics.py: fixed 0.5 threshold, drop in
percentage points from the NIH hold-out set, 95% intervals from resampling
source images (NIH hold-out: patients).

Test sets reported (lit review, Table 2)
    nih_test           reference
    bbbc041/site_a     sensitivity, specificity, AUC
    bbbc041/site_b     sensitivity, specificity, AUC
    mpidb/falciparum   sensitivity only (P. falciparum, the NIH species)
    mpidb/other        sensitivity only (P. malariae, ovale and vivax pooled)

Writes outputs/tables/
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
from malaria.metrics import (THRESHOLD, bootstrap_rates, interval,  # noqa: E402
                             reference_draws, share_recovered, summarise)

PREDICTIONS = paths.PREDICTIONS
MASK_STATUS = paths.TABLES / "masked_otsu-gray_status.csv"
VARIANTS = ["raw", "masked", "reinhard", "histmatch"]
SET_COLOURS = {"nih_test": plots.DOMAIN_COLORS["nih"],
               "bbbc041/site_a": plots.DOMAIN_COLORS["bbbc041/site_a"],
               "bbbc041/site_b": plots.DOMAIN_COLORS["bbbc041/site_b"],
               "mpidb/falciparum": plots.DOMAIN_COLORS["mpidb/Falciparum"],
               "mpidb/other": plots.DOMAIN_COLORS["mpidb/Vivax"]}


def split_sets(bbbc: pd.DataFrame, mpidb: pd.DataFrame) -> dict:
    """The reported test sets, cut from one variant's prediction files."""
    return {"bbbc041/site_a": bbbc[bbbc["source_split"] == "site_a"],
            "bbbc041/site_b": bbbc[bbbc["source_split"] == "site_b"],
            "mpidb/falciparum": mpidb[mpidb["species"] == "Falciparum"],
            "mpidb/other": mpidb[mpidb["species"] != "Falciparum"]}


def load(model: str, variant: str) -> dict | None:
    d = PREDICTIONS / model
    files = {n: d / f"{variant}_{n}.csv" for n in ("bbbc041", "mpidb_wholecell")}
    if not all(f.exists() for f in files.values()):
        return None
    sets = split_sets(pd.read_csv(files["bbbc041"]), pd.read_csv(files["mpidb_wholecell"]))
    if variant == "raw":
        sets = {"nih_test": pd.read_csv(d / "raw_nih_test.csv"), **sets}
    return sets


def rq3_table(raw: dict) -> pd.DataFrame:
    """Sensitivity per MP-IDB species group and per BBBC041 life stage."""
    rows = []
    for name in ("mpidb/falciparum", "mpidb/other"):
        pt, draws = bootstrap_rates(raw[name])["sensitivity"]
        lo, hi = interval(draws)
        rows.append({"group": name, "n_parasitised": len(raw[name]),
                     "sensitivity": pt, "ci_low": lo, "ci_high": hi})
    # The species comparison: is the pooled-species gap larger than the interval?
    f = bootstrap_rates(raw["mpidb/falciparum"], seed=1)["sensitivity"]
    o = bootstrap_rates(raw["mpidb/other"], seed=2)["sensitivity"]
    lo, hi = interval(100 * (f[1] - o[1]))
    rows.append({"group": "falciparum minus other (pp)", "sensitivity": 100 * (f[0] - o[0]),
                 "ci_low": lo, "ci_high": hi})

    pos = pd.concat([raw["bbbc041/site_a"], raw["bbbc041/site_b"]])
    pos = pos[pos["label_binary"] == 1]
    missed = pos["prob"] < THRESHOLD
    for (site, stage), g in pos.groupby(["source_split", "stage"]):
        pt, draws = bootstrap_rates(g)["sensitivity"]
        lo, hi = interval(draws)
        m = int((g["prob"] < THRESHOLD).sum())
        rows.append({"group": f"bbbc041/{site}/{stage}", "n_parasitised": len(g),
                     "sensitivity": pt, "ci_low": lo, "ci_high": hi,
                     "missed": m, "miss_rate": m / len(g),
                     "share_of_site_misses": m / max(int(missed[pos["source_split"] == site].sum()), 1)})
    return pd.DataFrame(rows)


def without_nearest(sets: dict) -> dict:
    """Drop BBBC041 cells whose background mask fell back to the nearest
    component (background.cell_mask), which may be a neighbouring cell."""
    st = pd.read_csv(MASK_STATUS, usecols=["dataset", "cell_id", "status"])
    nearest = set(st.loc[(st["dataset"] == "bbbc041") & (st["status"] == "nearest"), "cell_id"])
    return {k: s[~s["cell_id"].isin(nearest)] if k.startswith("bbbc041") else s
            for k, s in sets.items()}


def rq2_table(model: str, nih_test: pd.DataFrame, keep=None) -> pd.DataFrame | None:
    """Every variant against the same NIH reference, plus the share of the drop
    recovered. `keep` filters the test sets identically in every variant."""
    ref = reference_draws(nih_test)
    frames = []
    for v in VARIANTS:
        sets = load(model, v)
        if sets is None:
            continue
        sets = {k: s for k, s in sets.items() if k != "nih_test"}
        t = summarise({"nih_test": nih_test, **(keep(sets) if keep else sets)}, ref=ref)
        frames.append(t[t["test_set"] != "nih_test"].assign(variant=v))
    if len(frames) < 2:
        return None
    t = pd.concat(frames)
    ref = pd.Series({**{m: pt for m, (pt, _) in ref[0].items()}, "auc": ref[1][0]})
    raw = t[t["variant"] == "raw"].set_index(["test_set", "metric"])["value"]
    masked = t[t["variant"] == "masked"].set_index(["test_set", "metric"])["value"]
    shares = []
    for r in t.itertuples():
        key, h = (r.test_set, r.metric), ref[r.metric]
        total = share_recovered(h, raw.get(key, np.nan), r.value)
        step = (share_recovered(h, raw.get(key, np.nan), masked.get(key, np.nan))
                if r.variant == "masked" else
                share_recovered(h, raw.get(key, np.nan), r.value)
                - share_recovered(h, raw.get(key, np.nan), masked.get(key, np.nan))
                if r.variant in ("reinhard", "histmatch") else np.nan)
        shares.append((total if r.variant != "raw" else np.nan, step))
    t["share_recovered_total"], t["share_recovered_step"] = zip(*shares)
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
    ap.add_argument("--model", required=True, help="folder name under outputs/predictions")
    args = ap.parse_args()

    raw = load(args.model, "raw")
    if raw is None:
        raise SystemExit(f"no raw predictions under {PREDICTIONS / args.model}")
    paths.TABLES.mkdir(parents=True, exist_ok=True)

    rq1 = summarise(raw)
    rq1.to_csv(paths.TABLES / f"rq1_{args.model}.csv", index=False)
    print(rq1.round(3).to_string(index=False))

    rq2 = rq2_table(args.model, raw["nih_test"])
    if rq2 is not None:
        rq2.to_csv(paths.TABLES / f"rq2_{args.model}.csv", index=False)
        print(f"wrote rq2_{args.model}.csv ({sorted(set(rq2['variant']))})")
        if MASK_STATUS.exists():
            rob = rq2_table(args.model, raw["nih_test"], keep=without_nearest)
            rob.to_csv(paths.TABLES / f"rq2_{args.model}_without_nearest.csv", index=False)
            print(f"wrote rq2_{args.model}_without_nearest.csv (robustness check)")
        else:
            print(f"skipped the robustness check: {MASK_STATUS.name} not found")

    rq3 = rq3_table(raw)
    rq3.to_csv(paths.TABLES / f"rq3_{args.model}.csv", index=False)
    print(f"wrote rq3_{args.model}.csv")
    figure(args.model, rq1, rq2)


if __name__ == "__main__":
    main()
