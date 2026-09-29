"""Turn one model's prediction files into the RQ1, RQ2 and RQ3 tables.

    python scripts/summarise_results.py --model vgg16_s0
    python scripts/summarise_results.py --model mobilenet_v2_s0_smoke        # after --smoke runs

Reads outputs/predictions/<model>/<variant>_<set>.csv (written by evaluate.py)
and applies the definitions in malaria/metrics.py: fixed 0.5 threshold, drop in
percentage points from the NIH hold-out set, 95% intervals from resampling
source images.

Test sets reported (lit review, Table 2)
    nih_test           reference
    bbbc041/site_a     sensitivity, specificity, AUC
    bbbc041/site_b     sensitivity, specificity, AUC
    mpidb/falciparum   sensitivity only (P. falciparum, the NIH species)
    mpidb/other        sensitivity only (P. malariae, ovale and vivax pooled)

Writes outputs/tables/
    rq1_<model>.csv    value, interval, drop and drop interval per set and metric
    rq2_<model>.csv    the same per RQ2 variant, plus the share of the drop recovered
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
                             share_recovered, summarise)

PREDICTIONS = paths.OUTPUTS / "predictions"
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


def rq2_table(model: str, rq1: pd.DataFrame) -> pd.DataFrame | None:
    frames = []
    for v in VARIANTS[1:]:
        sets = load(model, v)
        if sets is None:
            continue
        t = summarise({"nih_test": pd.read_csv(PREDICTIONS / model / "raw_nih_test.csv"), **sets})
        frames.append(t[t["test_set"] != "nih_test"].assign(variant=v))
    if not frames:
        return None
    t = pd.concat([rq1[rq1["test_set"] != "nih_test"].assign(variant="raw"), *frames])
    ref = rq1[rq1["test_set"] == "nih_test"].set_index("metric")["value"]
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


def figure(model: str, rq1: pd.DataFrame) -> None:
    plots.set_style()
    metrics = [m for m in ("sensitivity", "specificity", "auc") if m in set(rq1["metric"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 2.8), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        t = rq1[rq1["metric"] == metric].reset_index(drop=True)
        for i, r in t.iterrows():
            c = SET_COLOURS[r["test_set"]]
            ax.errorbar(100 * r["value"], i, xerr=[[100 * (r["value"] - r["ci_low"])],
                                                  [100 * (r["ci_high"] - r["value"])]],
                        fmt="o", color=c, ecolor=c, capsize=3)
        ax.set_yticks(range(len(t)), t["test_set"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"{metric} (%)" if metric != "auc" else "AUC (x100)")
        ax.set_title(metric)
    fig.suptitle(f"{model}: NIH hold-out vs external test sets (95% image-resampled intervals)",
                 fontsize=9)
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

    rq2 = rq2_table(args.model, rq1)
    if rq2 is not None:
        rq2.to_csv(paths.TABLES / f"rq2_{args.model}.csv", index=False)
        print(f"wrote rq2_{args.model}.csv ({sorted(set(rq2['variant']))})")

    rq3 = rq3_table(raw)
    rq3.to_csv(paths.TABLES / f"rq3_{args.model}.csv", index=False)
    print(f"wrote rq3_{args.model}.csv")
    figure(args.model, rq1)


if __name__ == "__main__":
    main()
