"""Pool the per-run tables over training seeds, and check which seed-0 findings hold.

    python scripts/aggregate_seeds.py                     # seeds 0, 1 and 2 of all three architectures
    python scripts/aggregate_seeds.py --seeds 0 1

Reads the tables summarise_results.py and compare_models.py wrote for each run
(outputs/tables/rq1_<arch>_s<seed>.csv, rq1_compare_s<seed>.csv, ...). Each seed
is a separate training run of the same architecture on the same data; the
spread across seeds shows how much a result depends on the random start
(initial head weights, shuffle order, augmentation draws). Intervals within one
run come from resampling test images; the seed range is a separate, second
source of variation and is reported next to it, not combined with it.

A missing table, or a seed whose rows do not line up with the others, stops
the script, so no pooled number silently covers fewer seeds than stated.

Writes outputs/tables/
    rq1_seeds.csv          per arch, test set and metric: value and drop, mean and min-max over seeds
    rq1_f1_seeds.csv       precision and F1, the same way
    rq2_seeds.csv          per arch, variant, test set and metric: value, drop and share recovered
    rq3_seeds.csv          per arch and group: sensitivity, miss rate and share of a site's misses
    rq1_compare_seeds.csv  per model pair, test set and metric: difference in drop over seeds,
                           whether every seed points the same way, and in how many seeds the
                           Bonferroni-adjusted interval excludes 0
    seed_checks.csv        the seed-0 findings, one row per finding and seed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.models import ARCHS  # noqa: E402

# table -> (columns that identify a row within one run, columns pooled over seeds)
TABLES = {
    "rq1": (["test_set", "metric"], ["value", "drop_pp"]),
    "rq1_f1": (["test_set", "metric"], ["value"]),
    "rq2": (["variant", "test_set", "metric"],
            ["value", "drop_pp", "share_recovered_total", "share_recovered_step"]),
    "rq3": (["group"], ["sensitivity", "miss_rate", "share_of_site_misses"]),
}


def read_runs(table: str, seeds: list) -> pd.DataFrame:
    """One table for every arch and seed, stacked; stops if a run is missing or its rows differ."""
    keys, _ = TABLES[table]
    frames = []
    for arch in ARCHS:
        rows = None
        for seed in seeds:
            path = paths.TABLES / f"{table}_{paths.run_tag(arch, seed)}.csv"
            if not path.exists():
                raise SystemExit(f"missing {path.name}: run summarise_results.py for that run first")
            t = pd.read_csv(path)
            if rows is None:
                rows = t[keys]
            elif not t[keys].equals(rows):
                raise SystemExit(f"{path.name} has different rows from seed {seeds[0]}")
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def pool(t: pd.DataFrame, keys: list, values: list) -> pd.DataFrame:
    """Mean, min and max of each value column over seeds, one row per arch and key."""
    g = t.groupby(["arch", *keys], sort=False)
    out = g.size().rename("n_seeds").to_frame()
    for col in values:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_min"] = g[col].min()
        out[f"{col}_max"] = g[col].max()
    return out.reset_index()


def pool_compare(seeds: list) -> pd.DataFrame:
    """The RQ1 model comparison over seeds: mean difference, sign agreement, flagged count."""
    frames = []
    for seed in seeds:
        path = paths.TABLES / f"rq1_compare_s{seed}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path.name}: run compare_models.py --name s{seed} first")
        t = pd.read_csv(path)
        for col in ("model_a", "model_b"):
            t[col] = t[col].map(lambda tag: paths.parse_run_tag(tag)[0])
        frames.append(t.assign(seed=seed))
    t = pd.concat(frames, ignore_index=True)
    g = t.groupby(["test_set", "metric", "model_a", "model_b"], sort=False)
    out = g.size().rename("n_seeds").to_frame()
    out["diff_pp_mean"] = g["diff_pp"].mean()
    out["diff_pp_min"] = g["diff_pp"].min()
    out["diff_pp_max"] = g["diff_pp"].max()
    out["same_sign_all_seeds"] = g["diff_pp"].agg(lambda d: bool((d > 0).all() or (d < 0).all()))
    out["seeds_adj_interval_excludes_0"] = g["adj_interval_excludes_0"].sum()
    return out.reset_index()


def checks(rq1: pd.DataFrame, rq2: pd.DataFrame, rq3: pd.DataFrame) -> pd.DataFrame:
    """The seed-0 findings, evaluated per seed (one row each).

    RQ1  the lighter model does not lose less: MobileNetV2 has the largest
         sensitivity drop on a test set
    RQ2  histogram matching makes the models call more site_a cells parasitised:
         site_a specificity is lower with histmatch than raw
    RQ3  P. falciparum is detected less often than the other species
         (falciparum minus other < 0); rings are missed more often than
         trophozoites at each BBBC041 site
    """
    rows = []

    def add(finding, arch, seed, holds, value):
        rows.append({"finding": finding, "arch": arch, "seed": seed,
                     "holds": bool(holds), "value": value})

    sens = rq1[rq1["metric"] == "sensitivity"]
    for (test_set, seed), t in sens.groupby(["test_set", "seed"]):
        if test_set == "nih_test":
            continue
        worst = t.loc[t["drop_pp"].idxmax()]
        add(f"RQ1 MobileNetV2 has the largest sensitivity drop on {test_set}", worst["arch"],
            seed, worst["arch"] == "mobilenet_v2", worst["drop_pp"])
    spec = rq2[(rq2["test_set"] == "bbbc041/site_a") & (rq2["metric"] == "specificity")]
    for (arch, seed), t in spec.groupby(["arch", "seed"]):
        v = t.set_index("variant")["value"]
        add("RQ2 histmatch lowers site_a specificity", arch, seed,
            v["histmatch"] < v["raw"], 100 * (v["histmatch"] - v["raw"]))
    for (arch, seed), t in rq3.groupby(["arch", "seed"]):
        t = t.set_index("group")
        gap = t.loc["falciparum minus other (pp)", "sensitivity"]
        add("RQ3 P. falciparum detected less than other species", arch, seed, gap < 0, gap)
        for site in ("site_a", "site_b"):
            ring = t.loc[f"bbbc041/{site}/ring", "miss_rate"]
            troph = t.loc[f"bbbc041/{site}/trophozoite", "miss_rate"]
            add(f"RQ3 rings missed more than trophozoites on {site}", arch, seed,
                ring > troph, 100 * (ring - troph))
    return pd.DataFrame(rows)


def main() -> None:
    """Write the pooled tables and the per-seed check of each finding."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    runs = {name: read_runs(name, args.seeds) for name in TABLES}
    for name, (keys, values) in TABLES.items():
        out = pool(runs[name], keys, values)
        assert (out["n_seeds"] == len(args.seeds)).all(), name
        out.to_csv(paths.TABLES / f"{name}_seeds.csv", index=False)
        print(f"wrote {name}_seeds.csv ({len(out)} rows, seeds {args.seeds})")
    cmp = pool_compare(args.seeds)
    cmp.to_csv(paths.TABLES / "rq1_compare_seeds.csv", index=False)
    print(f"wrote rq1_compare_seeds.csv ({len(cmp)} rows)")

    found = checks(runs["rq1"], runs["rq2"], runs["rq3"])
    found.to_csv(paths.TABLES / "seed_checks.csv", index=False)
    print("wrote seed_checks.csv\n")
    summary = found.groupby("finding", sort=False)["holds"].agg(["sum", "size"])
    for finding, (held, n) in summary.iterrows():
        print(f"  {held:2d}/{n:<2d} {finding}")
    robust = cmp["same_sign_all_seeds"] & (cmp["seeds_adj_interval_excludes_0"] == len(args.seeds))
    print(f"\n  {int(cmp['same_sign_all_seeds'].sum())}/{len(cmp)} model differences point the same "
          f"way in every seed; {int(robust.sum())} of those are also flagged (adjusted "
          f"interval excludes 0) in every seed")
    if np.isnan(runs["rq1"]["value"]).any():
        raise AssertionError("NaN value in an RQ1 table")


if __name__ == "__main__":
    main()
