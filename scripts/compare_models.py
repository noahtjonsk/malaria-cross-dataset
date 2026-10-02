"""RQ1 model comparison: does one architecture lose less than another?

    python scripts/compare_models.py                                   # the three arms, seed 0
    python scripts/compare_models.py --models vgg16_s0 resnet50_s0 mobilenet_v2_s0
    python scripts/compare_models.py --models vgg16_s0_smoke mobilenet_v2_s0_smoke --name smoke

For every pair of models, test set and metric, the difference in drop
(drop of model A minus drop of model B, in percentage points) with a paired 95%
interval: both models are scored on the same cells, and each bootstrap
resample draws the same NIH patients and the same test photographs for both
(metrics.resample_weights is keyed by the set, not the model). A positive
difference means model A loses more than model B.

Refuses to compare models whose raw predictions came from different manifests
or crop contents (run.json, written by evaluate.py), or whose cells differ.

Writes outputs/tables/rq1_compare_<name>.csv and outputs/figures/R_compare_<name>.png
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.metrics import bootstrap, drop_draws, interval, reference_draws  # noqa: E402
from malaria.models import ARCHS  # noqa: E402
from malaria.results import REPORTED_SETS, SET_COLOURS, load, run_info  # noqa: E402


def check_same_inputs(models: list, sets: dict) -> None:
    """Same manifests (run.json) and the same cells, set by set, for every model."""
    hashes = {}
    for m in models:
        info = run_info(m)
        raw = (info or {}).get("variants", {}).get("raw")
        if raw is None:
            raise SystemExit(f"{m}: no run.json entry for the raw variant; "
                             "rerun evaluate.py --force so its inputs are recorded")
        hashes[m] = {**raw["manifests"], "crops": raw.get("crops", "unrecorded")}
    if len({tuple(sorted(h.items())) for h in hashes.values()}) > 1:
        raise SystemExit(f"models were scored on different manifests or crops: {hashes}")
    first = models[0]
    for name in REPORTED_SETS:
        for m in models[1:]:
            if not sets[m][name]["cell_id"].equals(sets[first][name]["cell_id"]):
                raise SystemExit(f"{m} and {first} have different cells in {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[paths.run_tag(a, 0) for a in ARCHS])
    ap.add_argument("--name", default="s0", help="suffix for the output files")
    args = ap.parse_args()
    if len(args.models) < 2:
        raise SystemExit("need at least two models")

    # Same row order for every model, so the cells line up one to one.
    sets = {m: {k: s.sort_values("cell_id").reset_index(drop=True)
                for k, s in load(m, "raw").items()} for m in args.models}
    check_same_inputs(args.models, sets)

    drops = {}   # (model, set, metric) -> (point drop, draws)
    for m in args.models:
        ref = reference_draws(sets[m]["nih_test"])
        for name in REPORTED_SETS[1:]:
            res = bootstrap(sets[m][name], name)
            for metric, (point, _) in res.items():
                if np.isfinite(point):
                    drops[m, name, metric] = drop_draws(ref, res, metric)

    rows = []
    for (name, metric) in dict.fromkeys((s, k) for _, s, k in drops):
        for a, b in itertools.combinations(args.models, 2):
            (da, wa), (db, wb) = drops[a, name, metric], drops[b, name, metric]
            lo, hi = interval(wa - wb)
            dlo_a, dhi_a = interval(wa)
            dlo_b, dhi_b = interval(wb)
            rows.append({"test_set": name, "metric": metric, "model_a": a, "model_b": b,
                         "drop_a": da, "drop_a_ci_low": dlo_a, "drop_a_ci_high": dhi_a,
                         "drop_b": db, "drop_b_ci_low": dlo_b, "drop_b_ci_high": dhi_b,
                         "diff_pp": da - db, "diff_ci_low": lo, "diff_ci_high": hi,
                         "interval_excludes_0": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    dst = paths.TABLES / f"rq1_compare_{args.name}.csv"
    out.to_csv(dst, index=False)
    print(out[["test_set", "metric", "model_a", "model_b", "diff_pp", "diff_ci_low",
               "diff_ci_high"]].round(2).to_string(index=False))
    print(f"wrote {dst.name}")
    figure(drops, args.models, args.name)


def figure(drops: dict, models: list, name: str) -> None:
    """Drop per model, one row per test set, one panel per metric."""
    plots.set_style()
    metrics = ["sensitivity", "specificity", "auc"]
    sets = list(REPORTED_SETS[1:])
    markers = dict(zip(models, "osD^v<>"))
    offset = {m: (i - (len(models) - 1) / 2) * 0.2 for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        ax.axvline(0, color="0.6", lw=0.8)
        for (m, s, k), (point, draws) in drops.items():
            if k != metric:
                continue
            lo, hi = interval(draws)
            c = SET_COLOURS[s]
            ax.errorbar(point, sets.index(s) + offset[m], xerr=[[point - lo], [hi - point]],
                        fmt=markers[m], color=c, ecolor=c, capsize=2, markersize=5)
        ax.set_yticks(range(len(sets)), sets)
        ax.set_ylim(len(sets) - 0.5, -0.5)
        ax.set_xlabel(f"drop in {metric} (pp)")
        ax.set_title(metric)
    handles = [plt.Line2D([], [], marker=markers[m], ls="", color="0.3", label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), fontsize=8,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Drop from the NIH hold-out set per architecture (95% intervals)",
                 fontsize=9, y=1.02)
    plots.save(fig, f"R_compare_{name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
