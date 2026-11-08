"""Report-ready tables and figures for the D4 Technical Report, pooled over seeds.

    python scripts/make_report_tables.py

Reads only committed tables (outputs/tables/), so it runs from a clean clone:
rq1_seeds.csv, rq2_seeds.csv, rq3_seeds.csv, seed_checks.csv (aggregate_seeds.py),
the per-run rq1/rq2/rq3_<arch>_s<seed>.csv tables and the training logs
<arch>_s<seed>_training_log.json. Nothing is re-scored.

Every headline number is the mean over the three seeds, with the lowest and
highest seed in brackets: "20.1 [13.2, 31.0]". Percentages and AUC are shown
x100; drops are in percentage points (pp) against the same model's NIH hold-out
value. Each .md file holds one Markdown table, built from the .csv next to it;
the technical report embeds the .md tables unchanged.

Writes outputs/tables/
    report_model_comparison.{csv,md}   RQ1: NIH hold-out values and the drop on each external set
    report_in_out_sample.{csv,md}      training -> NIH validation -> NIH hold-out -> external sets
    report_rq2.{csv,md}                RQ2: drop per test-input variant, share recovered by masking
    report_rq3.{csv,md}                RQ3: species gap, ring vs trophozoite miss rates
    report_compute.{csv,md}            parameters, epochs, best epoch, training time per architecture
    report_settings.md                 the training settings shared by all nine runs
and outputs/figures/
    report_rq1_drop.png                drop per architecture and test set (mean and seed range)
    report_rq2_tradeoff.png            sensitivity against specificity per test-input variant
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aggregate_seeds import read_runs  # noqa: E402
from malaria import paths, plots  # noqa: E402
from malaria.data import VARIANTS  # noqa: E402
from malaria.models import ARCHS  # noqa: E402

SEEDS = [0, 1, 2]
LABELS = {"vgg16": "VGG-16", "resnet50": "ResNet-50", "mobilenet_v2": "MobileNetV2"}
SETS = {"nih_test": "NIH hold-out", "bbbc041/site_a": "BBBC041 site_a",
        "bbbc041/site_b": "BBBC041 site_b", "mpidb/falciparum": "MP-IDB P. falciparum",
        "mpidb/other": "MP-IDB other species"}
EXTERNAL = [s for s in SETS if s != "nih_test"]
METRICS = {"sensitivity": "Sensitivity", "specificity": "Specificity", "auc": "AUC"}
VARIANT_LABELS = {"raw": "Raw crop", "masked": "Background removed",
                  "reinhard": "+ Reinhard", "histmatch": "+ Histogram matching"}
# categorical slots 1-3 of the dataviz palette; validated all-pairs (CVD dE >= 9.2).
# Slot 3 is below 3:1 contrast, so every figure has a table view (the .md tables).
COLOURS = {"vgg16": "#2a78d6", "resnet50": "#eb6834", "mobilenet_v2": "#1baf7a"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#d9d8d4"


def fmt(mean, lo, hi, nd: int = 1) -> str:
    """'mean [min, max]' (already scaled), or an empty string when the value is undefined."""
    if pd.isna(mean):
        return ""
    return f"{mean:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def to_markdown(df: pd.DataFrame) -> str:
    """A plain pipe table (no index), the form the report embeds."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write(name: str, csv: pd.DataFrame, md: pd.DataFrame | None) -> None:
    """Write <name>.csv and, when given, <name>.md to the tables folder."""
    csv.to_csv(paths.TABLES / f"{name}.csv", index=False, lineterminator="\n")
    if md is not None:
        (paths.TABLES / f"{name}.md").write_text(to_markdown(md), encoding="utf-8", newline="\n")
    print(f"wrote {name}.csv" + (" and .md" if md is not None else "") + f" ({len(csv)} rows)")


def check_pooled(pooled: pd.DataFrame, runs: pd.DataFrame, keys: list, col: str) -> None:
    """The pooled mean/min/max must equal a fresh pooling of the per-run tables."""
    g = runs.groupby(["arch", *keys])[col]
    fresh = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(), "n": g.size()})
    got = pooled.set_index(["arch", *keys])
    assert (fresh["n"] == len(SEEDS)).all(), f"{col}: not {len(SEEDS)} seeds per row"
    for stat in ("mean", "min", "max"):
        a, b = fresh[stat].sort_index(), got[f"{col}_{stat}"].reindex(fresh.index).sort_index()
        assert np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-12), f"{col}_{stat} differs"


# ---------------------------------------------------------------- RQ1 ---
def model_comparison(rq1: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """RQ1 table: NIH hold-out value per architecture, then the drop on each external set."""
    check_pooled(rq1, runs, ["test_set", "metric"], "value")
    check_pooled(rq1, runs, ["test_set", "metric"], "drop_pp")
    rows, md = [], []
    for test_set in SETS:
        for metric in METRICS:
            sub = rq1[(rq1["test_set"] == test_set) & (rq1["metric"] == metric)].set_index("arch")
            if sub.empty:
                continue
            assert list(sub.index) == list(ARCHS), (test_set, metric)
            nih = test_set == "nih_test"
            col = "value" if nih else "drop_pp"
            scale = 100 if nih else 1
            best = None if nih else sub["drop_pp_mean"].idxmin()
            k = None
            if best is not None:   # in how many seeds the smallest-mean model is also the smallest
                r = runs[(runs["test_set"] == test_set) & (runs["metric"] == metric)]
                winners = r.loc[r.groupby("seed")["drop_pp"].idxmin(), "arch"]
                k = int((winners == best).sum())
            line = {"Test set": SETS[test_set], "Metric": METRICS[metric],
                    "Shown as": ("AUC x100" if metric == "auc" else "value (%)") if nih else "drop (pp)"}
            for arch in ARCHS:
                cell = fmt(*(scale * sub.loc[arch, f"{col}_{s}"] for s in ("mean", "min", "max")))
                line[LABELS[arch]] = f"**{cell}**" if arch == best else cell
                rows.append({"arch": arch, "test_set": test_set, "metric": metric,
                             **{f"{c}_{s}": sub.loc[arch, f"{c}_{s}"]
                                for c in ("value", "drop_pp") for s in ("mean", "min", "max")},
                             "smallest_mean_drop": arch == best,
                             "seeds_smallest_drop": k if arch == best else np.nan})
            line["Smallest drop"] = "" if nih else f"{LABELS[best]} ({k}/{len(SEEDS)} seeds)"
            md.append(line)
    out = pd.DataFrame(rows)
    assert len(out) == len(rq1) == 33, len(out)
    write("report_model_comparison", out, pd.DataFrame(md))
    return out


# ------------------------------------------------- in vs out of sample ---
def training_log(arch: str, seed: int) -> dict:
    """A run's training log, with the epoch whose weights were kept (lowest NIH validation loss)."""
    log = json.loads((paths.TABLES / f"{paths.run_tag(arch, seed)}_training_log.json").read_text())
    h = log["history"]
    best = min(h, key=lambda r: r["val"]["loss"])
    assert abs(best["val"]["loss"] - log["best_val_loss"]) < 1e-12, (arch, seed)
    assert [r for r in h if r["best"]][-1]["epoch"] == best["epoch"], (arch, seed)
    assert len(h) == log["epoch"], (arch, seed)
    log["best_row"] = best
    return log


def in_out_sample(rq1_runs: pd.DataFrame) -> None:
    """Sensitivity and specificity from training data to each external set, per architecture."""
    rows = []
    for arch in ARCHS:
        for seed in SEEDS:
            best = training_log(arch, seed)["best_row"]
            for stage, part in (("Training (best epoch, augmented)", "train"),
                                ("NIH validation (best epoch)", "val")):
                for metric in ("sensitivity", "specificity"):
                    rows.append({"arch": arch, "seed": seed, "stage": stage, "metric": metric,
                                 "value": best[part][metric]})
        r = rq1_runs[(rq1_runs["arch"] == arch) & rq1_runs["metric"].isin(["sensitivity", "specificity"])]
        for _, x in r.iterrows():
            rows.append({"arch": arch, "seed": x["seed"], "stage": SETS[x["test_set"]],
                         "metric": x["metric"], "value": x["value"]})
    t = pd.DataFrame(rows)
    g = t.groupby(["stage", "metric", "arch"], sort=False)["value"]
    out = pd.DataFrame({"n_seeds": g.size(), "mean": g.mean(), "min": g.min(), "max": g.max()}).reset_index()
    assert (out["n_seeds"] == len(SEEDS)).all()
    stages = ["Training (best epoch, augmented)", "NIH validation (best epoch)", *SETS.values()]
    md = []
    for stage in stages:
        line = {"Data": stage}
        for arch in ARCHS:
            cells = []
            for metric in ("sensitivity", "specificity"):
                x = out[(out["stage"] == stage) & (out["metric"] == metric) & (out["arch"] == arch)]
                cells.append(f"{100 * x['mean'].iloc[0]:.1f}" if len(x) else "-")
            line[LABELS[arch]] = " / ".join(cells)
        md.append(line)
    assert len(md) == 7
    write("report_in_out_sample", out, pd.DataFrame(md))


# ---------------------------------------------------------------- RQ2 ---
def rq2_table(rq2: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """RQ2 table: drop per test-input variant, and the share of the raw drop recovered by masking."""
    check_pooled(rq2, runs, ["variant", "test_set", "metric"], "drop_pp")
    check_pooled(rq2, runs, ["variant", "test_set", "metric"], "value")
    assert tuple(rq2["variant"].unique()) == VARIANTS, rq2["variant"].unique()
    defined = (runs.groupby(["arch", "variant", "test_set", "metric"])["share_recovered_total"]
               .agg(lambda s: int(s.notna().sum())).rename("n_seeds_share_defined"))
    out = rq2.merge(defined.reset_index(), on=["arch", "variant", "test_set", "metric"])
    md = []
    for arch in ARCHS:
        for test_set in EXTERNAL:
            for metric in METRICS:
                sub = out[(out["arch"] == arch) & (out["test_set"] == test_set)
                          & (out["metric"] == metric)].set_index("variant")
                if sub.empty:
                    continue
                line = {"Model": LABELS[arch], "Test set": SETS[test_set], "Metric": METRICS[metric]}
                for v in VARIANTS:
                    line[VARIANT_LABELS[v]] = fmt(*(sub.loc[v, f"drop_pp_{s}"] for s in ("mean", "min", "max")))
                m = sub.loc["masked"]
                n = int(m["n_seeds_share_defined"])
                line["Share recovered by masking"] = (
                    f"{m['share_recovered_total_mean']:.2f} ({n}/{len(SEEDS)} seeds)" if n else "n/a")
                md.append(line)
    assert len(md) == 24, len(md)
    write("report_rq2", out, pd.DataFrame(md))
    return out


# ---------------------------------------------------------------- RQ3 ---
def rq3_table(rq3: pd.DataFrame, runs: pd.DataFrame, found: pd.DataFrame) -> None:
    """RQ3 table: species gap and ring vs trophozoite miss rates, with the seed counts that hold."""
    check_pooled(rq3, runs, ["group"], "sensitivity")
    check_pooled(rq3, runs, ["group"], "miss_rate")
    n = runs.groupby("group")["n_parasitised"].agg(["min", "max"])
    assert (n["min"].fillna(-1) == n["max"].fillna(-1)).all(), "cell counts differ between runs"
    held = found.groupby(["finding", "arch"])["holds"].sum()

    def k(finding, arch):
        return f"{int(held.get((finding, arch), 0))}/{len(SEEDS)}"

    spec = [  # label, group, column, scale, finding whose per-model seed count is shown
        ("Sensitivity, P. falciparum (%)", "mpidb/falciparum", "sensitivity", 100, None),
        ("Sensitivity, other species (%)", "mpidb/other", "sensitivity", 100, None),
        ("Gap, P. falciparum minus other (pp)", "falciparum minus other (pp)", "sensitivity", 1,
         "RQ3 P. falciparum detected less than other species"),
        ("site_a ring miss rate (%)", "bbbc041/site_a/ring", "miss_rate", 100, None),
        ("site_a trophozoite miss rate (%)", "bbbc041/site_a/trophozoite", "miss_rate", 100,
         "RQ3 rings missed more than trophozoites on site_a"),
        ("site_b ring miss rate (%)", "bbbc041/site_b/ring", "miss_rate", 100, None),
        ("site_b trophozoite miss rate (%)", "bbbc041/site_b/trophozoite", "miss_rate", 100,
         "RQ3 rings missed more than trophozoites on site_b"),
    ]
    rows, md = [], []
    for label, group, col, scale, finding in spec:
        cells = n.loc[group, "min"]
        line = {"Measure": label, "n cells": "" if pd.isna(cells) else f"{int(cells):,}"}
        for arch in ARCHS:
            x = rq3[(rq3["arch"] == arch) & (rq3["group"] == group)].iloc[0]
            line[LABELS[arch]] = fmt(*(scale * x[f"{col}_{s}"] for s in ("mean", "min", "max")))
            rows.append({"measure": label, "group": group, "arch": arch, "n_cells": cells,
                         **{s: scale * x[f"{col}_{s}"] for s in ("mean", "min", "max")},
                         "finding": finding or "",
                         "seeds_finding_holds": held.get((finding, arch), np.nan) if finding else np.nan})
        md.append(line)
        if finding:
            what = ("P. falciparum detected less" if "falciparum" in finding
                    else "rings missed more than trophozoites")
            md.append({"Measure": f"... seeds where {what}", "n cells": "",
                       **{LABELS[a]: k(finding, a) for a in ARCHS}})
    write("report_rq3", pd.DataFrame(rows), pd.DataFrame(md))


# ------------------------------------------------------------ compute ---
# keys that name the run or record where it was trained, not how
PROVENANCE_KEYS = ("arch", "seed", "git_commit", "resumed_at")


def compute_table() -> None:
    """Parameters, epochs, the kept epoch and training time per architecture and seed."""
    from malaria.models import build_model, count_parameters
    rows, settings = [], None
    for arch in ARCHS:
        params = count_parameters(build_model(arch, pretrained=False))
        for seed in SEEDS:
            log = training_log(arch, seed)
            s = {k: v for k, v in log["settings"].items() if k not in PROVENANCE_KEYS}
            assert log["settings"]["arch"] == arch and log["settings"]["seed"] == seed
            assert settings is None or s == settings, f"{arch}_s{seed} trained with other settings"
            settings = s
            sec = [r["seconds"] for r in log["history"]]
            rows.append({"arch": arch, "seed": seed, "parameters": params,
                         "epochs_run": len(sec), "best_epoch": log["best_row"]["epoch"],
                         "best_val_loss": log["best_val_loss"], "train_minutes": sum(sec) / 60,
                         "minutes_per_epoch": np.mean(sec) / 60})
    t = pd.DataFrame(rows)
    md = []
    for arch in ARCHS:
        x = t[t["arch"] == arch]
        md.append({"Model": LABELS[arch],
                   "Parameters (M)": f"{x['parameters'].iloc[0] / 1e6:.1f}",
                   "Epochs run (seeds 0, 1, 2)": ", ".join(map(str, x["epochs_run"])),
                   "Kept epoch (seeds 0, 1, 2)": ", ".join(map(str, x["best_epoch"])),
                   "Min per epoch": f"{x['minutes_per_epoch'].mean():.1f}",
                   "Training time (min)": fmt(x["train_minutes"].mean(), x["train_minutes"].min(),
                                              x["train_minutes"].max(), 0)})
    write("report_compute", t, pd.DataFrame(md))
    assert not settings["smoke"] and settings["pretrained"]
    names = {"lr": "Learning rate (Adam)", "batch_size": "Batch size", "max_epochs": "Maximum epochs",
             "patience": "Early-stopping patience (epochs, NIH validation loss)",
             "weight_decay": "Weight decay", "input": "Input size (px)",
             "augment": "Training augmentation", "threshold": "Decision threshold",
             "pretrained": "ImageNet-pretrained, all layers trained"}
    assert set(names) <= set(settings), set(names) - set(settings)
    md = pd.DataFrame([{"Setting": names[k], "Value": str(settings[k])} for k in names])
    (paths.TABLES / "report_settings.md").write_text(to_markdown(md), encoding="utf-8", newline="\n")
    print("wrote report_settings.md")


# ------------------------------------------------------------ figures ---
def drop_figure(cmp: pd.DataFrame) -> None:
    """Dot = mean drop over seeds, bar = lowest to highest seed, per external set and metric."""
    plots.set_style()
    groups = [(s, m) for s in EXTERNAL for m in METRICS
              if ((cmp["test_set"] == s) & (cmp["metric"] == m)).any()]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(groups) + 0.9))
    offsets = dict(zip(ARCHS, (-0.22, 0.0, 0.22)))
    for i, (s, m) in enumerate(groups):
        y0 = len(groups) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y0 - 0.45, y0 + 0.45, color="#f3f2ef", zorder=0, lw=0)
        for arch in ARCHS:
            x = cmp[(cmp["test_set"] == s) & (cmp["metric"] == m) & (cmp["arch"] == arch)].iloc[0]
            y = y0 - offsets[arch]
            ax.plot([x["drop_pp_min"], x["drop_pp_max"]], [y, y], color=COLOURS[arch], lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(x["drop_pp_mean"], y, "o", ms=6, color=COLOURS[arch], mec="white", mew=1.2, zorder=3)
    ax.axvline(0, color=INK_2, lw=1, zorder=1)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{SETS[s]}, {METRICS[m] if m == 'auc' else m}" for s, m in reversed(groups)])
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, alpha=1, lw=0.6)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.tick_params(axis="x", labelcolor=INK_2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("drop vs the same model's NIH hold-out value (pp)", color=INK_2)
    handles = [Line2D([], [], color=COLOURS[a], marker="o", ms=6, mec="white", lw=2, label=LABELS[a])
               for a in ARCHS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5,
              handlelength=1.6, columnspacing=1.6)
    ax.text(1.0, -0.13 - 0.02 * (8 - len(groups)), "dot = mean of seeds 0-2, bar = lowest to highest seed",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    out = paths.FIGURES / "report_rq1_drop.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT).as_posix()}")


def tradeoff_figure(rq2: pd.DataFrame, rq1: pd.DataFrame) -> None:
    """Sensitivity against specificity on each BBBC041 site, per architecture and test-input variant."""
    plots.set_style()
    markers = {"raw": "o", "masked": "s", "reinhard": "^", "histmatch": "D"}
    paths_ = [("raw", "masked"), ("masked", "reinhard"), ("masked", "histmatch")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), sharey=True)
    nih = rq1[rq1["test_set"] == "nih_test"].pivot(index="arch", columns="metric", values="value_mean")
    for ax, site in zip(axes, ("bbbc041/site_a", "bbbc041/site_b")):
        v = rq2[rq2["test_set"] == site].pivot_table(index=["arch", "variant"], columns="metric",
                                                     values="value_mean")
        for arch in ARCHS:
            for a, b in paths_:
                ax.plot([100 * v.loc[(arch, a), "specificity"], 100 * v.loc[(arch, b), "specificity"]],
                        [100 * v.loc[(arch, a), "sensitivity"], 100 * v.loc[(arch, b), "sensitivity"]],
                        color=COLOURS[arch], lw=1, alpha=0.55, zorder=2)
            for var in VARIANTS:
                ax.plot(100 * v.loc[(arch, var), "specificity"], 100 * v.loc[(arch, var), "sensitivity"],
                        markers[var], ms=7 if var != "histmatch" else 6, color=COLOURS[arch],
                        mec="white", mew=1, zorder=3)
        ax.plot(100 * nih["specificity"].mean(), 100 * nih["sensitivity"].mean(), "*", ms=11,
                color=INK_2, mec="white", mew=0.8, zorder=4)
        ax.annotate("NIH hold-out", (100 * nih["specificity"].mean(), 100 * nih["sensitivity"].mean()),
                    xytext=(-3, 7), textcoords="offset points", ha="right", fontsize=7.5, color=INK_2)
        ax.set_title(SETS[site], color=INK)
        ax.set_xlim(-3, 103)
        ax.set_ylim(0, 108)
        ax.set_xlabel("specificity (%)", color=INK_2)
        ax.grid(color=GRID, alpha=1, lw=0.6)
    axes[0].set_ylabel("sensitivity (%)", color=INK_2)
    arch_h = [Line2D([], [], color=COLOURS[a], lw=2, label=LABELS[a]) for a in ARCHS]
    var_h = [Line2D([], [], color=INK_2, marker=markers[v], ls="", ms=6, label=VARIANT_LABELS[v])
             for v in VARIANTS]
    fig.legend(handles=arch_h, loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3,
               fontsize=7.5, handlelength=1.5, columnspacing=1.6)
    fig.legend(handles=var_h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4,
               fontsize=7.5, handlelength=1.0, columnspacing=1.4)
    fig.text(0.99, -0.02, "mean of seeds 0-2; lines join each step to the input it starts from",
             ha="right", va="top", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    out = paths.FIGURES / "report_rq2_tradeoff.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT).as_posix()}")


def main() -> None:
    """Build every report table and figure from the committed result tables."""
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    pooled = {n: pd.read_csv(paths.TABLES / f"{n}_seeds.csv") for n in ("rq1", "rq2", "rq3")}
    runs = {n: read_runs(n, SEEDS) for n in ("rq1", "rq2", "rq3")}
    for n, t in runs.items():
        assert sorted(t["seed"].unique()) == SEEDS and t.groupby("arch")["seed"].nunique().eq(3).all(), n
    found = pd.read_csv(paths.TABLES / "seed_checks.csv")

    cmp = model_comparison(pooled["rq1"], runs["rq1"])
    in_out_sample(runs["rq1"])
    rq2_table(pooled["rq2"], runs["rq2"])
    rq3_table(pooled["rq3"], runs["rq3"], found)
    compute_table()
    drop_figure(cmp)
    tradeoff_figure(pooled["rq2"], pooled["rq1"])


if __name__ == "__main__":
    main()
