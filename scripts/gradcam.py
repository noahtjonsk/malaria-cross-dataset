"""Grad-CAM check: is each model's evidence on the cell, the background or the parasite?

    python scripts/gradcam.py                 # seed-0 models, 100 cells per group (CPU)
    python scripts/gradcam.py --smoke         # the *_smoke.pt checkpoints, 4 cells per group

An appendix check for RQ2 and RQ3 (malaria/saliency.py explains Grad-CAM and
its limits). A fixed random sample of cells, the same for every model:

    nih_test           50 parasitised + 50 uninfected NIH hold-out cells
    bbbc041/site_a     50 + 50, scored raw and with the background removed
    bbbc041/site_b     50 + 50, raw and background removed (the same cells)
    mpidb/falciparum   100 P. falciparum cells (MP-IDB has no uninfected cells)
    mpidb/other        100 cells of the other three species

For each model and cell: the share of the Grad-CAM map inside the cell (and,
for MP-IDB, on the expert-annotated parasite), next to that region's share of
the image area. The outcome (hit, miss, false alarm, correct rejection) is the
one in the stored prediction files, so it matches every other table; the model
is re-run on the CPU here only to get the map, and its probability is checked
against the stored one.

Sanity check (Adebayo et al., 2018): the same maps from each architecture with
random weights. Maps that do not depend on the trained weights would say
nothing about what the model learned.

Writes
    outputs/tables/gradcam_cells.csv        one row per model, cell and variant
    outputs/tables/gradcam_heat_share.csv   per model, group, variant and outcome: means
                                            with 95% intervals clustered by source image
                                            (NIH: patient)
    outputs/tables/gradcam_sanity.csv       trained vs random-weight maps on the NIH cells
    outputs/figures/gradcam_examples.png    one random cell per group, every model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402,F401  (before torch: avoids a duplicate OpenMP runtime on Windows)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.data import load_input, nih_cells, test_cells, to_tensor  # noqa: E402
from malaria.manifests import MpidbParasiteSeeds  # noqa: E402
from malaria.metrics import THRESHOLD, interval, resample_weights  # noqa: E402
from malaria.models import ARCHS, build_model  # noqa: E402
from malaria.saliency import (cell_region, gradcam, heat_share, target_layer,  # noqa: E402
                              to_input_grid)

SAMPLE_SEED = 0
# evaluate.py ran in mixed precision on a GPU, this runs in float32 on a CPU, so the
# probabilities differ slightly (MobileNetV2 most: median 0.0002, largest 0.011). A wrong
# checkpoint or input pipeline gives gaps of about 0.5.
PROB_TOL_MEDIAN, PROB_TOL_MAX = 0.001, 0.05
OUTCOMES = {(1, True): "hit", (1, False): "miss", (0, True): "false alarm",
            (0, False): "correct rejection"}


def sample_cells(n: int) -> list:
    """The sampled cells as one record per (group, variant, cell), same for every model."""
    rng = SAMPLE_SEED

    def per_label(cells, k):
        return pd.concat([g.sample(min(len(g), k), random_state=rng)
                          for _, g in cells.groupby("label_binary")]).sort_values("cell_id")

    items = []
    nih = per_label(nih_cells("test"), n // 2)
    items += [dict(group="nih_test", variant="raw", pred="raw_nih_test.csv", on_black=True,
                   cluster=r.patient_id, row=r) for r in nih.itertuples(index=False)]
    bbbc, masked = test_cells("bbbc041"), test_cells("bbbc041", "masked").set_index("cell_id")
    for site in ("site_a", "site_b"):
        cells = per_label(bbbc[bbbc["source_split"] == site], n // 2)
        for r in cells.itertuples(index=False):
            m = masked.loc[r.cell_id]
            items.append(dict(group=f"bbbc041/{site}", variant="raw", pred="raw_bbbc041.csv",
                              on_black=False, cluster=r.source_image, row=r))
            items.append(dict(group=f"bbbc041/{site}", variant="masked",
                              pred="masked_bbbc041.csv", on_black=True,
                              cluster=r.source_image, row=r._replace(path=m["path"])))
    mpidb = test_cells("mpidb_wholecell")
    pf = mpidb["species"] == "Falciparum"
    for group, cells in (("mpidb/falciparum", mpidb[pf]), ("mpidb/other", mpidb[~pf])):
        cells = cells.sample(min(len(cells), n), random_state=rng).sort_values("cell_id")
        items += [dict(group=group, variant="raw", pred="raw_mpidb_wholecell.csv",
                       on_black=False, cluster=r.source_image, row=r)
                  for r in cells.itertuples(index=False)]
    return items


def add_regions(items: list) -> None:
    """Cell and parasite masks on the 224 x 224 input grid, computed once per crop."""
    seeds = MpidbParasiteSeeds()
    for it in items:
        r = it["row"]
        bgr = cv2.imread(str(paths.ROOT / r.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(r.path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        parasite = None
        if it["group"].startswith("mpidb"):
            parasite = seeds(r.species, r.source_image, r.cell_id,
                             (int(r.r0), int(r.c0), int(r.r1), int(r.c1)))
        it["cell"] = to_input_grid(cell_region(rgb, it["on_black"], parasite).astype(np.uint8))
        it["parasite"] = None if parasite is None else to_input_grid(parasite.astype(np.uint8))


def load_net(arch: str, smoke: bool):
    """A trained checkpoint, as evaluate.py loads it, in eval mode on the CPU."""
    path = paths.MODELS / f"{paths.run_tag(arch, 0, smoke)}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}: download it from MyDrive/malaria_thesis/models/")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_model(ck["settings"]["arch"], pretrained=False)
    net.load_state_dict(ck["model"])
    return net.eval()


def maps_for(net, arch: str, items: list, batch: int = 16) -> tuple:
    """Grad-CAM maps and probabilities for every item, in order."""
    layer = target_layer(net, arch)
    cams, probs = [], []
    for i in range(0, len(items), batch):
        x = torch.stack([to_tensor(load_input(it["row"].path)) for it in items[i:i + batch]])
        c, p = gradcam(net, layer, x)
        cams.append(c)
        probs.append(p)
    return np.concatenate(cams), np.concatenate(probs)


def stored_probs(tag: str, items: list) -> np.ndarray:
    """Each item's probability from the stored prediction files (evaluate.py)."""
    files = {}
    out = np.empty(len(items))
    for k, it in enumerate(items):
        if it["pred"] not in files:
            p = pd.read_csv(paths.PREDICTIONS / tag / it["pred"], dtype={"cell_id": "string"})
            files[it["pred"]] = p.set_index("cell_id")["prob"]
        out[k] = files[it["pred"]].loc[it["row"].cell_id]
    return out


def cell_rows(arch: str, items: list, cams: np.ndarray, prob: np.ndarray,
              stored: np.ndarray) -> pd.DataFrame:
    """One row per item: outcome, heat and area shares."""
    rows = []
    for it, cam, p, s in zip(items, cams, prob, stored):
        r, label = it["row"], int(it["row"].label_binary)
        para = it["parasite"]
        rows.append({
            "arch": arch, "group": it["group"], "variant": it["variant"], "cell_id": r.cell_id,
            "cluster": str(it["cluster"]), "label": label,
            "outcome": OUTCOMES[(label, bool(s >= THRESHOLD))],
            "prob_stored": s, "prob_cpu": float(p),
            "heat_cell": heat_share(cam, it["cell"]), "area_cell": float(it["cell"].mean()),
            "heat_parasite": np.nan if para is None else heat_share(cam, para),
            "area_parasite": np.nan if para is None else float(para.mean()),
            "zero_map": bool(cam.sum() == 0)})
    return pd.DataFrame(rows)


def clustered_mean(t: pd.DataFrame, col: str, name: str) -> tuple:
    """Mean of `col` with a 95% interval from resampling clusters (paired across models)."""
    t = t[t[col].notna()]
    if not len(t):
        return np.nan, np.nan, np.nan
    by = t.groupby("cluster")[col].agg(["sum", "size"])
    ids, counts = resample_weights(by.index.to_numpy(), name)
    by = by.loc[ids]
    draws = (counts @ by["sum"].to_numpy()) / np.maximum(counts @ by["size"].to_numpy(), 1)
    return float(t[col].mean()), *interval(draws)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    """Per arch, group, variant and outcome (plus 'all'): n and clustered means."""
    rows = []
    for (arch, group, variant), t in cells.groupby(["arch", "group", "variant"], sort=False):
        for outcome, u in [("all", t), *t.groupby("outcome", sort=False)]:
            row = {"arch": arch, "group": group, "variant": variant, "outcome": outcome,
                   "n_cells": len(u), "n_zero_maps": int(u["zero_map"].sum())}
            for col in ("heat_cell", "area_cell", "heat_parasite", "area_parasite"):
                m, lo, hi = clustered_mean(u, col, f"gradcam/{group}/{outcome}")
                row.update({col: m, f"{col}_ci_low": lo, f"{col}_ci_high": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def sanity(arch: str, items: list, trained: np.ndarray) -> tuple:
    """Trained vs random-weight maps on the NIH cells: (summary row, the random maps)."""
    torch.manual_seed(SAMPLE_SEED)
    rand = build_model(arch, pretrained=False).eval()
    idx = [k for k, it in enumerate(items) if it["group"] == "nih_test"]
    cams, _ = maps_for(rand, arch, [items[k] for k in idx])
    rho = []
    for k, c in zip(idx, cams):
        a, b = trained[k].ravel(), c.ravel()
        rho.append(spearmanr(a, b).statistic if a.std() > 0 and b.std() > 0 else np.nan)
    heat_t = [heat_share(trained[k], items[k]["cell"]) for k in idx]
    heat_r = [heat_share(c, items[k]["cell"]) for k, c in zip(idx, cams)]
    return {"arch": arch, "n_cells": len(idx), "spearman_trained_vs_random": float(np.nanmean(rho)),
            "heat_cell_trained": float(np.nanmean(heat_t)),
            "heat_cell_random": float(np.nanmean(heat_r)),
            "area_cell": float(np.mean([items[k]["cell"].mean() for k in idx]))}, cams


def figure(items: list, maps: dict, random_maps: dict, cells: pd.DataFrame, dst: Path) -> None:
    """One random parasitised cell per group and variant; a row per model, plus random weights."""
    plots.set_style()
    rng = np.random.default_rng(SAMPLE_SEED)
    cols = []
    for key in dict.fromkeys((it["group"], it["variant"]) for it in items):
        pos = [k for k, it in enumerate(items) if (it["group"], it["variant"]) == key
               and int(it["row"].label_binary) == 1]
        cols.append((key, pos))
    picks = []
    chosen = {}
    for (group, variant), pos in cols:
        # the masked column shows the same cell as the raw column next to it
        cell_id = chosen.get(group) or items[pos[int(rng.integers(len(pos)))]]["row"].cell_id
        chosen[group] = cell_id
        picks.append(next(k for k in pos if items[k]["row"].cell_id == cell_id))
    rows = [(a, maps[a]) for a in maps] + [(f"{a} (random weights)", random_maps[a])
                                         for a in list(random_maps)[:1]]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(1.55 * len(picks), 1.7 * len(rows)))
    for i, (name, cams) in enumerate(rows):
        for j, k in enumerate(picks):
            ax = axes[i, j]
            it = items[k]
            ax.imshow(load_input(it["row"].path))
            if cams[k].max() > 0:   # the random-weight row has maps for NIH cells only
                ax.imshow(cams[k] / cams[k].max(), cmap="viridis", alpha=0.5)
            ax.contour(it["cell"], levels=[0.5], colors="white", linewidths=0.6)
            if it["parasite"] is not None:
                ax.contour(it["parasite"], levels=[0.5], colors="#ff6b6b", linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            arch = name.split(" ")[0]
            if " " not in name:
                out = cells[(cells["arch"] == arch) & (cells["cell_id"] == it["row"].cell_id)
                            & (cells["variant"] == it["variant"])]["outcome"].iloc[0]
                ax.set_title(out, fontsize=7)
            if i == 0:
                ax.text(0.5, 1.28, f"{it['group']}\n{it['variant']}", transform=ax.transAxes,
                        ha="center", fontsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.suptitle("Grad-CAM (yellow = strongest evidence for parasitised; no colour = no evidence "
                 "anywhere); white: cell, red: annotated parasite", fontsize=8, y=1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst)
    plt.close(fig)
    print(f"  saved {dst.relative_to(paths.ROOT).as_posix()}")


def main() -> None:
    """Sample the cells, compute every model's maps and write the tables and the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="cells per group")
    ap.add_argument("--smoke", action="store_true", help="*_smoke.pt checkpoints, 4 cells per group")
    args = ap.parse_args()
    n = 4 if args.smoke else args.n
    suffix = "_smoke" if args.smoke else ""
    torch.set_grad_enabled(True)

    items = sample_cells(n)
    add_regions(items)
    print(f"{len(items)} crops ({n} cells per group)")
    frames, sane, maps, random_maps = [], [], {}, {}
    for arch in ARCHS:
        net = load_net(arch, args.smoke)
        cams, prob = maps_for(net, arch, items)
        assert cams.shape == (len(items), paths.MODEL_INPUT, paths.MODEL_INPUT)
        assert (cams >= 0).all()
        stored = stored_probs(paths.run_tag(arch, 0), items)
        gap = np.abs(prob - stored)
        med, top = float(np.median(gap)), float(gap.max())
        print(f"{arch}: gap to the stored probability: median {med:.4f}, largest {top:.4f}")
        if not args.smoke and (med > PROB_TOL_MEDIAN or top > PROB_TOL_MAX):
            raise SystemExit(f"{arch}: CPU probabilities differ from the stored ones (median "
                             f"{med:.4f}, largest {top:.4f}); wrong checkpoint or input pipeline")
        frames.append(cell_rows(arch, items, cams, prob, stored))
        row, rcams = sanity(arch, items, cams)
        sane.append(row)
        maps[arch] = cams
        random_maps[arch] = np.zeros_like(cams)
        nih = [k for k, it in enumerate(items) if it["group"] == "nih_test"]
        random_maps[arch][nih] = rcams
    cells = pd.concat(frames, ignore_index=True)
    shares = cells[["heat_cell", "heat_parasite"]].stack()
    assert ((shares >= 0) & (shares <= 1 + 1e-9)).all()

    paths.TABLES.mkdir(parents=True, exist_ok=True)
    out = {"gradcam_cells": cells, "gradcam_heat_share": summarise(cells),
           "gradcam_sanity": pd.DataFrame(sane)}
    for name, t in out.items():
        t.to_csv(paths.TABLES / f"{name}{suffix}.csv", index=False)
        print(f"  wrote {name}{suffix}.csv ({len(t)} rows)")
    print(pd.DataFrame(sane).round(3).to_string(index=False))
    figure(items, maps, random_maps, cells, paths.FIGURES / f"gradcam_examples{suffix}.png")


if __name__ == "__main__":
    main()
