"""Score the background-removal methods before any of them touches a test crop.

    python scripts/validate_background.py
    python scripts/validate_background.py --per-domain 60 --methods otsu-gray rembg:u2net

rembg's models were trained on everyday photographs, not blood smears, so each
method is checked on three things before one is chosen:

1. NIH Polygon Set crops (raw variant). Dice and IoU of the kept cell against the
   hand-drawn outline -- the only domain with a true answer.
2. MP-IDB. Parasite retention: the share of the annotated parasite's pixels that
   the kept mask covers. A method that deletes parasite pixels is disqualified
   whatever its Dice, because the parasite is the thing being classified.
3. BBBC041 site_a and site_b. No ground truth, so the failure statuses (no
   foreground found, anchor missed) and the kept-area fraction are reported,
   and contact sheets are drawn for inspection.

Writes
    outputs/tables/background_method_validation.csv   one row per crop x method
    outputs/tables/background_method_summary.csv      one row per domain x method
    outputs/figures/P_background_<method>.png          original / result pairs
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from malaria import paths, plots  # noqa: E402
from malaria.background import (DEFAULT_METHODS, apply_mask, cell_mask, dice,  # noqa: E402
                                foreground, iou)
from malaria.manifests import (MpidbParasiteSeeds, load_manifest,  # noqa: E402
                               nihpoly_mask_path)


def sample(df, n, seed, by="label_binary"):
    """Up to n rows, split evenly over `by` so rare parasites are represented."""
    groups = [g for _, g in df.groupby(by, dropna=False)]
    per = max(1, n // len(groups))
    return pd.concat([g.sample(min(per, len(g)), random_state=seed) for g in groups])


def load_cells(per_domain: int, seed: int) -> pd.DataFrame:
    poly = load_manifest("nihpoly_cells.csv")
    poly = poly[(poly.source_split == "polygon_raw") & (poly.eval_group == "primary")]
    bbbc = load_manifest("bbbc041_cells.csv")
    bbbc = bbbc[bbbc.eval_group == "primary"]
    mp = load_manifest("mpidb_cells.csv")

    parts = [sample(poly, per_domain, seed).assign(domain="nihpoly/polygon_raw")]
    for site in ("site_a", "site_b"):
        parts.append(sample(bbbc[bbbc.source_split == site], per_domain, seed)
                     .assign(domain=f"bbbc041/{site}"))
    for species in paths.MPIDB_SPECIES:
        sub = mp[mp.source_split == species]
        parts.append(sub.sample(min(per_domain, len(sub)), random_state=seed)
                     .assign(domain=f"mpidb/{species}"))
    return pd.concat(parts, ignore_index=True)


def score_method(cells: pd.DataFrame, method: str, seeds) -> tuple[pd.DataFrame, dict]:
    """Score one method on every sampled crop."""
    records, examples = [], {}
    for row in cells.itertuples(index=False):
        rgb = plots.load_rgb(row.path)
        truth = seed = None
        if row.domain.startswith("nihpoly"):
            truth = cv2.imread(str(paths.ROOT / nihpoly_mask_path(row.path)),
                               cv2.IMREAD_GRAYSCALE) > 127
        elif row.domain.startswith("mpidb"):
            seed = seeds(row.species, row.source_image, row.cell_id,
                         (row.r0, row.c0, row.r1, row.c1))
        t0 = time.perf_counter()
        kept, status = cell_mask(foreground(rgb, method), seed)
        seconds = time.perf_counter() - t0
        rec = {"domain": row.domain, "cell_id": row.cell_id,
               "label_binary": row.label_binary, "method": method,
               "status": status, "kept_frac": float(kept.mean()),
               "seconds": seconds, "dice": np.nan, "iou": np.nan,
               "parasite_retention": np.nan}
        if truth is not None:
            rec["dice"], rec["iou"] = dice(kept, truth), iou(kept, truth)
        if seed is not None and seed.any():
            rec["parasite_retention"] = float((kept & seed).sum() / seed.sum())
        records.append(rec)
        ex = examples.setdefault(row.domain, [])
        if len(ex) < 8:
            ex.append((rgb, apply_mask(rgb, kept)))
    return pd.DataFrame(records), examples


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["domain", "method"], sort=False)
    out = pd.DataFrame({
        "n": g.size(),
        "dice_median": g["dice"].median(), "dice_p10": g["dice"].quantile(0.10),
        "iou_median": g["iou"].median(),
        "retention_median": g["parasite_retention"].median(),
        "retention_ge_0.99": g["parasite_retention"].apply(
            lambda s: float((s >= 0.99).mean()) if s.notna().any() else np.nan),
        "kept_frac_median": g["kept_frac"].median(),
        "status_empty": g["status"].apply(lambda s: float((s == "empty").mean())),
        "status_nearest": g["status"].apply(lambda s: float((s == "nearest").mean())),
        "sec_per_1000": g["seconds"].mean() * 1000,
    })
    return out.round(4).reset_index()


def draw(examples: dict, method: str):
    doms = list(examples)
    fig, axes = plt.subplots(2 * len(doms), 8, figsize=(8 * 1.1, 2 * len(doms) * 1.15))
    axes = np.asarray(axes).reshape(2 * len(doms), 8)
    for di, dom in enumerate(doms):
        for ci in range(8):
            for k in range(2):
                ax = axes[2 * di + k, ci]
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if ci < len(examples[dom]):
                    img = examples[dom][ci][k]
                    ax.imshow(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))
                else:
                    ax.set_visible(False)
                if ci == 0:
                    ax.set_ylabel(f"{dom}\n{'original' if k == 0 else 'kept'}",
                                  rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle(f"Background removal: {method}", y=1.002)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="suffix for the output tables, so a side comparison "
                         "does not overwrite the main one")
    args = ap.parse_args()

    plots.set_style()
    cells = load_cells(args.per_domain, args.seed)
    print(f"scoring {len(cells)} crops x {len(args.methods)} methods")
    print(cells.domain.value_counts(sort=False).to_string())

    # One method at a time, each appended to disk as soon as it finishes, so a
    # slow model or an interrupted run does not lose the methods already scored.
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    val_path = paths.TABLES / f"background_method_validation{suffix}.csv"
    sum_path = paths.TABLES / f"background_method_summary{suffix}.csv"
    val_path.unlink(missing_ok=True)
    seeds = MpidbParasiteSeeds()
    summaries = []
    for method in args.methods:
        # load the model before the clock starts, so seconds are per crop
        foreground(np.full((64, 64, 3), 200, np.uint8), method)
        res, examples = score_method(cells, method, seeds)
        res.to_csv(val_path, mode="a", header=not val_path.exists(), index=False)
        summaries.append(summarise(res))
        pd.concat(summaries, ignore_index=True).to_csv(sum_path, index=False)
        print(summaries[-1].to_string(index=False), flush=True)
        fig = draw(examples, method)
        plots.save(fig, f"P_background_{method.replace(':', '_')}")
        plt.close(fig)


if __name__ == "__main__":
    main()
