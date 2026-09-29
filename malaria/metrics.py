"""The measures the research questions are stated in (lit review, Table 2).

Task: binary classification of single-cell images, parasitised = positive class.

- Decision rule: parasitised when the model's probability is >= 0.5. The
  threshold is fixed before testing; no test set is used to move it.
- Sensitivity: parasitised cells called parasitised / all parasitised cells.
- Specificity: uninfected cells called uninfected / all uninfected cells. Not
  defined on MP-IDB, which has no uninfected cells.
- AUC: threshold-free check, on the NIH hold-out set and on BBBC041.
- Drop: value on the NIH hold-out set minus value on a test set, in
  percentage points.
- Share recovered (RQ2): (value after matching - value before) / drop.
  Undefined (NaN) when the drop is zero or negative: there is nothing to
  recover, and dividing by a tiny drop would produce meaningless shares.
- Uncertainty: 95% percentile intervals from resampling *source images*,
  because cells cut from one photograph share its stain, focus and lighting and
  are not independent. A cell-level bootstrap would give intervals that are too
  narrow.

Sensitivity and specificity are bootstrapped from per-image counts, so a
resample of 1,200 BBBC041 photographs costs one weighted sum rather than a pass
over 80,000 cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5
N_BOOT = 2000
ALPHA = 0.05


def _counts(pred: pd.DataFrame, cluster: str) -> pd.DataFrame:
    """Per-cluster true/false positive/negative counts at THRESHOLD."""
    y = pred["label_binary"].astype(int).to_numpy()
    yhat = (pred["prob"].to_numpy() >= THRESHOLD).astype(int)
    df = pd.DataFrame({"c": pred[cluster].astype(str).to_numpy(),
                       "tp": (y == 1) & (yhat == 1), "fn": (y == 1) & (yhat == 0),
                       "tn": (y == 0) & (yhat == 0), "fp": (y == 0) & (yhat == 1)})
    return df.groupby("c")[["tp", "fn", "tn", "fp"]].sum()


def _rates(tp, fn, tn, fp) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sensitivity": np.divide(tp, tp + fn),
                "specificity": np.divide(tn, tn + fp)}


def bootstrap_rates(pred: pd.DataFrame, cluster: str = "source_image",
                    n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimate and n_boot cluster-resampled draws of sensitivity and specificity.

    Returns {"sensitivity": (point, draws), "specificity": (point, draws)}; a
    rate with no cells of its class (specificity on MP-IDB) is NaN throughout.
    """
    c = _counts(pred, cluster)
    point = _rates(*(c[k].sum() for k in ("tp", "fn", "tn", "fp")))
    rng = np.random.default_rng(seed)
    # How often each cluster is drawn in each resample (n_boot x n_clusters).
    w = rng.multinomial(len(c), np.full(len(c), 1 / len(c)), size=n_boot)
    draws = _rates(*(w @ c[k].to_numpy() for k in ("tp", "fn", "tn", "fp")))
    return {k: (float(point[k]), draws[k]) for k in point}


def bootstrap_auc(pred: pd.DataFrame, cluster: str = "source_image",
                  n_boot: int = 500, seed: int = 0):
    """AUC and its cluster-resampled draws; NaN when only one class is present."""
    y = pred["label_binary"].astype(int).to_numpy()
    p = pred["prob"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan"), np.full(n_boot, np.nan)
    groups = pd.Series(np.arange(len(pred))).groupby(
        pred[cluster].astype(str).to_numpy()).apply(np.asarray).tolist()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        draws[b] = roc_auc_score(y[idx], p[idx]) if len(np.unique(y[idx])) == 2 else np.nan
    return float(roc_auc_score(y, p)), draws


def interval(draws: np.ndarray) -> tuple:
    d = np.asarray(draws, dtype=float)
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan"), float("nan")
    return (float(np.quantile(d, ALPHA / 2)), float(np.quantile(d, 1 - ALPHA / 2)))


def drop_pp(holdout: float, test: float) -> float:
    """NIH hold-out value minus test value, in percentage points."""
    return 100.0 * (holdout - test)


def share_recovered(holdout: float, before: float, after: float) -> float:
    """(after - before) / drop, NaN when the drop is not positive."""
    drop = holdout - before
    if not np.isfinite(drop) or drop <= 0:
        return float("nan")
    return (after - before) / drop


def summarise(preds: dict, reference: str = "nih_test", seed: int = 0) -> pd.DataFrame:
    """One row per test set and metric: value, 95% interval, drop and its interval.

    `preds` maps a test-set name to its prediction table (label_binary, prob,
    source_image). The drop interval comes from resampling the NIH hold-out set
    and the test set independently and differencing the draws.
    """
    ref = bootstrap_rates(preds[reference], seed=seed)
    ref_auc = bootstrap_auc(preds[reference], seed=seed)
    rows = []
    for name, pred in preds.items():
        rates = bootstrap_rates(pred, seed=seed + 1)
        auc = bootstrap_auc(pred, seed=seed + 1)
        for metric, (point, draws) in [*rates.items(), ("auc", auc)]:
            if not np.isfinite(point):
                continue
            ref_point, ref_draws = ref_auc if metric == "auc" else ref[metric]
            lo, hi = interval(draws)
            dlo, dhi = interval(100.0 * (ref_draws - draws))
            rows.append({"test_set": name, "metric": metric, "n_cells": len(pred),
                         "n_images": pred["source_image"].nunique(),
                         "value": point, "ci_low": lo, "ci_high": hi,
                         "drop_pp": np.nan if name == reference else drop_pp(ref_point, point),
                         "drop_ci_low": np.nan if name == reference else dlo,
                         "drop_ci_high": np.nan if name == reference else dhi})
    return pd.DataFrame(rows)


def self_test() -> None:
    """A perfect predictor must score 1 everywhere with no drop (continuous-analysis check)."""
    rng = np.random.default_rng(0)

    def perfect(n):
        y = rng.integers(0, 2, n)
        return pd.DataFrame({"label_binary": y, "prob": y.astype(float),
                             "source_image": rng.integers(0, 40, n)})

    t = summarise({"nih_test": perfect(500), "other": perfect(800)})
    assert (t["value"] == 1).all(), t
    assert (t["drop_pp"].fillna(0) == 0).all(), t
    assert share_recovered(0.9, 0.6, 0.75) == 0.5
    assert np.isnan(share_recovered(0.9, 0.95, 0.97))
    print("metrics self-test passed")


if __name__ == "__main__":
    self_test()
