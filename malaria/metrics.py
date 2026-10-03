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
  Undefined (NaN) unless the drop is clearly above zero, i.e. the lower end of
  its 95% interval is positive: with no measurable drop there is nothing to
  recover, and dividing by a drop near zero gives meaningless shares (e.g. -62
  on a 1.5-point drop whose interval includes 0).
- Uncertainty: 95% percentile intervals from resampling *source images*,
  because cells cut from one photograph share its stain, focus and lighting and
  are not independent. A cell-level bootstrap would give intervals that are too
  narrow. The NIH hold-out set is resampled by *patient*, one level up, since
  a patient's photographs come from one slide.
- Pairing: each set's resamples are keyed by its name (resample_weights), so
  two models, or two RQ2 variants, scored on the same cells are resampled
  identically, and the interval of their difference is a paired one.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5
N_BOOT = 2000
ALPHA = 0.05
BOOT_SEED = 0
REFERENCE_CLUSTER = "patient_id"   # NIH hold-out; test sets resample source images


def resample_weights(clusters, name: str, n_boot: int = N_BOOT) -> tuple:
    """How often each cluster is drawn in each resample of one named set.

    Returns (sorted cluster ids, n_boot x n_clusters counts). The random stream
    is keyed by BOOT_SEED and the set's name, so a set is resampled the same way
    in every table, for every model and every RQ2 variant scored on the same
    cells, while different sets (the NIH hold-out and a test set, or the two
    MP-IDB species groups) get independent streams.
    """
    ids = np.unique(np.asarray(clusters).astype(str))
    rng = np.random.default_rng([BOOT_SEED, zlib.crc32(name.encode())])
    return ids, rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot)


def _rates(tp, fn, tn, fp) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sensitivity": np.divide(tp, tp + fn),
                "specificity": np.divide(tn, tn + fp)}


def _weighted_auc(y, p, ci, n_clusters: int, weights: np.ndarray) -> np.ndarray:
    """AUC for each row of cluster weights, as if every cell were repeated as
    often as its cluster was drawn (Mann-Whitney statistic, ties count half).

    The numerator is bilinear in the weights: w' A w, where A[c, d] counts the
    (positive in cluster c, negative in cluster d) pairs the positive wins, ties
    counting half. A is built once (one sorted search per cluster), so every
    resample costs a small matrix product, with memory n_clusters^2.
    """
    pos, neg = y == 1, y == 0
    pc, pp = ci[pos], p[pos]
    A = np.zeros((n_clusters, n_clusters))
    for d in np.unique(ci[neg]):
        s = np.sort(p[neg & (ci == d)])
        lower = np.searchsorted(s, pp, side="left")
        tied = np.searchsorted(s, pp, side="right") - lower
        A[:, d] = np.bincount(pc, weights=lower + 0.5 * tied, minlength=n_clusters)
    n_pos = np.bincount(pc, minlength=n_clusters)
    n_neg = np.bincount(ci[neg], minlength=n_clusters)
    w = weights.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((w @ A) * w).sum(1) / ((w @ n_pos) * (w @ n_neg))


def bootstrap(pred: pd.DataFrame, name: str, cluster: str = "source_image",
              n_boot: int = N_BOOT) -> dict:
    """Point estimate and cluster-resampled draws of every metric for one set.

    Returns {"sensitivity" | "specificity" | "auc": (point, draws)}. A metric the
    set cannot define (specificity and AUC on parasitised-only cells) is NaN.
    Sensitivity and specificity come from per-cluster counts, so a resample of
    1,200 BBBC041 photographs is one weighted sum, not a pass over 80,000 cells.
    """
    clusters = pred[cluster].astype(str).to_numpy()
    ids, w = resample_weights(clusters, name, n_boot)
    ci = np.searchsorted(ids, clusters)
    y = pred["label_binary"].astype(int).to_numpy()
    p = pred["prob"].to_numpy(dtype=float)
    yhat = (p >= THRESHOLD).astype(int)
    counts = np.stack([np.bincount(ci, weights=((y == a) & (yhat == b)).astype(float),
                                   minlength=len(ids))
                       for a, b in ((1, 1), (1, 0), (0, 0), (0, 1))], axis=1)  # tp fn tn fp
    point = _rates(*counts.sum(0))
    draws = _rates(*(w @ counts).T)
    out = {k: (float(point[k]), draws[k]) for k in point}
    if len(np.unique(y)) < 2:
        out["auc"] = (float("nan"), np.full(n_boot, np.nan))
    else:
        auc = _weighted_auc(y, p, ci, len(ids), np.vstack([np.ones(len(ids)), w]))
        out["auc"] = (float(auc[0]), auc[1:])
    return out


def reference_draws(pred: pd.DataFrame, name: str = "nih_test") -> dict:
    """Bootstrap of the NIH hold-out set, resampling *patients*.

    NIH photographs of one patient come from one slide and staining session, so
    they are not independent either; the test sets have no patient IDs and are
    resampled by photograph.
    """
    return bootstrap(pred, name, cluster=REFERENCE_CLUSTER)


def interval(draws: np.ndarray, alpha: float = ALPHA) -> tuple:
    """Percentile interval at level 1 - alpha, ignoring NaN draws."""
    d = np.asarray(draws, dtype=float)
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan"), float("nan")
    return (float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2)))


def drop_pp(holdout: float, test: float) -> float:
    """NIH hold-out value minus test value, in percentage points."""
    return 100.0 * (holdout - test)


def drop_draws(ref: dict, test: dict, metric: str) -> tuple:
    """Point drop and its bootstrap draws (pp) from two bootstrap() results."""
    (rp, rd), (tp, td) = ref[metric], test[metric]
    return drop_pp(rp, tp), 100.0 * (rd - td)


def share_recovered(holdout: float, before: float, after: float,
                    drop_ci_low: float | None = None) -> float:
    """(after - before) / drop, NaN when the drop is not positive or, given the
    lower end of its 95% interval (pp), when that interval reaches zero."""
    drop = holdout - before
    if not np.isfinite(drop) or drop <= 0:
        return float("nan")
    if drop_ci_low is not None and not drop_ci_low > 0:
        return float("nan")
    return (after - before) / drop


def summarise(preds: dict, reference: str = "nih_test", ref: dict | None = None) -> pd.DataFrame:
    """One row per test set and metric: value, 95% interval, drop and its interval.

    `preds` maps a test-set name to its prediction table (label_binary, prob,
    source_image, and patient_id for the reference). The NIH hold-out set and
    each test set are resampled independently, and the drop interval comes from
    differencing their draws. `ref` is reference_draws() of the hold-out set;
    it is computed here when not given, and then `preds` must contain it.
    """
    ref = ref if ref is not None else reference_draws(preds[reference], reference)
    rows = []
    for name, pred in preds.items():
        is_ref = name == reference
        res = ref if is_ref else bootstrap(pred, name)
        for metric, (point, draws) in res.items():
            if not np.isfinite(point):
                continue
            lo, hi = interval(draws)
            drop, dd = drop_draws(ref, res, metric)
            dlo, dhi = interval(dd)
            rows.append({"test_set": name, "metric": metric, "n_cells": len(pred),
                         "n_images": pred["source_image"].nunique(),
                         "value": point, "ci_low": lo, "ci_high": hi,
                         "drop_pp": np.nan if is_ref else drop,
                         "drop_ci_low": np.nan if is_ref else dlo,
                         "drop_ci_high": np.nan if is_ref else dhi})
    return pd.DataFrame(rows)


def self_test() -> None:
    """Continuous-analysis checks on the metric code."""
    rng = np.random.default_rng(0)

    def fake(n, perfect=True):
        y = rng.integers(0, 2, n)
        img = rng.integers(0, 40, n)
        p = y.astype(float) if perfect else np.round(rng.random(n) * 0.6 + 0.4 * y, 2)
        return pd.DataFrame({"label_binary": y, "prob": p,
                             "source_image": img, "patient_id": img // 5})

    # A perfect predictor scores 1 everywhere with no drop.
    t = summarise({"nih_test": fake(500), "other": fake(800)})
    assert (t["value"] == 1).all(), t
    assert (t["drop_pp"].fillna(0) == 0).all(), t
    # The weighted AUC equals scikit-learn's (ties included), and so does a
    # resample on the cells it duplicates.
    d = fake(3000, perfect=False)
    b = bootstrap(d, "check")
    assert abs(b["auc"][0] - roc_auc_score(d["label_binary"], d["prob"])) < 1e-12
    ids, w = resample_weights(d["source_image"], "check")
    rep = np.repeat(np.arange(len(d)), w[0][np.searchsorted(ids, d["source_image"].astype(str))])
    assert abs(b["auc"][1][0] - roc_auc_score(d["label_binary"].iloc[rep],
                                              d["prob"].iloc[rep])) < 1e-12
    # A resample without one of the classes has no AUC.
    y, p = d["label_binary"].to_numpy(), d["prob"].to_numpy()
    ci = np.searchsorted(ids, d["source_image"].astype(str))
    only_pos = np.isin(np.arange(len(ids)), np.unique(ci[y == 1])) & ~np.isin(
        np.arange(len(ids)), np.unique(ci[y == 0]))
    if only_pos.any():
        assert np.isnan(_weighted_auc(y, p, ci, len(ids), only_pos[None, :].astype(int))[0])
    assert np.isnan(_weighted_auc(y, p, ci, len(ids), np.zeros((1, len(ids))))[0])
    # Same set, same draws (paired); a differently named set is independent.
    b2 = bootstrap(d.copy(), "check")
    assert all(np.array_equal(b[k][1], b2[k][1], equal_nan=True) for k in b)
    assert not np.array_equal(b["auc"][1], bootstrap(d, "other set")["auc"][1])
    assert share_recovered(0.9, 0.6, 0.75) == 0.5
    assert np.isnan(share_recovered(0.9, 0.95, 0.97))
    assert np.isnan(share_recovered(0.9, 0.6, 0.75, drop_ci_low=-0.5))
    assert share_recovered(0.9, 0.6, 0.75, drop_ci_low=20.0) == 0.5
    print("metrics self-test passed")


if __name__ == "__main__":
    self_test()
