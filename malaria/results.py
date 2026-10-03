"""Reading evaluate.py's prediction files back as the reported test sets.

Shared by summarise_results.py (one model) and compare_models.py (several), so
both cut site_a/site_b and P. falciparum/other species the same way.

    outputs/predictions/<tag>/<variant>_<set>.csv   one row per cell, with `prob`
    outputs/predictions/<tag>/run.json              checkpoint settings, code
                                                     commit and manifest hashes
"""
from __future__ import annotations

import json

import pandas as pd

from . import paths, plots
from .data import TEST_DATASETS

# The test sets results are reported on (lit review, Table 2), in table order.
SET_COLOURS = {"nih_test": plots.DOMAIN_COLORS["nih"],
               "bbbc041/site_a": plots.DOMAIN_COLORS["bbbc041/site_a"],
               "bbbc041/site_b": plots.DOMAIN_COLORS["bbbc041/site_b"],
               "mpidb/falciparum": plots.DOMAIN_COLORS["mpidb/Falciparum"],
               "mpidb/other": plots.DOMAIN_COLORS["mpidb/Vivax"]}
REPORTED_SETS = tuple(SET_COLOURS)


def split_sets(bbbc: pd.DataFrame, mpidb: pd.DataFrame) -> dict:
    """The reported external test sets, cut from one variant's prediction files."""
    return {"bbbc041/site_a": bbbc[bbbc["source_split"] == "site_a"],
            "bbbc041/site_b": bbbc[bbbc["source_split"] == "site_b"],
            "mpidb/falciparum": mpidb[mpidb["species"] == "Falciparum"],
            "mpidb/other": mpidb[mpidb["species"] != "Falciparum"]}


def prediction_files(tag: str, variant: str) -> dict:
    d = paths.PREDICTIONS / tag
    files = {n: d / f"{variant}_{n}.csv" for n in TEST_DATASETS}
    if variant == "raw":
        files["nih_test"] = d / "raw_nih_test.csv"
    return files


def load(tag: str, variant: str) -> dict:
    """{set name: predictions} for one run and variant.

    Raises if a file is missing, or if its row count differs from the count
    evaluate.py recorded for it (a truncated or replaced file).
    """
    files = prediction_files(tag, variant)
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{tag} {variant}: missing {missing}")
    frames = {n: pd.read_csv(f) for n, f in files.items()}
    recorded = (run_info(tag) or {}).get("files", {})
    for n, f in files.items():
        expected = recorded.get(f.name, {}).get("n_cells")
        if expected is not None and len(frames[n]) != expected:
            raise ValueError(f"{f}: {len(frames[n])} rows, run.json recorded {expected}")
    sets = split_sets(frames["bbbc041"], frames["mpidb_wholecell"])
    if variant == "raw":
        sets = {"nih_test": frames["nih_test"], **sets}
    return sets


def run_info(tag: str) -> dict | None:
    """evaluate.py's record of the checkpoint and inputs, if it wrote one."""
    f = paths.PREDICTIONS / tag / "run.json"
    return json.loads(f.read_text()) if f.exists() else None


def input_records(tag: str, variant: str) -> dict:
    """{prediction file name: {"manifests", "crops"}} that a run's files came from.

    Reads the per-file records evaluate.py now writes; for runs evaluated before
    those existed (the seed-0 baseline) it falls back to the per-variant record,
    whose crop digest was taken from the data zip. Raises if neither exists or
    the crops were not recorded, so placeholders can never count as a match.
    """
    info = run_info(tag) or {}
    out = {}
    for f in prediction_files(tag, variant).values():
        rec = info.get("files", {}).get(f.name) or info.get("variants", {}).get(variant)
        if not rec or rec.get("crops") in (None, "local", "unrecorded"):
            raise ValueError(f"{tag}: no input record with a crop digest for {f.name}; "
                             "rerun evaluate.py --force")
        out[f.name] = {"manifests": rec["manifests"], "crops": rec["crops"]}
    return out
