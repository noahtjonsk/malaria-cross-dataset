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
    """{set name: predictions} for one run and variant; raises if a file is missing."""
    files = prediction_files(tag, variant)
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{tag} {variant}: missing {missing}")
    sets = split_sets(pd.read_csv(files["bbbc041"]), pd.read_csv(files["mpidb_wholecell"]))
    if variant == "raw":
        sets = {"nih_test": pd.read_csv(files["nih_test"]), **sets}
    return sets


def run_info(tag: str) -> dict | None:
    """evaluate.py's record of the checkpoint and inputs, if it wrote one."""
    f = paths.PREDICTIONS / tag / "run.json"
    return json.loads(f.read_text()) if f.exists() else None
