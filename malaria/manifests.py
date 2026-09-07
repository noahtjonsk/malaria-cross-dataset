"""Per-cell manifests with one schema shared by all three datasets.

Every later stage of the project -- training, cross-dataset evaluation, the RQ2
degradation sweep, the RQ3 per-class breakdown -- reads cells through these
tables rather than by walking directories, so that label definitions and
exclusions are decided in exactly one place.

Schema
------
dataset        nih | bbbc041 | mpidb
source_split   NIH: parasitized/uninfected. BBBC041: site_a/site_b (the two
               acquisition batches). MP-IDB: the species folder.
cell_id        unique within the project
path           image on disk, relative to the project root
label_binary   1 parasitised, 0 uninfected, <NA> where the source does not
               support a binary call (see eval_group)
eval_group     primary | excluded_difficult | excluded_leukocyte
stage          ring/trophozoite/schizont/gametocyte, or <NA>/unknown
species        MP-IDB only
patient_id     NIH only; the C### prefix, used to group the train/val/test split
source_image   stem of the slide image the cell came from
r0 c0 r1 c1    crop window in source-image coordinates (test sets only)
at_border      the crop window was clipped by the image edge
pad_frac       the padding constant the crop was cut with
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm.auto import tqdm as _tqdm

from . import paths
from .crops import mask_components, read_gray_mask, square_padded_box

COLUMNS = ["dataset", "source_split", "cell_id", "path", "label_binary",
           "eval_group", "stage", "species", "patient_id", "source_image",
           "r0", "c0", "r1", "c1", "at_border", "pad_frac"]

# --- BBBC041 label policy -------------------------------------------------
# The NIH negative class is uninfected *red blood cells*, so BBBC041 leukocytes
# are a class the model was never trained on and cannot fairly be scored as
# either label; "difficult" cells are ambiguous by the annotators own account.
# Both are cropped and kept in the manifest, but held out of the primary metric
# and reported separately.
BBBC041_NEGATIVE = {"red blood cell"}
BBBC041_POSITIVE = {"ring", "trophozoite", "schizont", "gametocyte"}
BBBC041_EXCLUDED = {"difficult": "excluded_difficult",
                    "leukocyte": "excluded_leukocyte"}


# Nullable dtypes, fixed here so the three manifests concatenate predictably
# even though each leaves a different set of columns empty.
DTYPES = {"label_binary": "Int64", "r0": "Int64", "c0": "Int64",
          "r1": "Int64", "c1": "Int64", "pad_frac": "Float64",
          "at_border": "boolean", "dataset": "string", "source_split": "string",
          "cell_id": "string", "path": "string", "eval_group": "string",
          "stage": "string", "species": "string", "patient_id": "string",
          "source_image": "string"}


def tqdm(iterable, **kw):
    """Progress bars in a notebook or terminal, silence in a redirected log."""
    kw.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(iterable, **kw)


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(COLUMNS)).astype(DTYPES)


def _rel(p) -> str:
    return str(Path(p).resolve().relative_to(paths.ROOT)).replace("\\", "/")


# =========================================================================
# NIH
# =========================================================================
def build_nih_manifest() -> pd.DataFrame:
    rows = []
    for folder, label, split in ((paths.NIH_PARASITIZED, 1, "parasitized"),
                                 (paths.NIH_UNINFECTED, 0, "uninfected")):
        for f in sorted(folder.iterdir()):
            # Thumbs.db sits in both class folders; it is why the per-class
            # count is 13,779 rather than the 13,780 quoted in the literature.
            if f.suffix.lower() != ".png":
                continue
            m = re.match(r"(C\d+)", f.name)
            rows.append({
                "dataset": "nih", "source_split": split, "cell_id": f.stem,
                "path": _rel(f), "label_binary": label, "eval_group": "primary",
                "stage": pd.NA, "species": pd.NA,
                "patient_id": m.group(1) if m else pd.NA,
                "source_image": f.name.rsplit("_cell_", 1)[0],
                "r0": pd.NA, "c0": pd.NA, "r1": pd.NA, "c1": pd.NA,
                "at_border": False, "pad_frac": pd.NA,
            })
    return _frame(rows)


# =========================================================================
# BBBC041
# =========================================================================
def build_bbbc041_manifest(write_crops: bool = True,
                           skip_existing: bool = True) -> pd.DataFrame:
    rows = []
    for split, jf in paths.BBBC041_JSON.items():
        records = json.loads(jf.read_text())
        for rec in tqdm(records, desc=f"bbbc041/{split}", leave=False):
            src = paths.BBBC041_DIR / rec["image"]["pathname"].lstrip("/")
            stem = src.stem
            h, w = rec["image"]["shape"]["r"], rec["image"]["shape"]["c"]

            planned = []
            for i, obj in enumerate(rec["objects"]):
                cat = obj["category"]
                bb = obj["bounding_box"]
                box = square_padded_box(bb["minimum"]["r"], bb["minimum"]["c"],
                                        bb["maximum"]["r"], bb["maximum"]["c"],
                                        h, w)
                out = (paths.CROPS_BBBC041 / split / cat.replace(" ", "_")
                       / f"{stem}_{i}.png")

                if cat in BBBC041_EXCLUDED:
                    label, group, stage = pd.NA, BBBC041_EXCLUDED[cat], pd.NA
                elif cat in BBBC041_NEGATIVE:
                    label, group, stage = 0, "primary", pd.NA
                elif cat in BBBC041_POSITIVE:
                    label, group, stage = 1, "primary", cat
                else:
                    raise ValueError(f"unmapped BBBC041 category: {cat!r}")
                planned.append((out, box, label, group, stage, i))

            image = None
            if write_crops and any(not (skip_existing and p[0].exists())
                                   for p in planned):
                image = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"could not read slide: {src}")

            for out, box, label, group, stage, i in planned:
                if image is not None and not (skip_existing and out.exists()):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), box.apply(image))
                rows.append({
                    "dataset": "bbbc041", "source_split": split,
                    "cell_id": f"{stem}_{i}", "path": _rel(out),
                    "label_binary": label, "eval_group": group, "stage": stage,
                    "species": pd.NA, "patient_id": pd.NA, "source_image": stem,
                    "r0": box.r0, "c0": box.c0, "r1": box.r1, "c1": box.c1,
                    "at_border": box.at_border, "pad_frac": paths.CROP_PAD_FRAC,
                })
    return _frame(rows)


# =========================================================================
# MP-IDB
# =========================================================================
def _stem_key(stem: str) -> tuple:
    """Slide id, image index, stage suffix -- with the stage separator normalised.

    MP-IDB stems are {slide_id}-{image_index}-{stages}. Three gt/ files write the
    stage suffix with a hyphen (...-R-T.jpg) where img/ uses an underscore
    (...-R_T.jpg), so the two folders cannot be paired on the raw stem.
    """
    slide, index, stages = stem.split("-", 2)
    return slide, index, stages.replace("-", "_")


def _pair_img_gt(species: str):
    """Yield (stem, img_path, gt_path) for one species.

    One source file is named ...-R_T.jpg in img/ but ...-R-T.jpg in gt/, so the
    stage separator is normalised before pairing.
    """
    gt_by_key = {}
    for g in (paths.MPIDB_DIR / species / "gt").iterdir():
        if g.suffix.lower() == ".jpg":
            gt_by_key[_stem_key(g.stem)] = g

    for im in sorted((paths.MPIDB_DIR / species / "img").iterdir()):
        if im.suffix.lower() != ".jpg":
            continue
        gt = gt_by_key.get(_stem_key(im.stem))
        if gt is None:
            raise FileNotFoundError(f"no gt mask for {im}")
        yield im.stem, im, gt


def _shipped_stage_map(species: str):
    """Expert per-parasite stage labels from the shipped crops/ folders.

    Returns {image_stem: [stage, ...]} in ascending crop-index order, which the
    dataset authors assigned left to right across the image. Falciparum indexes
    from 0 and Vivax from 1, so only the order is used, never the index value.
    """
    cdir = paths.MPIDB_DIR / species / "crops"
    if not cdir.exists():
        return {}
    by_stem = collections.defaultdict(dict)
    for letter in "RTSG":
        for f in (cdir / letter).iterdir():
            m = re.match(r"(.+)_(\d+)\.png$", f.name)
            if m:
                by_stem[m.group(1)][int(m.group(2))] = paths.STAGE_NAMES[letter]
    return {stem: [d[i] for i in sorted(d)] for stem, d in by_stem.items()}


def _filename_stages(stem: str):
    return [paths.STAGE_NAMES[c] for c in _stem_key(stem)[2].split("_")
            if c in paths.STAGE_NAMES]


def build_mpidb_manifest(write_crops: bool = True,
                         skip_existing: bool = True):
    rows, audit = [], []
    for species in paths.MPIDB_SPECIES:
        shipped = _shipped_stage_map(species)
        for stem, img_path, gt_path in tqdm(list(_pair_img_gt(species)),
                                            desc=f"mpidb/{species}",
                                            leave=False):
            comps = mask_components(read_gray_mask(gt_path))
            fn_stages = _filename_stages(stem)

            # Stage labels. Where the authors shipped crops, carry their expert
            # per-parasite labels across by left-to-right order. Otherwise a
            # single-stage filename applies to every parasite in the image, and
            # a multi-stage one cannot be resolved per parasite -- the README
            # lists stages by first appearance, so R_S may be R, S, R.
            if stem in shipped and len(shipped[stem]) == len(comps):
                stages, source = shipped[stem], "shipped_crops"
            elif len(fn_stages) == 1:
                stages, source = fn_stages * len(comps), "filename_single_stage"
            else:
                stages, source = ["unknown"] * len(comps), "ambiguous_multistage"

            audit.append({"species": species, "source_image": stem,
                          "n_components": len(comps),
                          "n_shipped": len(shipped.get(stem, [])),
                          "stage_source": source})

            planned = [(paths.CROPS_MPIDB / species / stages[i] / f"{stem}_{i}.png",
                        comps[i][:4], stages[i], i)
                       for i in range(len(comps))]

            image = None
            if write_crops and any(not (skip_existing and p[0].exists())
                                   for p in planned):
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"could not read slide: {img_path}")

            ih, iw = (image.shape[:2] if image is not None
                      else read_gray_mask(gt_path).shape[:2])
            for out, (r0, c0, r1, c1), stage, i in planned:
                box = square_padded_box(r0, c0, r1, c1, ih, iw)
                if image is not None and not (skip_existing and out.exists()):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), box.apply(image))
                rows.append({
                    "dataset": "mpidb", "source_split": species,
                    "cell_id": f"{species}_{stem}_{i}", "path": _rel(out),
                    "label_binary": 1,   # MP-IDB ships no uninfected cells
                    "eval_group": "primary", "stage": stage, "species": species,
                    "patient_id": pd.NA, "source_image": stem,
                    "r0": box.r0, "c0": box.c0, "r1": box.r1, "c1": box.c1,
                    "at_border": box.at_border, "pad_frac": paths.CROP_PAD_FRAC,
                })
    return _frame(rows), pd.DataFrame(audit)


# =========================================================================
def build_all(write_crops: bool = True, skip_existing: bool = True):
    paths.ensure_dirs()

    nih = build_nih_manifest()
    nih.to_csv(paths.MANIFESTS / "nih_cells.csv", index=False)

    bbbc = build_bbbc041_manifest(write_crops, skip_existing)
    bbbc.to_csv(paths.MANIFESTS / "bbbc041_cells.csv", index=False)

    mp, audit = build_mpidb_manifest(write_crops, skip_existing)
    mp.to_csv(paths.MANIFESTS / "mpidb_cells.csv", index=False)
    audit.to_csv(paths.MANIFESTS / "mpidb_stage_audit.csv", index=False)

    allc = pd.concat([nih, bbbc, mp], ignore_index=True)
    allc.to_csv(paths.MANIFESTS / "all_cells.csv", index=False)
    return allc, audit


def load_manifest(name: str = "all_cells.csv") -> pd.DataFrame:
    """Read a manifest back with the dtypes it was written with."""
    return pd.read_csv(paths.MANIFESTS / name, dtype=DTYPES)
