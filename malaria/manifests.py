"""Per-cell manifests with one schema shared by all three datasets.

Every later stage of the project -- training, cross-dataset evaluation, the RQ2
degradation sweep, the RQ3 per-class breakdown -- reads cells through these
tables rather than by walking directories, so that label definitions and
exclusions are decided in exactly one place.

Schema
------
dataset        nih | bbbc041 | mpidb | nihpoly
source_split   NIH: parasitized/uninfected. BBBC041: site_a/site_b (the two
               acquisition batches). MP-IDB: the species folder. nihpoly:
               <set>_<variant>, e.g. polygon_raw, polygon_masked, point_raw.
cell_id        unique within the project
path           image on disk, relative to the project root
label_binary   1 parasitised, 0 uninfected, <NA> where the source does not
               support a binary call (see eval_group)
eval_group     primary | excluded_difficult | excluded_leukocyte
stage          ring/trophozoite/schizont/gametocyte, or <NA>/unknown
species        MP-IDB only
patient_id     NIH and nihpoly; the C### prefix, used to group the train/val/test split
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
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm

from . import paths
from .background import host_cell_box
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
                    if not cv2.imwrite(str(out), box.apply(image)):
                        raise IOError(f"cv2.imwrite failed for {out}")
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


def _mpidb_stages(stem: str, n: int, shipped: dict):
    """Per-parasite stage labels for one image, and where they came from.

    Where the authors shipped crops, their expert per-parasite labels are carried
    across by left-to-right order. Otherwise a single-stage filename applies to
    every parasite in the image, and a multi-stage one cannot be resolved per
    parasite -- the README lists stages by first appearance, so R_S may be R, S, R.
    """
    fn_stages = _filename_stages(stem)
    if stem in shipped and len(shipped[stem]) == n:
        return shipped[stem], "shipped_crops"
    if len(fn_stages) == 1:
        return fn_stages * n, "filename_single_stage"
    return ["unknown"] * n, "ambiguous_multistage"


def build_mpidb_manifest(write_crops: bool = True,
                         skip_existing: bool = True):
    rows, audit = [], []
    for species in paths.MPIDB_SPECIES:
        shipped = _shipped_stage_map(species)
        for stem, img_path, gt_path in tqdm(list(_pair_img_gt(species)),
                                            desc=f"mpidb/{species}",
                                            leave=False):
            comps = mask_components(read_gray_mask(gt_path))
            stages, source = _mpidb_stages(stem, len(comps), shipped)

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
                    if not cv2.imwrite(str(out), box.apply(image)):
                        raise IOError(f"cv2.imwrite failed for {out}")
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


def build_mpidb_wholecell_manifest(write_crops: bool = True,
                                   skip_existing: bool = True):
    """MP-IDB cells framed on the host red blood cell instead of the parasite.

    The same parasites in the same order as build_mpidb_manifest, so stage labels
    and cell_id indices are identical; only the crop window differs
    (background.host_cell_box). Returns the manifest and a per-cell framing table.
    """
    rows, framing = [], []
    for species in paths.MPIDB_SPECIES:
        shipped = _shipped_stage_map(species)
        for stem, img_path, gt_path in tqdm(list(_pair_img_gt(species)),
                                            desc=f"mpidb_wholecell/{species}",
                                            leave=False):
            gt = read_gray_mask(gt_path)
            comps = mask_components(gt)
            stages, _ = _mpidb_stages(stem, len(comps), shipped)
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not read slide: {img_path}")
            if image.shape[:2] != gt.shape[:2]:
                raise ValueError(f"{img_path}: image and gt mask sizes differ")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            parasites = gt > 127
            for i, (r0, c0, r1, c1, _area) in enumerate(comps):
                enlarged = (species in paths.MPIDB_ENLARGING_SPECIES
                            or stages[i] == "gametocyte")
                limit = paths.MPIDB_HOST_LIMIT["enlarged" if enlarged else "normal"]
                box, outcome, host_side = host_cell_box(
                    rgb, parasites, (r0, c0, r1, c1), paths.MPIDB_CELL_SIDE,
                    paths.CROP_PAD_FRAC, max_cells=limit)
                out = (paths.CROPS_MPIDB_WHOLECELL / species / stages[i]
                       / f"{stem}_{i}.png")
                if write_crops and not (skip_existing and out.exists()):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(out), box.apply(image)):
                        raise IOError(f"cv2.imwrite failed for {out}")
                cell_id = f"{species}_{stem}_{i}"
                rows.append({
                    "dataset": "mpidb_wholecell", "source_split": species,
                    "cell_id": cell_id, "path": _rel(out), "label_binary": 1,
                    "eval_group": "primary", "stage": stages[i], "species": species,
                    "patient_id": pd.NA, "source_image": stem,
                    "r0": box.r0, "c0": box.c0, "r1": box.r1, "c1": box.c1,
                    "at_border": box.at_border, "pad_frac": paths.CROP_PAD_FRAC,
                })
                framing.append({"cell_id": cell_id, "species": species,
                                "stage": stages[i], "framing": outcome,
                                "host_limit_px": round(limit * paths.MPIDB_CELL_SIDE),
                                "host_side": host_side,
                                "parasite_side": max(r1 - r0, c1 - c0),
                                "crop_side": max(box.height, box.width)})
    return _frame(rows), pd.DataFrame(framing)


class MpidbParasiteSeeds:
    """The annotated parasite of an MP-IDB cell, in its crop's coordinates.

    Each MP-IDB crop is built around one connected component of the gt/ mask
    (framed on the parasite or on its host cell), so that component's pixels mark
    what the crop is of. Background removal is
    seeded with them, and scored on whether it keeps them. Masks are read one
    source image at a time, which matches the manifest's row order.
    """

    def __init__(self):
        self._gt = {(species, stem): gt for species in paths.MPIDB_SPECIES
                    for stem, _, gt in _pair_img_gt(species)}
        self._key = None
        self._mask = None
        self._comps = []

    def __call__(self, species, source_image, cell_id, window) -> np.ndarray:
        key = (str(species), str(source_image))
        if key != self._key:
            self._mask = read_gray_mask(self._gt[key])
            self._comps = mask_components(self._mask)
            self._key = key
        mask = self._mask
        i = int(str(cell_id).rsplit("_", 1)[1])
        r0, c0, r1, c1, _ = self._comps[i]
        w_r0, w_c0, w_r1, w_c1 = (int(v) for v in window)
        if not (w_r0 <= r0 and w_c0 <= c0 and r1 <= w_r1 and c1 <= w_c1):
            raise ValueError(f"parasite {i} of {cell_id} lies outside its crop window")
        seed = np.zeros(mask.shape[:2], bool)
        seed[r0:r1, c0:c1] = mask[r0:r1, c0:c1] > 127
        return seed[w_r0:w_r1, w_c0:w_c1]


# =========================================================================
# NIH-NLM-ThinBloodSmearsPf: recutting the NIH cells from the photographs
# =========================================================================
# cell_images ships its cells already segmented onto black, with no record of
# where they sat in the photograph, so the training side of the crop-format
# difference could not be controlled. The full release has an expert outline
# (Polygon Set) or centre point (Point Set) for every cell, which lets the NIH
# cells be cut with exactly the rule the test sets are cut with.
#
# Polygon Set cells are written in two variants:
#   raw     the square window padded by CROP_PAD_FRAC, background kept -- the
#           test-set format
#   masked  the square window cut tight to the outline (no padding), with
#           everything outside the outline set to 0 -- the cell_images format.
#           cell_images crops are tight to the cell (median side 130 px, black
#           fraction 0.26); under the padded rule the black fraction doubles to
#           0.52, which would make the recut cells a different format again.
# plus the outline itself, in the raw window, as a binary PNG under mask/: the
# ground truth the background-removal methods are scored against.
#
# Point Set cells have no outline, so only a raw variant exists, boxed at
# paths.NIHPOLY_POINT_BOX around the point.
NIHPOLY_LABELS = {"Parasitized": (1, "primary", "parasitized"),
                  "Uninfected": (0, "primary", "uninfected"),
                  "White_Blood_Cell": (pd.NA, "excluded_leukocyte", "leukocyte")}
NIHPOLY_VARIANTS = {"polygon": ("raw", "masked"), "point": ("raw",)}
_NIHPOLY_FOLDER = re.compile(r"^\d+(C\d+)")


def nihpoly_mask_path(crop_path) -> str:
    """The outline mask belonging to a Polygon Set crop of either variant."""
    parts = str(crop_path).replace("\\", "/").split("/")
    i = parts.index("polygon")          # .../nihpoly/polygon/<variant>/<label>/<file>
    parts[i + 1] = "mask"
    return "/".join(parts)


def read_nihpoly_gt(gt_path):
    """(width, height, cells) for one annotation file.

    `cells` is a list of (index, label, shape, xy) with xy an (n, 2) array of
    x, y image coordinates: the outline for a Polygon, one row for a Point.
    """
    lines = [l for l in Path(gt_path).read_text(errors="ignore").splitlines()
             if l.strip()]
    n, width, height = (int(float(v)) for v in lines[0].split(",")[:3])
    cells = []
    for k, line in enumerate(lines[1:]):
        t = [v.strip() for v in line.split(",")]
        npts = int(t[4])
        xy = np.asarray(t[5:5 + 2 * npts], dtype=float).reshape(-1, 2)
        cells.append((k, t[1], t[3], xy))
    if len(cells) != n:
        raise ValueError(f"{gt_path}: header says {n} cells, file has {len(cells)}")
    return width, height, cells


def _polygon_mask(xy: np.ndarray, box) -> np.ndarray:
    mask = np.zeros((box.height, box.width), np.uint8)
    cv2.fillPoly(mask, [np.round(xy - [box.c0, box.r0]).astype(np.int32)], 255)
    return mask


def _nihpoly_window(shape: str, xy: np.ndarray, h: int, w: int,
                    pad_frac: float = paths.CROP_PAD_FRAC):
    if shape == "Polygon":
        c0, r0 = np.floor(xy.min(axis=0)).astype(int)
        c1, r1 = np.ceil(xy.max(axis=0)).astype(int) + 1   # half-open
    elif shape == "Point":
        (cx, cy), half = xy[0], paths.NIHPOLY_POINT_BOX / 2.0
        r0, r1 = int(round(cy - half)), int(round(cy + half))
        c0, c1 = int(round(cx - half)), int(round(cx + half))
    else:
        raise ValueError(f"unknown annotation shape: {shape!r}")
    return square_padded_box(int(r0), int(c0), int(r1), int(c1), h, w,
                             pad_frac=pad_frac)


def build_nihpoly_manifest(sets=("polygon",), write_crops: bool = True,
                           skip_existing: bool = True) -> pd.DataFrame:
    rows = []
    for set_name in sets:
        variants = NIHPOLY_VARIANTS[set_name]
        folders = [f for f in sorted(paths.NIHPOLY_SETS[set_name].iterdir())
                   if f.is_dir()]
        for folder in tqdm(folders, desc=f"nihpoly/{set_name}", leave=False):
            m = _NIHPOLY_FOLDER.match(folder.name)
            if m is None:
                raise ValueError(f"unexpected patient folder: {folder.name}")
            patient = m.group(1)
            for gt in sorted((folder / "GT").glob("*.txt")):
                img_path = folder / "Img" / f"{gt.stem}.jpg"
                if not img_path.exists():
                    raise FileNotFoundError(f"no photograph for {gt}")
                width, height, cells = read_nihpoly_gt(gt)

                planned = []
                for k, label_name, shape, xy in cells:
                    label, group, label_dir = NIHPOLY_LABELS[label_name]
                    boxes = {"raw": _nihpoly_window(shape, xy, height, width)}
                    if "masked" in variants:
                        boxes["masked"] = _nihpoly_window(shape, xy, height, width,
                                                          pad_frac=0.0)
                    name = f"{folder.name}_{gt.stem}_{k}.png"
                    outs = {v: paths.CROPS_NIHPOLY / set_name / v / label_dir / name
                            for v in variants}
                    if "masked" in variants:
                        outs["mask"] = (paths.CROPS_NIHPOLY / set_name / "mask"
                                        / label_dir / name)
                    planned.append((k, label, group, xy, boxes, outs))

                image = None
                if write_crops and any(not (skip_existing and o.exists())
                                       for p in planned for o in p[5].values()):
                    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError(f"could not read photograph: {img_path}")
                    if image.shape[:2] != (height, width):
                        raise ValueError(f"{img_path}: size {image.shape[:2]} does "
                                         f"not match annotation {(height, width)}")

                for k, label, group, xy, boxes, outs in planned:
                    if image is not None and any(not (skip_existing and o.exists())
                                                 for o in outs.values()):
                        written = {"raw": boxes["raw"].apply(image)}
                        if "masked" in boxes:
                            written["mask"] = _polygon_mask(xy, boxes["raw"])
                            tight = boxes["masked"].apply(image).copy()
                            tight[_polygon_mask(xy, boxes["masked"]) == 0] = 0
                            written["masked"] = tight
                        for v, out in outs.items():
                            if skip_existing and out.exists():
                                continue
                            out.parent.mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(out), written[v]):
                                raise IOError(f"cv2.imwrite failed for {out}")
                    for v in variants:
                        box = boxes[v]
                        rows.append({
                            "dataset": "nihpoly", "source_split": f"{set_name}_{v}",
                            "cell_id": f"{set_name}_{v}_{folder.name}_{gt.stem}_{k}",
                            "path": _rel(outs[v]), "label_binary": label,
                            "eval_group": group, "stage": pd.NA, "species": pd.NA,
                            "patient_id": patient,
                            "source_image": f"{folder.name}_{gt.stem}",
                            "r0": box.r0, "c0": box.c0, "r1": box.r1, "c1": box.c1,
                            "at_border": box.at_border,
                            "pad_frac": paths.CROP_PAD_FRAC if v == "raw" else 0.0,
                        })
    return _frame(rows)


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
