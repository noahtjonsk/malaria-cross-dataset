"""Every path the project uses, resolved from this file's location.

Edit here and nowhere else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Source datasets (read-only) ---
NIH_DIR = ROOT / "cell_images_NIH" / "cell_images"
NIH_PARASITIZED = NIH_DIR / "Parasitized"
NIH_UNINFECTED = NIH_DIR / "Uninfected"

BBBC041_DIR = ROOT / "malaria_BBBC041" / "malaria"
BBBC041_IMAGES = BBBC041_DIR / "images"
# BBBC041 ships two annotation files that are two distinct acquisition batches:
# training.json is 1208 images, all 1200x1600 PNG; test.json is 120 images, all
# 1383x1944 JPG. The split is named by that observable, not by country -- the
# source publication lists Brazilian and South-East Asian material but does not
# state which file is which, so the neutral label is what goes in the thesis.
BBBC041_JSON = {"site_a": BBBC041_DIR / "training.json",
                "site_b": BBBC041_DIR / "test.json"}

_MPIDB_OUTER = ROOT / ("MP-IDB-The-Malaria-Parasite-Image-Database-"
                       "for-Image-Processing-and-Analysis-master")
MPIDB_DIR = _MPIDB_OUTER / _MPIDB_OUTER.name
MPIDB_SPECIES = ["Falciparum", "Malariae", "Ovale", "Vivax"]

# NIH-NLM-ThinBloodSmearsPf (Kassim et al., 2020): the full 5312x2988 field
# photographs behind cell_images_NIH, with expert annotations. It is the same
# acquisition, not a new dataset: 192 of the 200 cell_images patients and 960 of
# its 965 photographs are the ones those cells were cut from, and per-photo
# parasitised counts agree (see scripts/check_nih_overlap.py). It must never be
# used as an external test set, and every cell recut from it inherits the
# patient split in nih_split.csv. "Polygon Set" outlines every cell (33 patients,
# 165 photos); "Point Set" marks each cell with a single point (160 patients,
# 800 photos).
NIHPOLY_DIR = ROOT / "NIH-NLM-ThinBloodSmearsPf"
NIHPOLY_SETS = {"polygon": NIHPOLY_DIR / "Polygon Set",
                "point": NIHPOLY_DIR / "Point Set"}

# --- Generated artefacts (written by this project) ---
DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"
CROPS = DATA / "crops"
CROPS_BBBC041 = CROPS / "bbbc041"
CROPS_MPIDB = CROPS / "mpidb"
CROPS_NIHPOLY = CROPS / "nihpoly"
CROPS_MPIDB_WHOLECELL = CROPS / "mpidb_wholecell"
CROPS_MASKED = CROPS / "masked"        # <method>/<path under data/crops>, RQ2 step 1
CROPS_COLOUR = CROPS / "colour"        # <method>/<path under the masked crops>, RQ2 step 2

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
PREDICTIONS = OUTPUTS / "predictions"  # <arch>_s<seed>/<variant>_<set>.csv

MODELS = ROOT / "models"               # trained checkpoints and training logs
COLAB = ROOT / "colab"                 # zips uploaded to Colab (pack_for_colab.py)
MO_FIGURES = ROOT / "docs" / "methodology" / "figures"

# --- Conventions shared across the project ---
MODEL_INPUT = 224      # every statistic that depends on scale is computed at this size
CROP_PAD_FRAC = 0.10   # see crops.py; recorded in every manifest
MIN_MASK_AREA = 50     # MP-IDB connected components smaller than this are noise
# Point Set cells carry a centre point but no outline, so they are boxed at the
# median longer side of the 34,213 hand-drawn Polygon Set red-cell outlines
# (135.1 px; parasitised 144.2, uninfected 134.8). All photographs share one
# camera and 5312x2988 frame, so one size fits the whole release.
NIHPOLY_POINT_BOX = 135
# MP-IDB annotates parasites, not the red cells hosting them. Red cells in its
# 2592x1944 images measure 130-145 px across (median longer side of Otsu
# components on 16 images, all four species), so host cells are searched for
# at that scale (background.host_cell_box).
MPIDB_CELL_SIDE = 140
# How wide a host-cell region may be, in MPIDB_CELL_SIDE units, before it is
# taken for touching cells the watershed failed to split. P. vivax and P. ovale
# enlarge the red cells they infect and gametocytes are elongated, so those get
# the wider limit. Set by inspecting the largest crops per species: Falciparum
# regions above ~200 px were two or more packed cells, while Vivax and Ovale
# regions up to 269 px were single enlarged cells.
MPIDB_ENLARGING_SPECIES = ("Vivax", "Ovale")
MPIDB_HOST_LIMIT = {"normal": 1.4, "enlarged": 2.0}

STAGE_NAMES = {"R": "ring", "T": "trophozoite", "S": "schizont", "G": "gametocyte"}


def ensure_dirs() -> None:
    """Create every generated-artefact directory. Safe to call repeatedly."""
    for d in (MANIFESTS, CROPS_BBBC041, CROPS_MPIDB, CROPS_NIHPOLY,
              CROPS_MPIDB_WHOLECELL, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
