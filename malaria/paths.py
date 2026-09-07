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

# --- Generated artefacts (written by this project) ---
DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"
CROPS = DATA / "crops"
CROPS_BBBC041 = CROPS / "bbbc041"
CROPS_MPIDB = CROPS / "mpidb"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

# --- Conventions shared across the project ---
MODEL_INPUT = 224      # every statistic that depends on scale is computed at this size
CROP_PAD_FRAC = 0.10   # see crops.py; recorded in every manifest
MIN_MASK_AREA = 50     # MP-IDB connected components smaller than this are noise

STAGE_NAMES = {"R": "ring", "T": "trophozoite", "S": "schizont", "G": "gametocyte"}


def ensure_dirs() -> None:
    """Create every generated-artefact directory. Safe to call repeatedly."""
    for d in (MANIFESTS, CROPS_BBBC041, CROPS_MPIDB, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
