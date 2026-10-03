"""Which code and which input files produced a checkpoint or a prediction file.

Colab runs from an unzipped code.zip without a .git folder, so pack_for_colab.py
writes the commit into each zip as PROVENANCE_<zip>.txt and git_commit() falls
back to the code one. Manifests list crop paths, not pixels (the colour crops
were rebuilt in place on 30 Sept), so the data zips also carry a digest of every
crop's contents per variant; evaluate.py records both, and compare_models.py
refuses to compare models scored on different manifests or crops.
"""
from __future__ import annotations

import hashlib
import subprocess

from . import paths

PROVENANCE_GLOB = "PROVENANCE_*.txt"
PROVENANCE_CODE = paths.ROOT / "PROVENANCE_code.txt"


def git_commit() -> str:
    """Current commit (with '+dirty' for uncommitted changes), or the packed one."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        # Dirty: uncommitted changes to tracked files, or untracked code that
        # pack_for_colab.py would still pack (malaria/*.py, scripts/*.py).
        status = [["git", "status", "--porcelain", "--untracked-files=no"],
                  ["git", "status", "--porcelain", "--untracked-files=all", "--",
                   "malaria", "scripts"]]
        dirty = any(subprocess.run(cmd, cwd=paths.ROOT, capture_output=True, text=True,
                                   check=True).stdout.strip() for cmd in status)
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        if PROVENANCE_CODE.exists():
            for line in PROVENANCE_CODE.read_text().splitlines():
                if line.startswith("commit "):
                    return line.split(" ", 1)[1]
        return "unknown"


def crop_digest(rel_paths, known: dict | None = None) -> str:
    """One hash over the contents of every crop (sorted paths), 12 hex digits.
    `known` maps paths to full SHA-256 hex digests already computed."""
    known = known or {}
    h = hashlib.sha256()
    for p in sorted(set(rel_paths)):
        h.update(p.encode())
        h.update(bytes.fromhex(known.get(p) or file_hash_full(paths.ROOT / p)))
    return h.hexdigest()[:12]


def packed_crop_digests() -> dict:
    """{variant: crop digest} from the PROVENANCE files of the unzipped data zips;
    empty when running on the laptop's own crops."""
    out = {}
    for f in paths.ROOT.glob(PROVENANCE_GLOB):
        for line in f.read_text().splitlines():
            if line.startswith("crops "):
                _, variant, digest = line.split()
                out[variant] = digest
    return out


def file_hash_full(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_hash(path) -> str:
    """First 12 hex digits of a file's SHA-256."""
    return file_hash_full(path)[:12]


def manifest_hashes(names) -> dict:
    """{manifest file name: hash} for manifests under data/manifests."""
    return {n: file_hash(paths.MANIFESTS / n) for n in sorted(set(names))}
