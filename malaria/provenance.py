"""Which code and which input files produced a checkpoint or a prediction file.

Colab runs from an unzipped code.zip without a .git folder, so pack_for_colab.py
writes the commit into the zip as PROVENANCE.txt and git_commit() falls back to
it. Manifest hashes let compare_models.py refuse to compare models that were
scored on different crops.
"""
from __future__ import annotations

import hashlib
import subprocess

from . import paths

PROVENANCE_FILE = paths.ROOT / "PROVENANCE.txt"


def git_commit() -> str:
    """Current commit (with '+dirty' for uncommitted changes), or the packed one."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=paths.ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        if PROVENANCE_FILE.exists():
            for line in PROVENANCE_FILE.read_text().splitlines():
                if line.startswith("commit "):
                    return line.split(" ", 1)[1]
        return "unknown"


def file_hash(path) -> str:
    """First 12 hex digits of a file's SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def manifest_hashes(names) -> dict:
    """{manifest file name: hash} for manifests under data/manifests."""
    return {n: file_hash(paths.MANIFESTS / n) for n in sorted(set(names))}
