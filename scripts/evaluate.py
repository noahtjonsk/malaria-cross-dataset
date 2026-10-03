"""Run one trained, frozen model over the NIH hold-out set and the test sets.

    python scripts/evaluate.py --checkpoint models/vgg16_s0.pt                  # RQ1: raw crops
    python scripts/evaluate.py --checkpoint models/vgg16_s0.pt --variants raw masked reinhard histmatch
    python scripts/evaluate.py --checkpoint models/vgg16_s0_smoke.pt --smoke    # 64 cells per set

The model is never updated here (eval mode, no gradients) and the decision
threshold is not touched: predictions are stored as probabilities and every
metric applies the fixed 0.5 rule afterwards (malaria/metrics.py).

Sets
    nih_test          the NIH hold-out patients (raw only: NIH is the reference format)
    bbbc041           site_a and site_b, primary cells (difficult and leukocyte excluded)
    mpidb_wholecell   every species, whole-cell crops (parasitised only)

Variants (RQ2 steps, applied to the external sets only)
    raw         the crop as cut from the source photograph
    masked      background removed (build_masked_crops.py, Otsu on brightness)
    reinhard    then colour-matched to NIH train cells, Reinhard (build_colour_crops.py)
    histmatch   then colour-matched by histogram matching

Writes outputs/predictions/<arch>_s<seed>/<variant>_<set>.csv: the manifest
columns plus `prob`, one row per cell. run.json in the same folder records, for
every prediction file, the checkpoint's hash, the manifest hashes, a digest of
the contents of the crops it scored, the cell count and the code commit. A file
is kept on a rerun (so a disconnected run resumes where it stopped) only if all
of these still match; otherwise, or with --force, it is redone. Files are
written to a temporary name and renamed, so a disconnect never leaves half a file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from malaria import paths, provenance  # noqa: E402
from malaria.data import (VARIANTS, CellDataset, TEST_DATASETS, nih_cells,  # noqa: E402
                          test_cells, variant_manifest_files)
from malaria.metrics import THRESHOLD  # noqa: E402
from malaria.models import build_model  # noqa: E402


@torch.no_grad()
def predict(net, cells: pd.DataFrame, device, batch_size: int, workers: int) -> np.ndarray:
    loader = DataLoader(CellDataset(cells), batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda")
    prob = np.empty(len(cells), dtype=np.float32)
    for x, _, idx in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit = net(x.to(device, non_blocking=True)).squeeze(1)
        prob[idx.numpy()] = torch.sigmoid(logit.float()).cpu().numpy()
    return prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["raw"], choices=VARIANTS)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="64 cells per set")
    ap.add_argument("--force", action="store_true", help="redo prediction files that exist")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    s = ck["settings"]
    net = build_model(s["arch"], pretrained=False)
    net.load_state_dict(ck["model"])
    net.to(device).eval()
    tag = paths.run_tag(s["arch"], s["seed"], args.smoke or s.get("smoke", False))
    out_dir = paths.PREDICTIONS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{tag}: best epoch {ck['epoch']}, val loss {ck['val']['loss']:.4f}, on {device}")

    info_path = out_dir / "run.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.setdefault("files", {})
    info.update(checkpoint=args.checkpoint.name, settings=s, best_epoch=ck["epoch"],
                val=ck["val"])
    ck_sha = provenance.file_hash(args.checkpoint)
    for variant in args.variants:
        manifests = provenance.manifest_hashes(variant_manifest_files(variant))
        jobs = ([("nih_test", lambda: nih_cells("test"))] if variant == "raw" else [])
        jobs += [(d, lambda d=d: test_cells(d, variant)) for d in TEST_DATASETS]
        for name, cells_fn in jobs:
            dst = out_dir / f"{variant}_{name}.csv"
            cells = cells_fn()
            if args.smoke:
                cells = cells.sample(min(64, len(cells)), random_state=0)
            cells = cells.reset_index(drop=True).copy()
            # What this file must have come from: this checkpoint, these
            # manifests and the contents of exactly these crops.
            record = {"checkpoint_sha": ck_sha, "best_epoch": ck["epoch"],
                      "manifests": manifests, "crops": provenance.crop_digest(cells["path"]),
                      "n_cells": len(cells), "smoke": args.smoke}
            old = info["files"].get(dst.name, {})
            same = (all(old.get(k) == v for k, v in record.items())
                    and dst.exists() and csv_rows(dst) == len(cells))
            if same and not args.force:
                print(f"  {variant:9s} {name:16s} up to date, kept (--force to redo)")
                continue
            cells["prob"] = predict(net, cells, device, args.batch_size, args.workers)
            tmp = dst.with_name(dst.name + ".tmp")     # a disconnect leaves no half file
            cells.to_csv(tmp, index=False)
            tmp.replace(dst)
            info["files"][dst.name] = {**record, "commit": provenance.git_commit()}
            write_json_atomic(info_path, info)
            called = (cells["prob"] >= THRESHOLD).mean()
            print(f"  {variant:9s} {name:16s} {len(cells):6,} cells  "
                  f"{100 * called:5.1f}% called parasitised -> {dst.name}")
    write_json_atomic(info_path, info)


def csv_rows(path: Path) -> int:
    """Data rows in a prediction CSV (no field holds a line break)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


if __name__ == "__main__":
    main()
