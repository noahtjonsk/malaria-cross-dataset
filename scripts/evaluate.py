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
columns plus `prob`, one row per cell. A prediction file that already exists is
kept (so a disconnected run resumes where it stopped) unless --force is given.
run.json in the same folder records the checkpoint settings, and for each
variant the code commit and the hashes of the manifests its cells came from.
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
    info = json.loads(info_path.read_text()) if info_path.exists() else {"variants": {}}
    info.update(checkpoint=args.checkpoint.name, settings=s, best_epoch=ck["epoch"],
                val=ck["val"], smoke_eval=args.smoke)
    for variant in args.variants:
        jobs = ([("nih_test", lambda: nih_cells("test"))] if variant == "raw" else [])
        jobs += [(d, lambda d=d: test_cells(d, variant)) for d in TEST_DATASETS]
        ran = False
        for name, cells_fn in jobs:
            dst = out_dir / f"{variant}_{name}.csv"
            if dst.exists() and not args.force:
                print(f"  {variant:9s} {name:16s} exists, kept (--force to redo)")
                continue
            cells = cells_fn()
            if args.smoke:
                cells = cells.sample(min(64, len(cells)), random_state=0)
            cells = cells.reset_index(drop=True).copy()
            cells["prob"] = predict(net, cells, device, args.batch_size, args.workers)
            cells.to_csv(dst, index=False)
            ran = True
            called = (cells["prob"] >= THRESHOLD).mean()
            print(f"  {variant:9s} {name:16s} {len(cells):6,} cells  "
                  f"{100 * called:5.1f}% called parasitised -> {dst.name}")
        if ran:
            info["variants"][variant] = {
                "commit": provenance.git_commit(),
                "manifests": provenance.manifest_hashes(variant_manifest_files(variant)),
                # from the unzipped data zip on Colab; the laptop's own crops have none
                "crops": provenance.packed_crop_digests().get(variant, "local")}
        elif variant not in info["variants"]:
            print(f"  warning: {variant} predictions predate run.json; "
                  "rerun with --force to record which crops they came from")
    info_path.write_text(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
