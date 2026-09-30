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
columns plus `prob`, one row per cell.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.data import CellDataset, TEST_DATASETS, nih_cells, test_cells  # noqa: E402
from malaria.metrics import THRESHOLD  # noqa: E402
from malaria.models import build_model  # noqa: E402

PREDICTIONS = paths.PREDICTIONS


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
    ap.add_argument("--variants", nargs="+", default=["raw"],
                    choices=["raw", "masked", "reinhard", "histmatch"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="64 cells per set")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    s = ck["settings"]
    net = build_model(s["arch"], pretrained=False)
    net.load_state_dict(ck["model"])
    net.to(device).eval()
    tag = f"{s['arch']}_s{s['seed']}" + ("_smoke" if args.smoke or s.get("smoke") else "")
    out_dir = PREDICTIONS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{tag}: best epoch {ck['epoch']}, val loss {ck['val']['loss']:.4f}, on {device}")

    jobs = [("raw", "nih_test", nih_cells("test"))] if "raw" in args.variants else []
    jobs += [(v, d, test_cells(d, v)) for v in args.variants for d in TEST_DATASETS]
    for variant, name, cells in jobs:
        if args.smoke:
            cells = cells.sample(min(64, len(cells)), random_state=0).reset_index(drop=True)
        cells = cells.copy()
        cells["prob"] = predict(net, cells, device, args.batch_size, args.workers)
        dst = out_dir / f"{variant}_{name}.csv"
        cells.to_csv(dst, index=False)
        called = (cells["prob"] >= THRESHOLD).mean()
        print(f"  {variant:9s} {name:16s} {len(cells):6,} cells  "
              f"{100 * called:5.1f}% called parasitised -> {dst.name}")


if __name__ == "__main__":
    main()
