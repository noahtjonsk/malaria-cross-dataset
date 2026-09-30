"""Fine-tune one ImageNet-pretrained CNN on the NIH training patients.

    python scripts/train.py --arch vgg16 --seed 0                 # the baseline (Colab GPU)
    python scripts/train.py --arch vgg16 --seed 0 --resume        # continue after a disconnect
    python scripts/train.py --arch mobilenet_v2 --smoke           # 2 tiny batches on CPU

Every architecture gets the same schedule, fixed in advance rather than tuned,
so that RQ1 can attribute differences to the architecture:

    loss        binary cross-entropy on one logit (BCEWithLogitsLoss)
    optimiser   Adam, learning rate 1e-4, all layers trainable
    batch       32
    epochs      at most 20; early stopping when NIH validation loss has not
                improved for 3 epochs, keeping the best-validation weights
    input       black-padded square, 224 px, ImageNet normalisation
    augment     flips and 90-degree rotations only (malaria/data.py says why)

NIH is balanced (9,619 of 19,203 training cells parasitised), so the classes are
not resampled or reweighted. Only the NIH train and validation patients are
read here; the NIH hold-out patients and all external test sets are touched only
by evaluate.py, after training is finished.

Writes to --out (default models/):
    <arch>_s<seed>.pt          best-validation weights plus the settings used
    <arch>_s<seed>_last.pt     full state after each epoch, for --resume
    <arch>_s<seed>_log.json    per-epoch losses and validation metrics
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from malaria import paths  # noqa: E402
from malaria.data import CellDataset, nih_cells  # noqa: E402
from malaria.metrics import THRESHOLD  # noqa: E402
from malaria.models import ARCHS, build_model, check_all_trainable, count_parameters  # noqa: E402

SETTINGS = {"lr": 1e-4, "batch_size": 32, "max_epochs": 20, "patience": 3,
            "weight_decay": 0.0, "input": paths.MODEL_INPUT,
            "augment": "hflip, vflip, rot90", "threshold": THRESHOLD}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(net, loader, device, loss_fn, optimiser=None, scaler=None):
    """One pass; trains when an optimiser is given. Returns loss and val metrics."""
    train = optimiser is not None
    net.train(train)
    total, n, tp, fn, tn, fp = 0.0, 0, 0, 0, 0, 0
    amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logit = net(x).squeeze(1)
                loss = loss_fn(logit, y)
            if train:
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            pred = (torch.sigmoid(logit.float()) >= THRESHOLD).long()
            y = y.long()
            tp += int(((pred == 1) & (y == 1)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
    return {"loss": total / max(n, 1),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / max(n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=paths.MODELS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="64 training and 32 validation cells, one epoch: checks the code runs")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random initialisation (only to test without downloading weights)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = dict(SETTINGS, arch=args.arch, seed=args.seed,
                    pretrained=not args.no_pretrained, smoke=args.smoke)

    train_cells, val_cells = nih_cells("train"), nih_cells("val")
    if args.smoke:
        train_cells = train_cells.groupby("label_binary").sample(32, random_state=args.seed)
        val_cells = val_cells.groupby("label_binary").sample(16, random_state=args.seed)
        settings["max_epochs"] = 1
    if set(train_cells["patient_id"]) & set(val_cells["patient_id"]):
        raise AssertionError("a patient is in both train and val")
    print(f"{args.arch} seed {args.seed} on {device}: {len(train_cells):,} train, "
          f"{len(val_cells):,} val cells")

    # The shuffle order and every worker's augmentation seed are drawn from this
    # one generator at the start of each epoch. Workers are not persistent, so
    # saving its state after an epoch is enough for --resume to continue
    # exactly as an uninterrupted run would.
    loader_rng = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(CellDataset(train_cells, train=True), shuffle=True,
                            batch_size=settings["batch_size"], num_workers=args.workers,
                            pin_memory=device.type == "cuda", drop_last=False,
                            generator=loader_rng),
        "val": DataLoader(CellDataset(val_cells), batch_size=2 * settings["batch_size"],
                          num_workers=args.workers, pin_memory=device.type == "cuda"),
    }

    net = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=settings["lr"],
                                 weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arch}_s{args.seed}" + ("_smoke" if args.smoke else "")
    best_path, last_path = args.out / f"{tag}.pt", args.out / f"{tag}_last.pt"
    log_path = args.out / f"{tag}_log.json"
    state = {"epoch": 0, "best_val_loss": float("inf"), "bad_epochs": 0, "history": []}
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimiser"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"])
        if "loader_rng" in ck:   # checkpoints from before this was saved lack it
            loader_rng.set_state(ck["loader_rng"])
        if ck.get("cuda_rng") and device.type == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        state = ck["state"]
        print(f"resumed after epoch {state['epoch']}")

    print(f"{count_parameters(net):,} parameters, all trainable")
    while state["epoch"] < settings["max_epochs"] and state["bad_epochs"] < settings["patience"]:
        t0 = time.time()
        tr = run_epoch(net, loaders["train"], device, loss_fn, optimiser, scaler)
        va = run_epoch(net, loaders["val"], device, loss_fn)
        state["epoch"] += 1
        improved = va["loss"] < state["best_val_loss"]
        if improved:
            state["best_val_loss"], state["bad_epochs"] = va["loss"], 0
            torch.save({"model": net.state_dict(), "settings": settings,
                        "epoch": state["epoch"], "val": va}, best_path)
        else:
            state["bad_epochs"] += 1
        state["history"].append({"epoch": state["epoch"], "train": tr, "val": va,
                                 "seconds": round(time.time() - t0, 1), "best": improved})
        torch.save({"model": net.state_dict(), "optimiser": optimiser.state_dict(),
                    "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "loader_rng": loader_rng.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "state": state, "settings": settings}, last_path)
        log_path.write_text(json.dumps({"settings": settings, **state}, indent=1))
        print(f"epoch {state['epoch']:2d}  train loss {tr['loss']:.4f}  val loss {va['loss']:.4f}"
              f"  val sens {va['sensitivity']:.3f}  spec {va['specificity']:.3f}"
              f"  {'*' if improved else ' '}  {time.time() - t0:.0f}s")

    check_all_trainable(net)
    stop = "early stopping" if state["bad_epochs"] >= settings["patience"] else "max epochs"
    print(f"done ({stop}); best val loss {state['best_val_loss']:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
