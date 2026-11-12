"""Train a decoder variant on cached frozen-CLIP features.

Examples:
    pixi run train --fusion film                       # M1
    pixi run train --fusion none                       # M0 image-only
    pixi run train --fusion concat                     # M1b
    pixi run train --fusion film --overfit 16          # sanity check: memorise 16 samples
    pixi run train --fusion film --no-coords --feat-drop 0.1 --drop2d 0.1 --weight-decay 0.05
    pixi run train --fusion film --no-coords --flip    # needs `pixi run cache-features --flip`

Outputs go to outputs/runs/<run-name>/:
    config.json, history.csv, curves.png, best.pt (by val mIoU), last.pt, summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.refcoco_m import PROJECT_ROOT, load_split
from src.features import DEFAULT_CACHE, CachedSplit, load_meta
from src.losses import seg_loss
from src.metrics import iou_dice
from src.models.decoder import FUSIONS, TextSegDecoder, count_trainable

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


def make_run_name(args) -> str:
    """Deterministic run name; regularisation suffixes appear only when non-default."""
    name = f"{args.fusion}_c{args.channels}{'_nocoord' if args.no_coords else ''}_dice{args.dice_weight:g}"
    if args.feat_drop:
        name += f"_fd{args.feat_drop:g}"
    if args.drop2d:
        name += f"_dd{args.drop2d:g}"
    if args.weight_decay != 1e-2:
        name += f"_wd{args.weight_decay:g}"
    if args.flip:
        name += "_flip"
    name += f"_s{args.seed}"
    return name + (f"_overfit{args.overfit}" if args.overfit else "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fusion", choices=FUSIONS, default="film")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dice-weight", type=float, default=1.0, help="0 = BCE only")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--no-coords", action="store_true", help="disable (x, y) coordinate channels")
    ap.add_argument("--feat-drop", type=float, default=0.0, help="dropout on frozen CLIP features")
    ap.add_argument("--drop2d", type=float, default=0.0, help="channel dropout inside the decoder")
    ap.add_argument("--flip", action="store_true",
                    help="random horizontal flip (p=0.5) with left<->right swapped text (train only)")
    ap.add_argument("--patience", type=int, default=8, help="early stop after N epochs w/o val gain (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="train AND evaluate on N train samples from distinct images (sanity check)")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--quiet", action="store_true", help="progress bars only, no per-epoch lines")
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, data: CachedSplit, indices: torch.Tensor, batch_size: int, desc: str):
    model.eval()
    ious, dices, losses = [], [], []
    for a in tqdm(range(0, len(indices), batch_size), desc=desc, leave=False, unit="batch"):
        b = data.batch(indices[a:a + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["feats"], b["text"])
        logits = logits.float()
        losses.append(seg_loss(logits, b["mask"]).item() * len(b["idx"]))
        iou, dice = iou_dice(logits, b["mask"])
        ious.append(iou)
        dices.append(dice)
    iou, dice = torch.cat(ious), torch.cat(dices)
    return {
        "loss": sum(losses) / len(indices),
        "mIoU": iou.mean().item(),
        "dice": dice.mean().item(),
        "P@0.5": (iou > 0.5).float().mean().item(),
    }


def one_per_image(data: CachedSplit, n: int) -> torch.Tensor:
    seen, picks = set(), []
    for i, s in enumerate(data.samples):
        if s["image_id"] not in seen:
            seen.add(s["image_id"])
            picks.append(i)
        if len(picks) == n:
            break
    return torch.tensor(picks)


def plot_curves(history: list[dict], path: Path, title: str) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("BCE + Dice loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, [h["train_mIoU"] for h in history], label="train mIoU")
    a2.plot(ep, [h["val_mIoU"] for h in history], label="val mIoU")
    a2.plot(ep, [h["val_dice"] for h in history], label="val Dice", ls="--")
    a2.set_xlabel("epoch"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA is required"

    name = args.run_name or make_run_name(args)
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    split = load_split()
    meta = load_meta(args.cache)
    if not args.quiet:
        print(f"loading cached features to GPU from {args.cache.name} ...")
    train = CachedSplit("train", split, args.cache, flip=args.flip)
    if args.overfit:
        train_idx = one_per_image(train, args.overfit)
        val, val_idx = train, train_idx
        args.patience = 0
    else:
        val = CachedSplit("val", split, args.cache)
        train_idx, val_idx = torch.arange(len(train)), torch.arange(len(val))

    model = TextSegDecoder(fusion=args.fusion, n_layers=len(meta["layers"]),
                           channels=args.channels, coords=not args.no_coords,
                           feat_drop=args.feat_drop, drop2d=args.drop2d).cuda()
    n_params = count_trainable(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_idx) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)

    config = {**vars(args), "cache": str(args.cache), "run_name": name,
              "trainable_params": n_params, "train_samples": len(train_idx),
              "val_samples": len(val_idx), "layers": meta["layers"]}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    if not args.quiet:
        print(f"run {name}: fusion={args.fusion} params={n_params:,} "
          f"train={len(train_idx)} val={len(val_idx)} samples")

    gen = torch.Generator().manual_seed(args.seed)
    history, best, best_epoch, bad = [], -1.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    epochs = tqdm(range(1, args.epochs + 1), desc=f"{name}", unit="epoch", leave=not args.quiet)
    for epoch in epochs:
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=gen)]
        running, seen, tr_iou = 0.0, 0, []
        steps = tqdm(range(0, len(order), args.batch_size), desc=f"  epoch {epoch}",
                     leave=False, unit="batch")
        for a in steps:
            idx = order[a:a + args.batch_size]
            flip = (torch.rand(len(idx), generator=gen) < 0.5) if args.flip else None
            b = train.batch(idx, flip)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["feats"], b["text"])
            loss = seg_loss(logits.float(), b["mask"], args.dice_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            n = len(b["idx"])
            running += loss.item() * n
            seen += n
            tr_iou.append(iou_dice(logits.detach().float(), b["mask"])[0])
            steps.set_postfix(loss=f"{running / seen:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        v = evaluate(model, val, val_idx, 64, desc="  val")
        row = {"epoch": epoch, "train_loss": running / seen,
               "train_mIoU": torch.cat(tr_iou).mean().item(),
               "val_loss": v["loss"], "val_mIoU": v["mIoU"], "val_dice": v["dice"],
               "val_P@0.5": v["P@0.5"], "lr": sched.get_last_lr()[0]}
        history.append(row)

        if v["mIoU"] > best:
            best, best_epoch, bad = v["mIoU"], epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val": v}, out / "best.pt")
        else:
            bad += 1
        epochs.set_postfix(train_loss=f"{row['train_loss']:.3f}", val_mIoU=f"{v['mIoU']:.3f}",
                           best=f"{best:.3f}@{best_epoch}")
        if not args.quiet:
            tqdm.write(f"epoch {epoch:3d}  train loss {row['train_loss']:.4f} mIoU {row['train_mIoU']:.3f}"
                       f" | val loss {v['loss']:.4f} mIoU {v['mIoU']:.3f} Dice {v['dice']:.3f}"
                       f" P@0.5 {v['P@0.5']:.3f}")
        if args.patience and bad >= args.patience:
            tqdm.write(f"early stop: no val mIoU gain for {args.patience} epochs")
            break

    torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, out / "last.pt")
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, out / "curves.png", name)

    summary = {"run": name, "best_val_mIoU": best, "best_epoch": best_epoch,
               "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1),
               "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
               "trainable_params": n_params}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not args.quiet:
        print(json.dumps(summary, indent=1))
        print(f"outputs: {out.relative_to(PROJECT_ROOT)}/")
    del train, val
    torch.cuda.empty_cache()
    return out, summary


if __name__ == "__main__":
    main()
