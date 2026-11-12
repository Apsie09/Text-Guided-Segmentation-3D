"""Evaluate one or more trained runs on the same partition and compare them.

Examples:
    pixi run evaluate outputs/runs/none_c128_dice1_s0 outputs/runs/film_c128_dice1_s0
    pixi run evaluate outputs/runs/* --partition test --i-am-done-tuning   # final, once

Writes to outputs/eval/<partition>/:
    results.md / results.json   mIoU, Dice, P@0.5, params, latency per run
    pairs.png                   same image, two different sentences, prediction of every run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.refcoco_m import PROJECT_ROOT, load_raw, load_split
from src.features import DEFAULT_CACHE, CachedSplit, FrozenCLIP
from src.metrics import iou_dice
from src.models.decoder import TextSegDecoder, count_trainable

EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"


def load_run(run_dir: Path, which: str = "best"):
    ckpt = torch.load(run_dir / f"{which}.pt", map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    model = TextSegDecoder(fusion=cfg["fusion"], n_layers=len(cfg["layers"]),
                           channels=cfg["channels"], coords=not cfg["no_coords"],
                           feat_drop=cfg.get("feat_drop", 0.0), drop2d=cfg.get("drop2d", 0.0)).cuda()
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


@torch.no_grad()
def predict(model, data: CachedSplit, bs: int = 64, desc: str = ""):
    logits_all, ious, dices = [], [], []
    for a in tqdm(range(0, len(data), bs), desc=desc, leave=False, unit="batch"):
        b = data.batch(torch.arange(a, min(a + bs, len(data))))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["feats"], b["text"]).float()
        iou, dice = iou_dice(logits, b["mask"])
        ious.append(iou.cpu())
        dices.append(dice.cpu())
        logits_all.append(logits.cpu())
    return torch.cat(logits_all), torch.cat(ious), torch.cat(dices)


@torch.no_grad()
def latency_ms(model, clip: FrozenCLIP, n: int = 100) -> dict:
    """End-to-end batch-1 latency: CLIP image + text encoding + decoder, on GPU."""
    img = torch.randn(1, 3, 224, 224, device="cuda")
    text = ["the second bowl from the right"]

    def run_full():
        f = clip.encode_image(img).float()
        t = clip.encode_text(text).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(f, t)

    f = clip.encode_image(img).float()
    t = clip.encode_text(text).float()

    def run_dec():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(f, t)

    # Laptop GPUs idle at low clocks: warm up long enough to reach steady state,
    # then report the median of several timed repeats.
    out = {}
    for key, fn in (("full_ms", run_full), ("decoder_ms", run_dec)):
        for _ in range(50):
            fn()
        reps = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            reps.append((time.perf_counter() - t0) / n * 1000)
        out[key] = float(np.median(reps))
    return out


def pick_pairs(data: CachedSplit, n_pairs: int, seed: int = 0) -> list[int]:
    """Sample pairs (same image, different referred object) -> language sensitivity."""
    by_img: dict[int, dict[int, int]] = {}
    for i, s in enumerate(data.samples):
        by_img.setdefault(s["image_id"], {}).setdefault(s["ann_id"], i)
    multi = [list(anns.values()) for anns in by_img.values() if len(anns) >= 2]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(multi), size=min(n_pairs, len(multi)), replace=False)
    return [k for c in chosen for k in multi[c][:2]]


def pairs_figure(ds, data, names, preds, idxs, path: Path) -> None:
    cols = 1 + len(names)
    fig, axes = plt.subplots(len(idxs), cols, figsize=(2.6 * cols, 2.7 * len(idxs)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        s = data.samples[i]
        img = np.asarray(ds[s["img_idx"]]["image"].convert("RGB").resize((224, 224), 3))
        gt = data.masks[data.ann_index[i]].cpu().numpy()
        panels = [("GT", gt, None)] + [
            (n, (torch.sigmoid(p[i, 0]) > 0.5).numpy(), iou[i].item()) for n, (p, iou) in zip(names, preds)
        ]
        for c, (title, m, iou) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            tint = np.zeros((224, 224, 4), np.float32)
            tint[m > 0] = (0.1, 0.9, 0.1, 0.5) if c == 0 else (1.0, 0.1, 0.1, 0.5)
            ax.imshow(tint)
            ax.set_title(title if iou is None else f"{title}\nIoU {iou:.2f}", fontsize=7)
            ax.axis("off")
        sent = s["sentence"] if len(s["sentence"]) < 40 else s["sentence"][:37] + "..."
        axes[r, 0].text(0, -30, f'"{sent}"', fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--partition", choices=("val", "test"), default="val")
    ap.add_argument("--i-am-done-tuning", action="store_true",
                    help="required for --partition test: the test set is used once, at the end")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ap.add_argument("--pairs", type=int, default=5)
    args = ap.parse_args()
    if args.partition == "test" and not args.i_am_done_tuning:
        ap.error("test set is held out; pass --i-am-done-tuning once the comparison is final")

    runs = [r for r in args.runs if (r / f"{args.checkpoint}.pt").exists()]
    out = EVAL_DIR / args.partition
    out.mkdir(parents=True, exist_ok=True)

    data = CachedSplit(args.partition, load_split(), DEFAULT_CACHE)
    clip = FrozenCLIP().cuda()
    print(f"{args.partition}: {len(data)} samples, {len(runs)} runs")

    names, preds, rows = [], [], []
    for run in tqdm(runs, desc="runs", unit="run"):
        model, cfg = load_run(run, args.checkpoint)
        logits, iou, dice = predict(model, data, desc=f"  {run.name}")
        lat = latency_ms(model, clip)
        rows.append({
            "run": run.name, "fusion": cfg["fusion"], "coords": not cfg["no_coords"],
            "dice_weight": cfg["dice_weight"], "params": count_trainable(model),
            "mIoU": iou.mean().item(), "Dice": dice.mean().item(),
            "P@0.5": (iou > 0.5).float().mean().item(), **lat,
        })
        names.append(f"{cfg['fusion']}")
        preds.append((logits, iou))

    hdr = "| run | fusion | coords | params | mIoU ↑ | Dice ↑ | P@0.5 ↑ | full ms ↓ | decoder ms ↓ |"
    lines = [f"Partition: **{args.partition}** ({len(data)} samples), checkpoint: {args.checkpoint}, "
             f"GPU: {torch.cuda.get_device_name(0)}, batch 1", "", hdr, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['run']} | {r['fusion']} | {r['coords']} | {r['params']:,} | {r['mIoU']:.3f} | "
                     f"{r['Dice']:.3f} | {r['P@0.5']:.3f} | {r['full_ms']:.1f} | {r['decoder_ms']:.2f} |")
    table = "\n".join(lines)
    print("\n" + table)
    (out / "results.md").write_text(table + "\n")
    (out / "results.json").write_text(json.dumps(rows, indent=1))

    idxs = pick_pairs(data, args.pairs)
    pairs_figure(load_raw(), data, names, preds, idxs, out / "pairs.png")
    print(f"\nsaved {out.relative_to(PROJECT_ROOT)}/results.md and pairs.png")


if __name__ == "__main__":
    main()
