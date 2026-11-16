"""Qualitative figures on the held-out TEST partition for the report.

Figures (report/figures/):
    gallery_random.png       12 random samples, not selected by outcome (honest overview)
    gallery_successes.png    highest-IoU samples of the final model (one per image)
    gallery_failures.png     lowest-IoU samples in three groups:
                               colour/attribute words, crowded images, everything else
    gallery_language.png     same image, two sentences: image-only M0 vs final model

Each panel: original image, ground truth outline (green), prediction (red fill), IoU on
the 224x224 evaluation grid. Uses seed 0 of each model; no selection affects any metric.

Usage: pixi run gallery
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.refcoco_m import PROJECT_ROOT, decode_mask, load_raw, load_split
from src.evaluate import load_run, predict
from src.features import DEFAULT_CACHE, CachedSplit, load_meta
from src.predict import DEFAULT_RUN

FIG_DIR = PROJECT_ROOT / "report" / "figures"
M0_RUN = PROJECT_ROOT / "outputs" / "runs" / "none_c128_dice1_s0"
ATTRIBUTE = re.compile(
    r"\b(black|white|red|blue|green|yellow|orange|pink|purple|brown|gr[ae]y|dark|light|"
    r"striped|plaid|shirt|jacket|hat|cap|dress|coat|shorts|jeans|glasses|hair|blond|bald)\b", re.I)


def wrap(text: str, width: int = 34) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:2]) + ("…" if len(lines) > 2 else "")


class Panels:
    def __init__(self, ds, data: CachedSplit):
        self.ds, self.data = ds, data

    def draw(self, ax, i: int, logits: torch.Tensor, iou: float, label: str | None = None) -> None:
        s = self.data.samples[i]
        rec = self.ds[s["img_idx"]]
        img = np.asarray(rec["image"].convert("RGB"))
        ann = next(a for a in rec["samples"] if a["id"] == s["ann_id"])
        gt = decode_mask(ann["mask"])
        prob = torch.sigmoid(F.interpolate(logits[i:i + 1], size=img.shape[:2], mode="bilinear",
                                           align_corners=False))[0, 0].numpy()
        ax.imshow(img)
        tint = np.zeros((*img.shape[:2], 4), np.float32)
        tint[prob > 0.5] = (1.0, 0.1, 0.1, 0.5)
        ax.imshow(tint)
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
        head = f"{label}  " if label else ""
        ax.set_title(f'{head}IoU {iou:.2f}\n"{wrap(s["sentence"])}"', fontsize=8)
        ax.axis("off")


def grid(panels: Panels, items, logits, ious, path, title: str, cols: int = 4, labels=None) -> None:
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(items):
        panels.draw(axes[k], i, logits, ious[i].item(), labels[k] if labels else None)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title + "   (green outline = ground truth, red = prediction)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved {path.relative_to(PROJECT_ROOT)}")


def one_per_image(order, data, k: int) -> list[int]:
    seen, out = set(), []
    for i in order:
        img = data.samples[int(i)]["image_id"]
        if img not in seen:
            seen.add(img)
            out.append(int(i))
        if len(out) == k:
            break
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    data = CachedSplit("test", load_split(), DEFAULT_CACHE)
    panels = Panels(ds, data)

    final, _ = load_run(DEFAULT_RUN)
    m0, _ = load_run(M0_RUN)
    logits, iou, _ = predict(final, data, desc="final model")
    logits0, iou0, _ = predict(m0, data, desc="image-only")
    print(f"test samples: {len(data)}   final mIoU {iou.mean():.3f}   image-only mIoU {iou0.mean():.3f}")

    rng = np.random.default_rng(0)
    # ---- random, unselected
    rand = one_per_image(rng.permutation(len(data)), data, 12)
    grid(panels, rand, logits, iou, FIG_DIR / "gallery_random.png",
         f"Random test samples, final model ({DEFAULT_RUN.name})")

    # ---- successes
    best = one_per_image(torch.argsort(iou, descending=True), data, 8)
    grid(panels, best, logits, iou, FIG_DIR / "gallery_successes.png", "Best test samples, final model")

    # ---- failures by type
    meta = load_meta(DEFAULT_CACHE)
    anns_per_img: dict[int, set] = {}
    for s in meta["samples"]:
        anns_per_img.setdefault(s["image_id"], set()).add(s["ann_id"])
    attr = torch.tensor([bool(ATTRIBUTE.search(s["sentence"])) for s in data.samples])
    crowd = torch.tensor([len(anns_per_img[s["image_id"]]) >= 4 for s in data.samples])
    worst = torch.argsort(iou)
    groups = [("attribute", worst[attr[worst]]), ("crowded", worst[crowd[worst] & ~attr[worst]]),
              ("other", worst[~attr[worst] & ~crowd[worst]])]
    items, labels, used_imgs = [], [], set()
    for name, order in groups:
        fresh = [int(i) for i in order if data.samples[int(i)]["image_id"] not in used_imgs]
        picks = one_per_image(fresh, data, 4)
        used_imgs |= {data.samples[i]["image_id"] for i in picks}
        items += picks
        labels += [name] * len(picks)
    for name, m in (("attribute words", attr), ("crowded (>=4 objects)", crowd)):
        print(f"  {name:22s}: {int(m.sum()):3d} samples, final mIoU {iou[m].mean():.3f} "
              f"vs rest {iou[~m].mean():.3f}")
    grid(panels, items, logits, iou, FIG_DIR / "gallery_failures.png",
         "Worst test samples by type, final model", labels=labels)

    # ---- language sensitivity: same image, two different targets
    by_img: dict[int, dict[int, int]] = {}
    for i, s in enumerate(data.samples):
        by_img.setdefault(s["image_id"], {}).setdefault(s["ann_id"], i)
    pairs = [list(a.values())[:2] for a in by_img.values() if len(a) >= 2]
    # Rank pairs by how much better the final model is than M0 on BOTH sentences.
    pairs.sort(key=lambda p: -min(iou[p[0]] - iou0[p[0]], iou[p[1]] - iou0[p[1]]).item())
    chosen = pairs[:3]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13.6, 3.3 * len(chosen)))
    for r, (a, b) in enumerate(chosen):
        panels.draw(axes[r, 0], a, logits0, iou0[a].item(), "image-only")
        panels.draw(axes[r, 1], a, logits, iou[a].item(), "final")
        panels.draw(axes[r, 2], b, logits0, iou0[b].item(), "image-only")
        panels.draw(axes[r, 3], b, logits, iou[b].item(), "final")
    fig.suptitle("Same image, two sentences: the image-only model cannot switch targets, the "
                 "text-conditioned model can (selected examples)", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "gallery_language.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved {path.relative_to(PROJECT_ROOT)}")

    # how often does the final model switch correctly on a pair?
    both = [(iou[a] > 0.5) & (iou[b] > 0.5) for a, b in pairs]
    both0 = [(iou0[a] > 0.5) & (iou0[b] > 0.5) for a, b in pairs]
    print(f"pairs on the same image ({len(pairs)}): both sentences IoU > 0.5 -> "
          f"final {np.mean(both):.1%}, image-only {np.mean(both0):.1%}")


if __name__ == "__main__":
    main()
