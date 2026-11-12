"""Day-1 data check for RefCOCO-M.

* creates (or reuses) the fixed image-grouped split file in splits/,
* prints dataset statistics and sanity checks (mask size, mask-vs-bbox agreement, leakage),
* saves N random (image, sentence, mask) overlays at original and 224x224 resolution
  to outputs/inspect/ for visual verification.

Usage: pixi run inspect-data [--n 20] [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.data.refcoco_m import (
    DEFAULT_SPLIT_FILE,
    PROJECT_ROOT,
    RefCocoM,
    build_index,
    decode_mask,
    load_raw,
    load_split,
    make_group_split,
    resize_pair,
    save_split,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "inspect"
FRACTIONS = (0.7, 0.15, 0.15)


def mask_bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]


def box_iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def overlay(ax, img, m, title, bbox=None):
    ax.imshow(img)
    tint = np.zeros((*m.shape, 4), dtype=np.float32)
    tint[m > 0] = (1.0, 0.1, 0.1, 0.45)
    ax.imshow(tint)
    if bbox is not None:
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, ec="lime", lw=1.5
            )
        )
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def statistics(ds, refs, split) -> None:
    print("=" * 70)
    n_ann = sum(len(r["samples"]) for r in ds.select_columns(["samples"]))
    print(f"images: {len(ds)}   annotations(masks): {n_ann}   sentences(samples): {len(refs)}")

    anns_per_img = collections.Counter()
    sents_per_ann = collections.Counter()
    cats = collections.Counter()
    area_frac, bbox_ious, size_mismatch, empty = [], [], 0, 0
    for rec in ds:
        img = rec["image"]
        anns_per_img[len(rec["samples"])] += 1
        for ann in rec["samples"]:
            sents_per_ann[len(ann["sentences"])] += 1
            cats[ann["category"]] += 1
            m = decode_mask(ann["mask"])
            if m.shape != (img.height, img.width):
                size_mismatch += 1
            mb = mask_bbox(m)
            if mb is None:
                empty += 1
                continue
            area_frac.append(m.mean())
            bbox_ious.append(box_iou(mb, ann["bbox"]))

    af = np.array(area_frac)
    bi = np.array(bbox_ious)
    print(f"annotations per image : {dict(sorted(anns_per_img.items()))}")
    print(f"sentences per mask    : {dict(sorted(sents_per_ann.items()))}")
    print(f"mask/image size mismatches: {size_mismatch}   empty masks: {empty}")
    print(
        "mask area fraction    : "
        f"min {af.min():.4f}  p10 {np.percentile(af, 10):.4f}  median {np.median(af):.4f}  "
        f"p90 {np.percentile(af, 90):.4f}  max {af.max():.4f}"
    )
    print(f"  masks < 1% of image : {(af < 0.01).sum()} ({(af < 0.01).mean():.1%})")
    print(
        "IoU(mask bbox, stored bbox): "
        f"median {np.median(bi):.3f}  p05 {np.percentile(bi, 5):.3f}  "
        f"<0.5: {(bi < 0.5).sum()}"
    )
    print(f"top categories        : {cats.most_common(10)}")

    print("-" * 70)
    by_img = collections.Counter(r.image_id for r in refs)
    for part in ("train", "val", "test"):
        ids = set(split[part])
        n_s = sum(c for i, c in by_img.items() if i in ids)
        print(f"{part:5s}: {len(ids):4d} images  {n_s:5d} samples")
    a, b, c = (set(split[k]) for k in ("train", "val", "test"))
    assert not (a & b or a & c or b & c), "image ids leak across partitions"
    assert a | b | c == set(by_img), "split does not cover all image ids"
    print("split check: disjoint and complete ✓")
    print("=" * 70)


def save_overlays(ds, n: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = RefCocoM("train", ds=ds)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(train), size=min(n, len(train)), replace=False)

    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    grid, gaxes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    gaxes = np.atleast_1d(gaxes).ravel()

    for k, i in enumerate(picks):
        image, mask, ref = train.load_original(int(i))
        ann = ds[ref.row]["samples"][ref.ann_idx]
        img_r, m_r = resize_pair(image, mask, 224)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        overlay(axes[0], image, mask, f"original {image.width}x{image.height}", ann["bbox"])
        overlay(axes[1], img_r, m_r, "resized 224x224 (input to model)")
        fig.suptitle(f'"{ref.sentence}"  [{ref.category}]  img {ref.image_id}', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{k:02d}_img{ref.image_id}_ann{ref.ann_id}.png", dpi=110)
        plt.close(fig)

        title = ref.sentence if len(ref.sentence) < 34 else ref.sentence[:31] + "..."
        overlay(gaxes[k], img_r, m_r, f'"{title}"')

    for ax in gaxes[len(picks) :]:
        ax.axis("off")
    grid.tight_layout()
    grid.savefig(OUT_DIR / "grid.png", dpi=110)
    plt.close(grid)
    print(f"saved {len(picks)} overlays + grid.png to {OUT_DIR.relative_to(PROJECT_ROOT)}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_raw()
    refs = build_index(ds)

    if DEFAULT_SPLIT_FILE.exists():
        print(f"using existing split {DEFAULT_SPLIT_FILE.relative_to(PROJECT_ROOT)}")
    else:
        split = make_group_split([r.image_id for r in refs], seed=0, fractions=FRACTIONS)
        save_split(split, DEFAULT_SPLIT_FILE, seed=0, fractions=FRACTIONS)
        print(f"wrote split {DEFAULT_SPLIT_FILE.relative_to(PROJECT_ROOT)}")
    split = load_split()

    statistics(ds, refs, split)
    save_overlays(ds, args.n, args.seed)


if __name__ == "__main__":
    main()
