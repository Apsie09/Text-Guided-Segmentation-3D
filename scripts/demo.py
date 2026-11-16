"""Segment any image from one or more sentences with the final model.

Usage:
    pixi run demo path/to/image.jpg "the red cup on the left" "the plate"
    pixi run demo photo.png "man in white shirt" --out my_result.png --threshold 0.4

Saves one figure: the input image, then one panel per sentence showing the probability
map and the thresholded mask. Default output: outputs/demo/<image>_<sentence>.png
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.refcoco_m import PROJECT_ROOT
from src.predict import DEFAULT_RUN, Predictor

OUT_DIR = PROJECT_ROOT / "outputs" / "demo"


def overlay(ax, img: np.ndarray, prob: np.ndarray, thr: float, title: str) -> None:
    ax.imshow(img)
    heat = np.zeros((*prob.shape, 4), np.float32)
    heat[..., 0] = 1.0
    heat[..., 3] = 0.55 * (prob > thr)
    ax.imshow(heat)
    ax.contour(prob, levels=[thr], colors="yellow", linewidths=1.2)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("texts", nargs="+", help="one or more referring sentences")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    img = np.asarray(image)
    predictor = Predictor(args.run)

    n = len(args.texts)
    fig, axes = plt.subplots(2, n + 1, figsize=(4.2 * (n + 1), 7.0),
                             gridspec_kw={"height_ratios": [1, 1]})
    axes = np.asarray(axes).reshape(2, n + 1)
    axes[0, 0].imshow(img); axes[0, 0].set_title(f"input {image.width}x{image.height}"); axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, f"model: {args.run.name}\nthreshold: {args.threshold}",
                    fontsize=9, family="monospace", va="center")

    for k, text in enumerate(args.texts, start=1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        prob = predictor.predict(image, text)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
        mask = prob > args.threshold
        print(f'"{text}": mask covers {mask.mean():.1%} of the image, max p = {prob.max():.2f}, {ms:.0f} ms')
        overlay(axes[0, k], img, prob, args.threshold, f'"{text}"')
        im = axes[1, k].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[1, k].set_title("P(target | text)", fontsize=9)
        axes[1, k].axis("off")
    fig.colorbar(im, ax=axes[1, 1:].tolist(), fraction=0.02)

    out = args.out
    if out is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", args.texts[0].lower()).strip("-")[:30]
        out = OUT_DIR / f"{args.image.stem}_{slug}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
