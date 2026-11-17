"""Learning-curve figure that shows overfitting, for the paper.

Rows: FiLM without flip augmentation / FiLM with flip (final model).
Dotted line: epoch with the best mean validation mIoU.
Columns: loss and mIoU, training vs validation, mean over 3 seeds (band = min..max of the seeds).
Training values are the running averages logged during each epoch (training mode, with
augmentation); validation values are computed after each epoch.

Usage:
    pixi run python scripts/make_overfit_figure.py            # English labels
    pixi run python scripts/make_overfit_figure.py --lang bg  # Bulgarian labels
Writes report/paper/figures/overfitting_curves[_bg].pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from src.data.refcoco_m import PROJECT_ROOT

RUNS = PROJECT_ROOT / "outputs" / "runs"
OUT = PROJECT_ROOT / "report" / "paper" / "figures"
TEMPORA = Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/tempora")

TRAIN_C, VAL_C = "#2a78d6", "#eb6834"   # validated categorical pair (light surface)
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"

LABELS = {
    "en": dict(rows=["no flip", "with flip (final)"], loss="loss", miou="mIoU",
               epoch="epoch", train="training", val="validation"),
    "bg": dict(rows=["без отражение", "с отражение (финален)"], loss="загуба",
               miou="mIoU", epoch="епоха", train="обучение", val="валидация"),
}
GROUPS = ["film_c128_nocoord_dice1_s{}", "film_c128_nocoord_dice1_flip_s{}"]


def load(pattern: str) -> dict[str, np.ndarray]:
    hists = [list(csv.DictReader(open(RUNS / pattern.format(s) / "history.csv"))) for s in range(3)]
    n = min(len(h) for h in hists)  # early stopping: keep the epochs every seed has
    return {k: np.array([[float(r[k]) for r in h[:n]] for h in hists])
            for k in ("train_loss", "val_loss", "train_mIoU", "val_mIoU")}


def setup_font() -> None:
    if TEMPORA.exists():
        for f in TEMPORA.glob("*.otf"):
            font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Tempora"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                         "pdf.fonttype": 3})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "bg"], default="en")
    lang = ap.parse_args().lang
    L = LABELS[lang]
    setup_font()

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.1), sharex="col", sharey="col")
    for r, pattern in enumerate(GROUPS):
        h = load(pattern)
        ep = np.arange(1, h["val_loss"].shape[1] + 1)
        best = int(np.argmax(h["val_mIoU"].mean(0))) + 1
        for c, (key, ylab) in enumerate((("loss", L["loss"]), ("mIoU", L["miou"]))):
            ax = axes[r, c]
            for split, color in (("train", TRAIN_C), ("val", VAL_C)):
                y = h[f"{split}_{key}"]
                ax.fill_between(ep, y.min(0), y.max(0), color=color, alpha=0.18, lw=0)
                ax.plot(ep, y.mean(0), color=color, lw=1.6, label=L[split])
            ax.axvline(best, color=MUTED, lw=0.8, ls=(0, (2, 2)))
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(length=2, width=0.5)
            if r == 1:
                ax.set_xlabel(L["epoch"])
            if r == 0:
                ax.set_title(ylab, fontsize=8, color=INK)
        axes[r, 0].set_ylabel(L["rows"][r], fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, 1.02), handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"overfitting_curves{'' if lang == 'en' else '_bg'}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
