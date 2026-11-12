"""Train every variant with several seeds, then compare them on the validation set.

Variants (all share split, optimiser, schedule, epochs, early stopping):
    M0      image-only                       --fusion none
    M1      FiLM                             --fusion film
    M1b     concat                           --fusion concat
    M1-nc   FiLM without coordinate channels --fusion film --no-coords
    M1-bce  FiLM with BCE only               --fusion film --dice-weight 0
plus CLIPSeg as an external reference (not a controlled ablation).

Runs that already have best.pt + summary.json are reused (use --force to retrain).

Group "reg" (overfitting study), all on top of FiLM without coordinate channels:
    baseline  no regularisation
    R1        feature dropout 0.1 + decoder channel dropout 0.1 + weight decay 0.05
    R2        horizontal flip with left<->right swapped text (needs `cache-features --flip`)
    R3        R1 + R2

Every row also reports train mIoU of the selected checkpoint (eval mode, no dropout/flip)
and the train - val gap, which is the direct measure of overfitting.

Usage: pixi run experiments [--group main|reg] [--seeds 0 1 2] [--force]
       pixi run experiments --group main --partition test --i-am-done-tuning   # final, once
Writes outputs/eval/<partition>/experiments[_<group>].md and .json

On the test partition nothing is trained: every checkpoint must already exist, and CLIPSeg
reuses the threshold chosen on val.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from src import train as train_mod
from src.clipseg_baseline import OUT_DIR
from src.data.refcoco_m import PROJECT_ROOT, load_split
from src.evaluate import latency_ms, load_run, predict
from src.features import DEFAULT_CACHE, CachedSplit, FrozenCLIP, load_meta

_NC = ["--fusion", "film", "--no-coords"]
_R1 = ["--feat-drop", "0.1", "--drop2d", "0.1", "--weight-decay", "0.05"]
GROUPS = {
    "main": {
        "M0 image-only": ["--fusion", "none"],
        "M1 FiLM": ["--fusion", "film"],
        "M1b concat": ["--fusion", "concat"],
        "M1 FiLM, no coords": _NC,
        "M1 FiLM, BCE only": ["--fusion", "film", "--dice-weight", "0"],
    },
    "reg": {
        "FiLM-nc baseline": _NC,
        "R1 dropout + wd": _NC + _R1,
        "R2 flip + L/R swap": _NC + ["--flip"],
        "R3 R1 + R2": _NC + _R1 + ["--flip"],
    },
}
# Final comparison: every variant once (the reg baseline is "M1 FiLM, no coords").
GROUPS["all"] = {**GROUPS["main"],
                 **{k: v for k, v in GROUPS["reg"].items() if k != "FiLM-nc baseline"}}

SPATIAL = re.compile(
    r"\b(left|right|top|bottom|middle|center|centre|front|back|behind|closest|nearest|"
    r"far|farthest|upper|lower|first|second|third|1st|2nd|3rd|above|below|next)\b", re.I)


def run_dir_for(argv: list[str]):
    return train_mod.RUNS_DIR / train_mod.make_run_name(train_mod.parse_args(argv))


def subsets(data: CachedSplit) -> dict[str, torch.Tensor]:
    """Boolean masks over samples for the breakdown columns."""
    meta = load_meta(DEFAULT_CACHE)
    anns_per_img: dict[int, set] = {}
    for s in meta["samples"]:
        anns_per_img.setdefault(s["image_id"], set()).add(s["ann_id"])
    multi = torch.tensor([len(anns_per_img[s["image_id"]]) >= 2 for s in data.samples])
    spatial = torch.tensor([bool(SPATIAL.search(s["sentence"])) for s in data.samples])
    return {"multi-object img": multi, "spatial words": spatial, "no spatial words": ~spatial}


def fmt(vals, digits: int = 3) -> str:
    v = np.asarray(vals, dtype=float)
    if len(v) > 1:
        return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"
    return f"{v.mean():.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), default="main")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--partition", choices=("val", "test"), default="val")
    ap.add_argument("--i-am-done-tuning", action="store_true",
                    help="required for --partition test: the test set is used once, at the end")
    args = ap.parse_args()
    if args.partition == "test" and not args.i_am_done_tuning:
        ap.error("test set is held out; pass --i-am-done-tuning once the comparison is final")
    if args.partition == "test" and args.force:
        ap.error("--force retrains models; never combine it with the test partition")
    VARIANTS = GROUPS[args.group]

    # ---------------------------------------------------------------- 1. train
    jobs = [(v, s) for v in VARIANTS for s in args.seeds]
    bar = tqdm(jobs, desc="training", unit="run")
    for variant, seed in bar:
        argv = VARIANTS[variant] + ["--seed", str(seed), "--quiet"]
        rd = run_dir_for(argv)
        bar.set_postfix_str(rd.name)
        if not args.force and (rd / "best.pt").exists() and (rd / "summary.json").exists():
            continue
        if args.partition == "test":
            raise SystemExit(f"{rd.name} is not trained; train on val first, test only evaluates")
        train_mod.main(argv)

    # ---------------------------------------------------------------- 2. CLIPSeg
    with_clipseg = args.group in ("main", "all")
    if with_clipseg and not (OUT_DIR / "val" / "clipseg.json").exists():
        print("running CLIPSeg reference on val ...")
        subprocess.run([sys.executable, "-m", "src.clipseg_baseline", "--partition", "val"], check=True)
    if with_clipseg and args.partition == "test" and not (OUT_DIR / "test" / "clipseg.json").exists():
        print("running CLIPSeg reference on test (threshold from val) ...")
        subprocess.run([sys.executable, "-m", "src.clipseg_baseline", "--partition", "test",
                        "--i-am-done-tuning"], check=True)

    # ---------------------------------------------------------------- 3. evaluate
    split = load_split()
    data = CachedSplit(args.partition, split, DEFAULT_CACHE)
    train_data = CachedSplit("train", split, DEFAULT_CACHE)
    subs = subsets(data)
    clip = FrozenCLIP().cuda()
    results: dict[str, dict] = {}
    for variant in tqdm(VARIANTS, desc="evaluating", unit="variant"):
        r = {"mIoU": [], "Dice": [], "P@0.5": [], "best_epoch": [], "train mIoU": [], "gap": [],
             **{k: [] for k in subs}}
        for seed in args.seeds:
            rd = run_dir_for(VARIANTS[variant] + ["--seed", str(seed)])
            model, cfg = load_run(rd)
            _, iou, dice = predict(model, data, desc=f"  {rd.name}")
            _, tr_iou, _ = predict(model, train_data, desc=f"  {rd.name} (train)")
            r["train mIoU"].append(tr_iou.mean().item())
            r["gap"].append(tr_iou.mean().item() - iou.mean().item())
            r["mIoU"].append(iou.mean().item())
            r["Dice"].append(dice.mean().item())
            r["P@0.5"].append((iou > 0.5).float().mean().item())
            r["best_epoch"].append(json.loads((rd / "summary.json").read_text())["best_epoch"])
            for k, m in subs.items():
                r[k].append(iou[m].mean().item())
            if seed == args.seeds[0]:
                r["params"] = cfg["trainable_params"]
                r.update(latency_ms(model, clip))
        results[variant] = r

    if with_clipseg:
        clipseg = json.loads((OUT_DIR / args.partition / "clipseg.json").read_text())
        cs_iou = torch.tensor(clipseg["iou"])
        results["CLIPSeg (reference)"] = {
            "mIoU": [clipseg["mIoU"]], "Dice": [clipseg["Dice"]], "P@0.5": [clipseg["P@0.5"]],
            **{k: [cs_iou[m].mean().item()] for k, m in subs.items()},
            "params": clipseg["total_params"], "full_ms": clipseg["full_ms"], "decoder_ms": float("nan"),
            "best_epoch": [], "train mIoU": [], "gap": [],
            "note": f"pretrained, threshold {clipseg['threshold']} chosen on val",
        }

    # ---------------------------------------------------------------- 4. report
    n_sub = {k: int(m.sum()) for k, m in subs.items()}
    cols = ["mIoU", "Dice", "P@0.5"] + list(subs) + ["train mIoU", "gap"]
    lines = [
        f"{args.partition.capitalize()} partition: {len(data)} samples / {len(set(s['image_id'] for s in data.samples))} images. "
        f"Seeds: {args.seeds} (mean ± sample std). GPU: {torch.cuda.get_device_name(0)}, latency at batch 1.",
        "Subset sizes: " + ", ".join(f"{k} = {v}" for k, v in n_sub.items()) + ".",
        "",
        "| variant | trainable params | " + " | ".join(cols) + " | full ms | best epoch |",
        "|---|---:|" + "---:|" * len(cols) + "---:|---:|",
    ]
    for v, r in results.items():
        params = f"{r['params']:,}" + (" (all frozen)" if v.startswith("CLIPSeg") else "")
        be = fmt(r["best_epoch"], 1) if r["best_epoch"] else "–"
        lines.append(f"| {v} | {params} | " + " | ".join(fmt(r[c]) if r[c] else "–" for c in cols)
                     + f" | {r['full_ms']:.1f} | {be} |")

    names = list(VARIANTS)
    ref = np.array(results[names[0]]["mIoU"])
    lines += ["", f"Paired by seed, mIoU relative to '{names[0]}':"]
    for v in names[1:]:
        d = np.array(results[v]["mIoU"]) - ref
        lines.append(f"- {v}: {fmt(d)} (ahead in {(d > 0).sum()}/{len(d)} seeds)")
    if args.group in ("main", "all"):
        film, concat = np.array(results["M1 FiLM"]["mIoU"]), np.array(results["M1b concat"]["mIoU"])
        d = film - concat
        lines.append(f"- FiLM − concat: {fmt(d)} (FiLM ahead in {(d > 0).sum()}/{len(d)} seeds)")
    if with_clipseg:
        lines.append(f"\nCLIPSeg: {results['CLIPSeg (reference)']['note']}; different pretraining "
                     "(PhraseCut), so it is a reference point, not a controlled comparison.")
    lines.append("train mIoU: selected checkpoint evaluated on the training partition in eval mode; "
                 "gap = train mIoU − val mIoU.")
    table = "\n".join(lines)
    print("\n" + table)
    out = OUT_DIR / args.partition
    out.mkdir(parents=True, exist_ok=True)
    stem = "experiments" if args.group == "main" else f"experiments_{args.group}"
    (out / f"{stem}.md").write_text(table + "\n")
    (out / f"{stem}.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"\nsaved {(out / f'{stem}.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
