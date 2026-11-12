"""Off-the-shelf CLIPSeg (CIDAS/clipseg-rd64-refined) as an external reference.

Not a controlled ablation: different decoder and pretraining (PhraseCut). It uses its own
preprocessing (352x352 input). Its logits are resized to 224x224 and scored against exactly
the same 224x224 masks and samples as our models.

CLIPSeg's sigmoid outputs are poorly calibrated around 0.5, so the binarisation threshold is
selected on the VALIDATION partition only and then reused unchanged for test.

Usage: pixi run clipseg [--partition val|test]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from src.data.refcoco_m import PROJECT_ROOT, load_raw, load_split
from src.features import DEFAULT_CACHE, CachedSplit

CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
THRESHOLDS = [float(round(t, 2)) for t in np.arange(0.10, 0.61, 0.05)]
OUT_DIR = PROJECT_ROOT / "outputs" / "eval"


def iou_dice_np(prob: torch.Tensor, gt: torch.Tensor, thr: float):
    pred = (prob > thr).flatten(1)
    t = gt.flatten(1) > 0.5
    inter = (pred & t).sum(1).float()
    union = (pred | t).sum(1).float()
    s = pred.sum(1).float() + t.sum(1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    dice = torch.where(s > 0, 2 * inter / s.clamp(min=1), torch.ones_like(union))
    return iou, dice


@torch.no_grad()
def run_clipseg(partition: str, batch: int = 16) -> dict:
    data = CachedSplit(partition, load_split(), DEFAULT_CACHE)
    ds = load_raw()
    proc = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID, dtype=torch.float16).cuda().eval()

    probs = []
    img_cache: dict[int, object] = {}
    for a in tqdm(range(0, len(data), batch), desc=f"CLIPSeg {partition}", unit="batch"):
        ss = data.samples[a:a + batch]
        imgs = []
        for s in ss:
            if s["img_idx"] not in img_cache:  # ~180 images per partition: fits in RAM
                img_cache[s["img_idx"]] = ds[s["img_idx"]]["image"].convert("RGB")
            imgs.append(img_cache[s["img_idx"]])
        inp = proc(text=[s["sentence"] for s in ss], images=imgs, padding=True, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        logits = model(**inp).logits.float()
        if logits.dim() == 2:  # batch of 1 gets squeezed by the model
            logits = logits.unsqueeze(0)
        p = torch.sigmoid(F.interpolate(logits.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False))
        probs.append(p.cpu())
    probs = torch.cat(probs)
    gts = data.masks[data.ann_index].unsqueeze(1).cpu()

    per_thr = {}
    for t in THRESHOLDS:
        iou, dice = iou_dice_np(probs, gts, t)
        per_thr[t] = {"mIoU": iou.mean().item(), "Dice": dice.mean().item(),
                      "P@0.5": (iou > 0.5).float().mean().item(), "iou": iou.tolist()}

    # latency, batch 1, end-to-end (preprocessing excluded, like our models)
    s = data.samples[0]
    inp = proc(text=[s["sentence"]], images=[ds[s["img_idx"]]["image"].convert("RGB")],
               return_tensors="pt").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].half()
    for _ in range(50):
        model(**inp)
    reps = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(100):
            model(**inp)
        torch.cuda.synchronize(); reps.append((time.perf_counter() - t0) / 100 * 1000)

    n_params = sum(p.numel() for p in model.parameters())
    return {"partition": partition, "n": len(data), "per_threshold": per_thr,
            "full_ms": float(np.median(reps)), "total_params": n_params}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=("val", "test"), default="val")
    ap.add_argument("--i-am-done-tuning", action="store_true")
    args = ap.parse_args()
    if args.partition == "test" and not args.i_am_done_tuning:
        ap.error("test set is held out; pass --i-am-done-tuning once the comparison is final")

    res = run_clipseg(args.partition)
    if args.partition == "val":
        thr = max(res["per_threshold"], key=lambda t: res["per_threshold"][t]["mIoU"])
    else:
        thr = json.loads((OUT_DIR / "val" / "clipseg.json").read_text())["threshold"]
    best = res["per_threshold"][thr]
    out = {"model": CLIPSEG_ID, "partition": args.partition, "threshold": thr,
           "threshold_source": "val", "mIoU": best["mIoU"], "Dice": best["Dice"],
           "P@0.5": best["P@0.5"], "mIoU@0.5thr": res["per_threshold"][0.5]["mIoU"],
           "full_ms": res["full_ms"], "total_params": res["total_params"], "iou": best["iou"]}
    (OUT_DIR / args.partition).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / args.partition / "clipseg.json").write_text(json.dumps(out))
    print("threshold sweep (mIoU):",
          {t: round(v["mIoU"], 3) for t, v in res["per_threshold"].items()})
    print(f"CLIPSeg {args.partition}: thr={thr}  mIoU {out['mIoU']:.3f}  Dice {out['Dice']:.3f}  "
          f"P@0.5 {out['P@0.5']:.3f}  ({out['full_ms']:.1f} ms, {out['total_params'] / 1e6:.0f}M params)")


if __name__ == "__main__":
    main()
