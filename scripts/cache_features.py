"""Run the frozen CLIP encoders once over RefCOCO-M and store features on disk.

Usage: pixi run cache-features [--batch 32]
       pixi run cache-features --flip    # + mirrored images and left<->right swapped text
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from src.data.refcoco_m import decode_mask, load_raw, resize_pair, to_clip_tensor
from PIL import Image

from src.features import CLIP_ID, DEFAULT_CACHE, DEFAULT_LAYERS, FrozenCLIP, load_meta, swap_left_right

SIZE = 224


def cache_flip(batch: int) -> None:
    """Mirrored-image features and left/right-swapped text, in the existing row order."""
    out = DEFAULT_CACHE
    meta = load_meta(out)
    ds = load_raw()
    clip = FrozenCLIP().cuda()
    n_img, L = len(meta["image_ids"]), len(meta["layers"])
    feats = np.lib.format.open_memmap(out / "img_feats_flip.npy", mode="w+",
                                      dtype=np.float16, shape=(n_img, L, 196, 768))
    for a in tqdm(range(0, n_img, batch), desc="flipped images"):
        rows = ds[a:a + batch]
        assert [int(i) for i in rows["image_id"]] == meta["image_ids"][a:a + batch]
        pix = [to_clip_tensor(img.convert("RGB").resize((SIZE, SIZE), resample=3)
                              .transpose(Image.FLIP_LEFT_RIGHT)) for img in rows["image"]]
        feats[a:a + len(pix)] = clip.encode_image(torch.stack(pix).cuda()).cpu().numpy()
    feats.flush()

    swapped = [swap_left_right(s["sentence"]) for s in meta["samples"]]
    n_changed = sum(a != s["sentence"] for a, s in zip(swapped, meta["samples"]))
    text = np.zeros((len(swapped), 512), dtype=np.float16)
    for a in tqdm(range(0, len(swapped), 256), desc="swapped text"):
        text[a:a + 256] = clip.encode_text(swapped[a:a + 256]).cpu().numpy()
    np.save(out / "text_emb_flip.npy", text)
    print(f"flip cache: {n_img} mirrored images; {n_changed}/{len(swapped)} sentences had left/right swapped")
    for s, t in [(s["sentence"], t) for s, t in zip(meta["samples"], swapped) if s["sentence"] != t][:5]:
        print(f"   {s!r:45} -> {t!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--flip", action="store_true", help="only build the horizontal-flip cache")
    args = ap.parse_args()
    if args.flip:
        cache_flip(args.batch)
        return

    out = DEFAULT_CACHE
    out.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    clip = FrozenCLIP().cuda()

    # ---- index: images, annotations, sentences
    image_ids, samples, n_ann = [], [], 0
    meta_rows = ds.select_columns(["image_id", "samples"])
    for img_idx, rec in enumerate(meta_rows):
        image_ids.append(int(rec["image_id"]))
        for ann in rec["samples"]:
            for sent in ann["sentences"]:
                samples.append({
                    "img_idx": img_idx, "ann_idx": n_ann, "text_idx": len(samples),
                    "image_id": int(rec["image_id"]), "ann_id": int(ann["id"]),
                    "sentence": sent.strip(), "category": ann["category"],
                })
            n_ann += 1

    n_img, L = len(image_ids), len(DEFAULT_LAYERS)
    feats = np.lib.format.open_memmap(out / "img_feats.npy", mode="w+",
                                      dtype=np.float16, shape=(n_img, L, 196, 768))
    masks = np.lib.format.open_memmap(out / "masks.npy", mode="w+",
                                      dtype=np.uint8, shape=(n_ann, SIZE, SIZE))

    # ---- images + masks
    ann_cursor = 0
    for a in tqdm(range(0, n_img, args.batch), desc="images"):
        rows = ds[a:a + args.batch]
        pix = []
        for img, anns in zip(rows["image"], rows["samples"]):
            img = img.convert("RGB")
            img_r = img.resize((SIZE, SIZE), resample=3)  # bicubic, same as resize_pair
            pix.append(to_clip_tensor(img_r))
            for ann in anns:
                _, m_r = resize_pair(img, decode_mask(ann["mask"]), SIZE)
                masks[ann_cursor] = m_r
                ann_cursor += 1
        f = clip.encode_image(torch.stack(pix).cuda())
        feats[a:a + len(pix)] = f.cpu().numpy()
    assert ann_cursor == n_ann

    # ---- text
    sentences = [s["sentence"] for s in samples]
    text = np.zeros((len(sentences), 512), dtype=np.float16)
    for a in tqdm(range(0, len(sentences), 256), desc="text"):
        text[a:a + 256] = clip.encode_text(sentences[a:a + 256]).cpu().numpy()
    np.save(out / "text_emb.npy", text)

    feats.flush(); masks.flush()
    (out / "meta.json").write_text(json.dumps({
        "model": CLIP_ID, "layers": list(DEFAULT_LAYERS), "size": SIZE,
        "resize": "plain resize, bicubic image / nearest mask, no crop",
        "image_ids": image_ids, "samples": samples,
    }))
    print(f"cached {n_img} images, {n_ann} masks, {len(samples)} sentences -> {out}")


if __name__ == "__main__":
    main()
