"""Frozen CLIP encoders and the on-disk feature cache.

Cache layout (data/cache/<name>/):
    img_feats.npy  float16 [N_img, L, 196, 768]  patch tokens (CLS dropped) of selected layers
    text_emb.npy   float16 [N_sent, 512]         projected CLIP text embedding per sentence
    masks.npy      uint8   [N_ann, S, S]         binary masks resized (nearest) to S x S
    meta.json      model id, layers, size, image_ids, and one record per sample:
                   {img_idx, ann_idx, text_idx, image_id, ann_id, sentence, category}

Optional horizontal-flip augmentation cache (same row order as above):
    img_feats_flip.npy  CLIP features of the mirrored 224x224 image
    text_emb_flip.npy   CLIP embedding of the sentence with left <-> right swapped
A ViT is not flip-equivariant, so flipped features must come from re-encoding the
mirrored image; mirroring the cached feature grid would be wrong. Masks are flipped on
the fly. The text must be swapped too, or "the left kid" would label the right one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

from src.data.refcoco_m import PROJECT_ROOT

CLIP_ID = "openai/clip-vit-base-patch16"
DEFAULT_LAYERS = (3, 6, 9, 12)
CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
DEFAULT_CACHE = CACHE_ROOT / "clip-vit-b16_L3-6-9-12_224"


class FrozenCLIP(torch.nn.Module):
    """Frozen CLIP ViT-B/16: multi-layer patch grids for images, projected embeddings for text."""

    def __init__(self, model_id: str = CLIP_ID, layers=DEFAULT_LAYERS, dtype=torch.float16):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id, dtype=dtype).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = tuple(layers)
        self.dtype = dtype

    def train(self, mode: bool = True):  # always stay in eval mode
        return super().train(False)

    @torch.inference_mode()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B,3,224,224] normalised -> [B, L, 196, 768] patch tokens (CLS removed)."""
        out = self.model.vision_model(
            pixel_values=pixel_values.to(self.dtype), output_hidden_states=True
        )
        hs = out.hidden_states  # hs[0] = embeddings, hs[i] = output of layer i
        return torch.stack([hs[i][:, 1:, :] for i in self.layers], dim=1)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """list[str] -> [B, 512] projected text embeddings (not normalised)."""
        device = next(self.model.parameters()).device
        tok = self.tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        out = self.model.get_text_features(
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
        )
        return out if isinstance(out, torch.Tensor) else out.pooler_output


_LR = {"left": "right", "right": "left", "leftmost": "rightmost", "rightmost": "leftmost",
       "lefthand": "righthand", "righthand": "lefthand"}
_LR_RE = re.compile(r"\b(" + "|".join(_LR) + r")\b", re.IGNORECASE)


def swap_left_right(text: str) -> str:
    """Swap left/right words for a horizontally mirrored image (case-preserving)."""
    def sub(m):
        w = _LR[m.group(0).lower()]
        return w.capitalize() if m.group(0)[0].isupper() else w
    return _LR_RE.sub(sub, text)


def has_flip_cache(cache_dir: Path = DEFAULT_CACHE) -> bool:
    return (Path(cache_dir) / "img_feats_flip.npy").exists() and (Path(cache_dir) / "text_emb_flip.npy").exists()


def load_meta(cache_dir: Path = DEFAULT_CACHE) -> dict:
    return json.loads((Path(cache_dir) / "meta.json").read_text())


class CachedSplit:
    """One partition of the cache held on the GPU; yields ready-to-use batches.

    Image features are copied in chunks (the memmap is never fully materialised in RAM).
    """

    def __init__(self, partition: str, split: dict, cache_dir: Path = DEFAULT_CACHE,
                 device: str = "cuda", max_samples: int | None = None, flip: bool = False):
        cache_dir = Path(cache_dir)
        meta = load_meta(cache_dir)
        keep = set(split[partition])
        samples = [s for s in meta["samples"] if s["image_id"] in keep]
        if max_samples is not None:
            samples = samples[:max_samples]
        self.samples = samples

        img_rows = sorted({s["img_idx"] for s in samples})
        ann_rows = sorted({s["ann_idx"] for s in samples})
        img_pos = {r: i for i, r in enumerate(img_rows)}
        ann_pos = {r: i for i, r in enumerate(ann_rows)}

        self.img_feats = self._load_rows(cache_dir / "img_feats.npy", img_rows, device)
        self.img_feats_flip = None
        if flip:
            if not has_flip_cache(cache_dir):
                raise FileNotFoundError("flip cache missing: run `pixi run cache-features --flip`")
            self.img_feats_flip = self._load_rows(cache_dir / "img_feats_flip.npy", img_rows, device)

        masks = np.load(cache_dir / "masks.npy", mmap_mode="r")
        self.masks = torch.from_numpy(np.ascontiguousarray(masks[ann_rows])).to(device)
        text = np.load(cache_dir / "text_emb.npy", mmap_mode="r")
        text_rows = [s["text_idx"] for s in samples]
        self.text = torch.from_numpy(np.ascontiguousarray(text[text_rows])).to(device)
        self.text_flip = None
        if flip:
            tf = np.load(cache_dir / "text_emb_flip.npy", mmap_mode="r")
            self.text_flip = torch.from_numpy(np.ascontiguousarray(tf[text_rows])).to(device)

        self.img_index = torch.tensor([img_pos[s["img_idx"]] for s in samples], device=device)
        self.ann_index = torch.tensor([ann_pos[s["ann_idx"]] for s in samples], device=device)
        self.device = device

    @staticmethod
    def _load_rows(path: Path, rows: list[int], device: str) -> torch.Tensor:
        arr = np.load(path, mmap_mode="r")
        out = torch.empty((len(rows), *arr.shape[1:]), dtype=torch.float16, device=device)
        for a in range(0, len(rows), 64):  # chunked: never materialise the memmap in RAM
            r = rows[a:a + 64]
            out[a:a + len(r)] = torch.from_numpy(np.ascontiguousarray(arr[r])).to(device)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def batch(self, idx: torch.Tensor, flip: torch.Tensor | None = None) -> dict:
        """flip: optional bool [B]; flipped samples use mirrored image features,
        left/right-swapped text and a horizontally mirrored mask."""
        idx = idx.to(self.device)
        feats = self.img_feats[self.img_index[idx]]
        text = self.text[idx]
        mask = self.masks[self.ann_index[idx]].unsqueeze(1)
        if flip is not None and flip.any():
            assert self.img_feats_flip is not None, "CachedSplit was created without flip=True"
            f = flip.to(self.device)
            feats = torch.where(f[:, None, None, None], self.img_feats_flip[self.img_index[idx]], feats)
            text = torch.where(f[:, None], self.text_flip[idx], text)
            mask = torch.where(f[:, None, None, None], mask.flip(-1), mask)
        return {"feats": feats.float(), "text": text.float(), "mask": mask.float(), "idx": idx}

    def iterate(self, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for a in range(0, n, batch_size):
            yield self.batch(order[a:a + batch_size])
