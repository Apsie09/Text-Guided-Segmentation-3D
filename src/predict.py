"""Inference on arbitrary images: frozen CLIP + a trained decoder checkpoint.

The image goes through exactly the training preprocessing (plain resize to 224x224,
CLIP normalisation). The 224x224 probability map is resized back to the original
image size with bilinear interpolation before thresholding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.refcoco_m import PROJECT_ROOT, to_clip_tensor
from src.evaluate import load_run
from src.features import FrozenCLIP

# Final model (selected on val): FiLM, no coordinate channels, BCE + Dice, flip augmentation.
DEFAULT_RUN = PROJECT_ROOT / "outputs" / "runs" / "film_c128_nocoord_dice1_flip_s0"


class Predictor:
    def __init__(self, run_dir: Path = DEFAULT_RUN, checkpoint: str = "best"):
        if not (Path(run_dir) / f"{checkpoint}.pt").exists():
            raise FileNotFoundError(
                f"{run_dir}/{checkpoint}.pt not found; train it with "
                "`pixi run train --fusion film --no-coords --flip` (or pass --run)")
        self.model, self.cfg = load_run(Path(run_dir), checkpoint)
        self.clip = FrozenCLIP().cuda()
        self.size = 224

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str) -> np.ndarray:
        """Return the foreground probability map [H, W] (float32 in [0, 1]) at image size."""
        image = image.convert("RGB")
        x = to_clip_tensor(image.resize((self.size, self.size), Image.BICUBIC)).unsqueeze(0).cuda()
        feats = self.clip.encode_image(x).float()
        text_emb = self.clip.encode_text([text]).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(feats, text_emb).float()
        prob = torch.sigmoid(F.interpolate(logits, size=(image.height, image.width),
                                           mode="bilinear", align_corners=False))
        return prob[0, 0].cpu().numpy()
