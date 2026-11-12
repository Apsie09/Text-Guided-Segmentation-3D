"""Lightweight segmentation decoder on frozen CLIP patch features.

One class, three variants that differ ONLY in how text enters:
    fusion="none"   M0  image-only baseline
    fusion="film"   M1  FiLM:  F' = (1 + gamma(t)) * F + beta(t), at two decoder stages
    fusion="concat" M1b text projected to C channels, broadcast, concatenated, 1x1 conv

Shared by all variants (so the comparison is controlled):
    * per-layer LayerNorm + Linear(768 -> C) projections, summed over the cached layers
    * optional CoordConv: normalised (x, y) channels so "left/right/top" can be expressed
    * the same conv/upsampling stack and prediction head
    * optional regularisation (off by default): element-wise dropout on the frozen CLIP
      features (`feat_drop`) and channel dropout (Dropout2d) after the 14x14 and 28x28
      blocks (`drop2d`). Dropout has no weights, so checkpoints load either way.

Input : feats [B, L, 196, 768] (cached patch tokens), text [B, 512] (CLIP text embedding)
Output: mask logits [B, 1, out_size, out_size]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FUSIONS = ("none", "film", "concat")


def coord_grid(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy]).unsqueeze(0).expand(b, -1, -1, -1)


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )


class FiLM(nn.Module):
    """Text -> per-channel (gamma, beta). Zero-initialised so it starts as identity."""

    def __init__(self, text_dim: int, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(t).chunk(2, dim=-1)
        return (1 + gamma[..., None, None]) * x + beta[..., None, None]


class TextSegDecoder(nn.Module):
    def __init__(
        self,
        fusion: str = "film",
        n_layers: int = 4,
        feat_dim: int = 768,
        text_dim: int = 512,
        channels: int = 128,
        text_hidden: int = 256,
        coords: bool = True,
        grid: int = 14,
        out_size: int = 224,
        feat_drop: float = 0.0,
        drop2d: float = 0.0,
    ):
        super().__init__()
        assert fusion in FUSIONS, fusion
        self.fusion, self.coords, self.grid, self.out_size = fusion, coords, grid, out_size
        C = channels

        self.feat_drop = nn.Dropout(feat_drop)
        self.drop2d = nn.Dropout2d(drop2d)
        self.norms = nn.ModuleList(nn.LayerNorm(feat_dim) for _ in range(n_layers))
        self.projs = nn.ModuleList(nn.Linear(feat_dim, C) for _ in range(n_layers))
        self.stem = nn.Conv2d(C + (2 if coords else 0), C, 1)

        if fusion == "film":
            self.film1 = FiLM(text_dim, C, text_hidden)
            self.film2 = FiLM(text_dim, C, text_hidden)
        elif fusion == "concat":
            self.text_proj = nn.Sequential(nn.Linear(text_dim, text_hidden), nn.GELU(),
                                           nn.Linear(text_hidden, C))
            self.fuse = nn.Conv2d(2 * C, C, 1)

        self.block14 = ConvBlock(C, C)  # 14x14
        self.block28 = ConvBlock(C, C // 2)  # 28x28
        self.block56 = ConvBlock(C // 2, C // 4)  # 56x56
        self.head = nn.Conv2d(C // 4, 1, 1)

    def forward(self, feats: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        B, L, N, D = feats.shape
        g = self.grid
        assert N == g * g, (N, g)

        feats = self.feat_drop(feats)
        x = sum(proj(norm(feats[:, i])) for i, (norm, proj) in enumerate(zip(self.norms, self.projs)))
        x = x.transpose(1, 2).reshape(B, -1, g, g)  # [B, C, 14, 14]
        if self.coords:
            x = torch.cat([x, coord_grid(B, g, g, x.device, x.dtype)], dim=1)
        x = self.stem(x)

        if self.fusion != "none":
            assert text is not None, "text-conditioned variant needs text"
            text = F.normalize(text, dim=-1)
        if self.fusion == "film":
            x = self.film1(x, text)
        elif self.fusion == "concat":
            t = self.text_proj(text)[..., None, None].expand(-1, -1, g, g)
            x = self.fuse(torch.cat([x, t], dim=1))

        x = self.block14(x)
        if self.fusion == "film":
            x = self.film2(x, text)
        x = self.drop2d(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.drop2d(self.block28(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block56(x)
        logits = self.head(x)  # [B, 1, 56, 56]
        return F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear",
                             align_corners=False)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
