import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, computed per sample then averaged. logits/target: [B,1,H,W]."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    dice = (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return (1 - dice).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0) -> torch.Tensor:
    """BCEWithLogits + lambda * soft Dice (lambda = 0 gives the BCE-only ablation)."""
    loss = F.binary_cross_entropy_with_logits(logits, target)
    if dice_weight > 0:
        loss = loss + dice_weight * soft_dice_loss(logits, target)
    return loss
