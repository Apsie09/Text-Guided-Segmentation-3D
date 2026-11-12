import torch


@torch.no_grad()
def iou_dice(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """Per-sample IoU and Dice of the thresholded prediction. Returns two [B] tensors.

    Empty prediction AND empty target counts as IoU = Dice = 1.
    """
    pred = (torch.sigmoid(logits) > thr).flatten(1)
    t = target.flatten(1) > 0.5
    inter = (pred & t).sum(1).float()
    union = (pred | t).sum(1).float()
    psum, tsum = pred.sum(1).float(), t.sum(1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    dice = torch.where(psum + tsum > 0, 2 * inter / (psum + tsum).clamp(min=1), torch.ones_like(union))
    return iou, dice
