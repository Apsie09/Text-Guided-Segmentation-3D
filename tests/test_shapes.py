import pytest
import torch

from src.losses import seg_loss
from src.metrics import iou_dice
from src.models.decoder import FUSIONS, TextSegDecoder, count_trainable


@pytest.mark.parametrize("fusion", FUSIONS)
def test_output_shape_and_grad(fusion):
    m = TextSegDecoder(fusion=fusion)
    feats = torch.randn(2, 4, 196, 768)
    text = torch.randn(2, 512)
    out = m(feats, text)
    assert out.shape == (2, 1, 224, 224)
    seg_loss(out, torch.zeros_like(out)).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    assert 0.3e6 < count_trainable(m) < 3e6


def test_image_only_ignores_text():
    torch.manual_seed(0)
    m = TextSegDecoder(fusion="none").eval()
    feats = torch.randn(1, 4, 196, 768)
    assert torch.equal(m(feats, torch.randn(1, 512)), m(feats, torch.randn(1, 512)))


def test_text_changes_output_after_film_is_trained():
    torch.manual_seed(0)
    m = TextSegDecoder(fusion="film").eval()
    for p in m.film1.parameters():  # break the identity init
        torch.nn.init.normal_(p, std=0.1)
    feats = torch.randn(1, 4, 196, 768)
    assert not torch.allclose(m(feats, torch.randn(1, 512)), m(feats, torch.randn(1, 512)))


def test_metrics_perfect_and_disjoint():
    t = torch.zeros(2, 1, 8, 8)
    t[0, 0, :4] = 1
    t[1, 0, :, :4] = 1
    perfect = torch.where(t > 0, 10.0, -10.0)
    iou, dice = iou_dice(perfect, t)
    assert torch.allclose(iou, torch.ones(2)) and torch.allclose(dice, torch.ones(2))
    iou, _ = iou_dice(-perfect, t)
    assert torch.allclose(iou, torch.zeros(2))
