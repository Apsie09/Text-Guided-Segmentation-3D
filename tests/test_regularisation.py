import torch

from src.features import swap_left_right
from src.models.decoder import TextSegDecoder


def test_swap_left_right():
    assert swap_left_right("left guy") == "right guy"
    assert swap_left_right("second from right") == "second from left"
    assert swap_left_right("Leftmost zebra, not the right one") == "Rightmost zebra, not the left one"
    assert swap_left_right("leftover pizza") == "leftover pizza"   # whole words only
    assert swap_left_right("man in red shirt") == "man in red shirt"


def test_dropout_active_in_train_off_in_eval():
    torch.manual_seed(0)
    m = TextSegDecoder(fusion="film", feat_drop=0.5, drop2d=0.5)
    feats, text = torch.randn(2, 4, 196, 768), torch.randn(2, 512)
    m.train()
    assert not torch.equal(m(feats, text), m(feats, text))
    m.eval()
    assert torch.equal(m(feats, text), m(feats, text))


def test_old_checkpoint_loads_into_regularised_model():
    plain = TextSegDecoder(fusion="film")
    reg = TextSegDecoder(fusion="film", feat_drop=0.1, drop2d=0.1)
    reg.load_state_dict(plain.state_dict())  # dropout adds no parameters
