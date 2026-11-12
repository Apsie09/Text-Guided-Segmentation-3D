import numpy as np
import pytest
from pycocotools import mask as mask_utils

from src.data.refcoco_m import RefCocoM, decode_mask, load_raw, resize_pair
from PIL import Image


def test_rle_roundtrip():
    m = np.zeros((40, 60), dtype=np.uint8)
    m[5:20, 10:45] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("ascii")  # HF stores counts as str
    assert np.array_equal(decode_mask(rle), m)


def test_resize_pair_keeps_geometry():
    img = Image.new("RGB", (400, 200))
    m = np.zeros((200, 400), dtype=np.uint8)
    m[50:150, 100:200] = 1  # x in [100,200) of 400, y in [50,150) of 200
    _, mr = resize_pair(img, m, 224)
    assert set(np.unique(mr)) <= {0, 1}
    ys, xs = np.nonzero(mr)
    # expected box in 224 space: x [56,112), y [56,168)
    assert abs(xs.min() - 56) <= 1 and abs(xs.max() - 111) <= 1
    assert abs(ys.min() - 56) <= 1 and abs(ys.max() - 167) <= 1


@pytest.fixture(scope="module")
def ds():
    return load_raw()


def test_real_masks_match_image_size_and_bbox(ds):
    rng = np.random.default_rng(0)
    for row in rng.choice(len(ds), size=50, replace=False):
        rec = ds[int(row)]
        for ann in rec["samples"]:
            m = decode_mask(ann["mask"])
            assert m.shape == (rec["image"].height, rec["image"].width)
            assert m.sum() > 0
            ys, xs = np.nonzero(m)
            cx, cy = xs.mean(), ys.mean()
            x, y, w, h = ann["bbox"]
            # mask centroid must lie inside the stored COCO [x, y, w, h] box
            assert x - 2 <= cx <= x + w + 2 and y - 2 <= cy <= y + h + 2


def test_dataset_item_shapes(ds):
    d = RefCocoM("val", ds=ds)
    item = d[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["mask"].shape == (1, 224, 224)
    assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}
    assert isinstance(item["text"], str) and item["text"]
