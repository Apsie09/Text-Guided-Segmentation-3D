# text-guided-segmentation-3d

Language-conditioned image segmentation on **frozen CLIP features** with a small trainable
decoder, plus **RGB-D back-projection** of the predicted mask into a 3D point cloud.

Given an RGB image and a sentence such as *"second bowl from right"*, the model predicts a
pixel mask of the referred region. With aligned depth and camera intrinsics, the mask is lifted
to 3D and rendered as a highlighted point cloud.

> Computer Graphics course project. **Status: in progress.** No results are reported yet.

## Scope

- **Task:** referring-expression segmentation (an object named explicitly by the text).
  This is a first step toward robotic affordance grounding, **not** affordance grounding itself.
- **Data:** [RefCOCO-M](https://huggingface.co/datasets/moondream/refcoco-m) (MIT),
  1,190 images / 2,080 masks / 5,598 sentences, with a custom split by image
  (train 833 / val 178 / test 179 images, seed 0, in `splits/`).
  RefCOCO-M only ships a `validation` split, so all numbers here are **internal course
  experiments, not official benchmark results**.
- **Frozen encoder:** [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16).

## Models

| ID | Text | Fusion | Purpose |
|---|---|---|---|
| M0 | no | none | Image-only baseline: does text help at all? |
| M1 | yes | FiLM | Main model: text embedding modulates CLIP patch features |
| M1b | yes | concat | Fusion ablation |
| CLIPSeg | yes | pretrained | External reference (different pretraining, not a controlled ablation) |

Only the decoder is trained (target ~0.5–3M parameters). CLIP stays frozen.

## Graphics module

Back-projection of masked pixels with the pinhole model:

```
Z = depth(u, v) / 1000      X = (u - cx) * Z / fx      Y = (v - cy) * Z / fy
P_world = T_world_camera @ [X, Y, Z, 1]^T
```

Back-projection, the world transform and rendering use Open3D. `tests/test_projection.py`
checks that Open3D follows exactly these formulas and conventions. Foreground points are drawn in
red, the rest in their RGB colours, together with camera frustums and the target's centroid and
bounding box. Full derivation, checks and limitations: [`report/graphics.md`](report/graphics.md).

## Setup

Requires [pixi](https://pixi.sh) and an NVIDIA GPU with a CUDA 12 driver
(developed on an RTX 4060 Laptop, 8 GB).

```bash
pixi install                # Python 3.12 + PyTorch (CUDA) + transformers + Open3D
pixi run check-gpu          # confirm CUDA is visible
pixi run smoke-test         # download CLIP/CLIPSeg, check feature shapes and VRAM
pixi run inspect-data       # dataset stats, split check, 20 overlays -> outputs/inspect/
pixi run test               # unit tests
pixi run cache-features     # frozen CLIP features for all images/sentences -> data/cache/
pixi run experiments        # train all variants x 3 seeds, compare on val (+ CLIPSeg reference)
pixi run viz3d --text "white armchair" --frames 0 1 2 3 4 --show   # 3D demo on Redwood RGB-D
```

## Layout

```
src/data/refcoco_m.py       dataset loading, RLE decoding, image-grouped split, torch Dataset
src/features.py             frozen CLIP wrapper and GPU feature cache
src/models/decoder.py       decoder with fusion = none | film | concat
src/train.py, evaluate.py   training (early stopping on val mIoU) and comparison of runs
src/clipseg_baseline.py     off-the-shelf CLIPSeg reference
src/predict.py              inference on any image + sentence
src/visualize_3d.py         RGB-D back-projection and Open3D rendering
scripts/                    download, smoke test, dataset inspection
splits/                     fixed train/val/test image ids (committed)
tests/                      split, mask-alignment, model-shape and projection tests
report/                     results tables, figures and the 3D module write-up
outputs/                    generated figures and checkpoints (not committed)
```

## Limitations

- RefCOCO-M has no depth or camera poses. The 3D demo uses separate RGB-D data.
- Monocular depth estimates (if used) are **relative**, not metric.
- A correct 3D mask is a localization/visualization result, not evidence of successful manipulation.
- CLIP and CLIPSeg may have seen COCO images during pretraining.
