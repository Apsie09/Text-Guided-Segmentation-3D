# 3D module: from a text-conditioned mask to a localised point cloud

Code: `src/visualize_3d.py` · Tests: `tests/test_projection.py` · Run: `pixi run viz3d --text "..."`

The segmentation model works in 2D. This module lifts its mask into 3D using a calibrated
RGB-D camera, places it in a common world frame and renders it. Back-projection, the world
transform and rendering use Open3D's implementations. This document states the math they
implement. The tests check that Open3D follows exactly these formulas and conventions.

## 1. Data

Open3D `SampleRedwoodRGBDImages`: 5 frames of a rendered living room, 640×480, with:

- **colour** `I(u, v)` (8-bit RGB),
- **depth** `d(u, v)` in millimetres (16-bit; 0 = no measurement, 13% of pixels),
- **intrinsics** (PrimeSense): `fx = fy = 525.0`, `cx = 319.5`, `cy = 239.5`,
- **camera poses** for every frame (`trajectory.log`).

Depth is aligned to colour, so pixel `(u, v)` in both images sees the same surface point.

## 2. Pinhole camera model

A camera-frame point `P_c = (X, Y, Z)` with `Z > 0` (in front of the camera) projects to pixel

```
u = fx · X / Z + cx
v = fy · Y / Z + cy
```

or, in homogeneous form, `λ [u, v, 1]ᵀ = K P_c` with

```
      ⎡ fx   0  cx ⎤
K  =  ⎢  0  fy  cy ⎥
      ⎣  0   0   1 ⎦
```

- `fx, fy` are the focal length in pixel units.
- `(cx, cy)` is the principal point, where the optical axis meets the image.
- Camera axes follow the OpenCV/Open3D convention: **x right, y down, z forward**.

## 3. Back-projection (inverse projection)

Projection loses depth, but an RGB-D camera measures it. For each pixel with a valid depth sample:

```
Z = d(u, v) / s            s = depth_scale = 1000   (mm → m)
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
```

This is `P_c = Z · K⁻¹ [u, v, 1]ᵀ`. Pixels with `d = 0` and points beyond `depth_trunc = 4 m` are
skipped. Implemented by `o3d.geometry.PointCloud.create_from_rgbd_image`.

## 4. Camera → world transform

Each frame has a rigid pose `T_wc ∈ SE(3)` (rotation `R`, camera centre `t`):

```
          ⎡ R  t ⎤
T_wc  =   ⎣ 0  1 ⎦ ,      P_w = T_wc · [X, Y, Z, 1]ᵀ = R P_c + t
```

Open3D stores the **extrinsic** matrix as the inverse, world → camera:
`E = T_wc⁻¹ = [Rᵀ | −Rᵀ t]`. Passing `E` to `create_from_rgbd_image` returns points already in
world coordinates. With all frames in one world frame, clouds from different viewpoints line up.

## 5. Mask → 3D target

1. **Mask.** The model returns `p(u, v) ∈ [0, 1]` at 224×224. It is resized bilinearly to 640×480
   and thresholded at 0.5.
2. **Remove stray blobs.** Keep 2D connected components with at least 10% of the largest one's
   area. False-positive specks elsewhere in the image would otherwise drag the 3D localisation.
3. **Erode by 3 px.** At an object's silhouette the depth sensor often returns the background
   behind it (*depth discontinuity*). Back-projected, those pixels become "flying" points between
   object and wall.
4. **Back-project** only the remaining masked pixels (§3–4), voxel-downsample to 1 cm and apply a
   statistical outlier filter (20 neighbours, 2σ).
5. **Localise.** The target is reported by:
   - its centroid `c = (1/N) Σ P_w,i`,
   - its axis-aligned bounding box in world axes.

## 6. Rendering

The whole scene cloud is coloured from RGB, with masked pixels blended to red (α = 0.65). It is
drawn together with:

- the world origin axes,
- each camera's **viewing frustum** (`LineSet.create_camera_visualization(K, E)`): the image
  rectangle back-projected at a fixed depth and joined to the camera centre,
- the camera axes placed at `T_wc`,
- a yellow sphere at each frame's target centroid,
- the red bounding box of all frames' target points.

`pixi run viz3d … --show` opens the interactive viewer. A fixed ¾ view is also saved as an
image (`outputs/graphics/<text>_3d.png`).

## 7. Checks

**`tests/test_projection.py`** runs on synthetic depth with non-square pixels (`fx ≠ fy`) and an
off-centre principal point. It verifies that Open3D:

- reproduces §3 exactly for every pixel, including skipping missing depth,
- maps the principal point onto the optical axis `(0, 0, Z)`,
- treats `extrinsic` as world → camera, i.e. returns `T_wc · P_c` (tested with a rotated,
  translated camera),
- drops points beyond `depth_trunc`.

**Multi-view consistency** checks the poses and masks on real data. The chair is a single static
object, so its centroid computed from 5 different camera poses should land at the same world point.

Final model (FiLM, no coordinate channels, flip augmentation, `film_c128_nocoord_dice1_flip_s0`):

| text | frames | centroid spread (mean / max) | target box (world axes) |
|---|---|---|---|
| "the chair" | 0–4 | 3.0 cm / 4.4 cm | 0.94 × 0.72 × 0.68 m |
| "white armchair" | 0–4 | 3.0 cm / 4.4 cm | 0.94 × 0.61 × 0.92 m |

For comparison, the model without flip augmentation gave 4.4 cm / 5.2 cm for "the chair".

This spread comes mostly from the masks covering slightly different parts of the chair in each
frame, not from pose error: the five scene clouds overlap without visible double outlines.

## 8. Limitations

- **Domain shift.** The model was trained on COCO photographs. Redwood is a synthetic render.
  With the final model the chair mask covers mainly the seat cushion. It misses most of the
  backrest and the thin metal frame, so the box underestimates the chair's height.
- **Background regions cannot be named.** RefCOCO only annotates discrete objects. "The window"
  and "the curtain" still highlight the chair, and "the floor" lands on unrelated blobs. The model
  learned that the answer is always an object.
- **Single-view geometry.** Each cloud contains only surfaces visible from that camera. Hidden and
  back faces are absent, so a per-frame box underestimates the object.
- **Axis-aligned box.** The box follows the world axes, not the object's orientation, so it
  over-estimates the extent of rotated objects.
- This is **3D localisation and visualisation**, not grasp planning or a manipulation result.
