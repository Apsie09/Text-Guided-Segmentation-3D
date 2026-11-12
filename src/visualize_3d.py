"""Lift a text-conditioned 2D mask into 3D with calibrated RGB-D and render it (Open3D).

Pipeline per frame
    1. predict a mask from RGB + sentence (frozen CLIP + trained FiLM decoder)
    2. back-project every pixel with valid depth through the pinhole model (Open3D):
           Z = depth(u, v) / depth_scale
           X = (u - cx) * Z / fx
           Y = (v - cy) * Z / fy
    3. move camera-frame points to the world frame with the camera pose:
           P_world = T_world_camera @ [X, Y, Z, 1]^T,   T_world_camera = inverse(extrinsic)
       (Open3D's `extrinsic` is world -> camera, so it is passed as is)
    4. colour masked points red, the rest with their RGB colour; localise the target
       (centroid + axis-aligned box of the masked points, after outlier removal)

Default data: Open3D's SampleRedwoodRGBDImages (5 frames, 640x480, depth in mm,
PrimeSense intrinsics fx = fy = 525, cx = 319.5, cy = 239.5, camera poses in trajectory.log).

Usage
    pixi run viz3d --text "the chair"                       # frame 0, saves figures
    pixi run viz3d --text "the chair" --frames 0 1 2 3 4    # multi-view consistency check
    pixi run viz3d --text "the window" --show               # open the interactive viewer
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage

from src.data.refcoco_m import PROJECT_ROOT
from src.predict import DEFAULT_RUN, Predictor

OUT_DIR = PROJECT_ROOT / "outputs" / "graphics"
RED = np.array([1.0, 0.1, 0.1])


# --------------------------------------------------------------------------- data


def load_redwood():
    d = o3d.data.SampleRedwoodRGBDImages()
    intr = o3d.io.read_pinhole_camera_intrinsic(d.camera_intrinsic_path)
    traj = o3d.io.read_pinhole_camera_trajectory(d.trajectory_log_path)
    frames = []
    for i, (c, z) in enumerate(zip(d.color_paths, d.depth_paths)):
        frames.append({
            "name": f"redwood_{i:02d}",
            "color": np.asarray(o3d.io.read_image(c)),           # [H, W, 3] uint8
            "depth": np.asarray(o3d.io.read_image(z)),           # [H, W] uint16, millimetres
            "extrinsic": traj.parameters[i].extrinsic,           # world -> camera (4x4)
        })
    return frames, intr


# --------------------------------------------------------------------------- geometry


def rgbd_to_pcd(color: np.ndarray, depth: np.ndarray, intr, extrinsic,
                depth_scale: float = 1000.0, depth_trunc: float = 4.0) -> o3d.geometry.PointCloud:
    """Back-project an RGB-D frame to a world-frame coloured point cloud (Open3D)."""
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr, extrinsic)


def clean_mask(mask: np.ndarray, min_component_frac: float = 0.1) -> np.ndarray:
    """Drop small disconnected blobs: keep 2D components >= frac x the largest one."""
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= min_component_frac * sizes.max()) + 1
    return np.isin(labels, keep)


def target_cloud(depth, mask, intr, extrinsic, erode_px: int = 3):
    """World-frame points of the masked region only.

    1. small disconnected mask blobs are dropped in 2D (`clean_mask`), so stray false
       positives elsewhere in the image do not distort the localisation;
    2. the mask is eroded: pixels on an object's silhouette often take the depth of the
       background behind it (depth discontinuity), which would smear "flying" points;
    3. remaining stragglers are removed with a statistical outlier filter.
    """
    m = clean_mask(mask)
    m = ndimage.binary_erosion(m, iterations=erode_px) if erode_px else m
    d = np.where(m, depth, 0).astype(depth.dtype)
    gray = np.zeros((*depth.shape, 3), np.uint8)
    pcd = rgbd_to_pcd(gray, d, intr, extrinsic).voxel_down_sample(0.01)
    if len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def tint(color: np.ndarray, mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    out = color.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * RED
    return (out * 255).astype(np.uint8)


def camera_geometries(intr, extrinsic, scale: float = 0.25):
    """Camera frustum (LineSet) + camera axes placed at the camera pose in world space."""
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        intr.width, intr.height, intr.intrinsic_matrix, extrinsic, scale=scale)
    frustum.paint_uniform_color([0.1, 0.3, 0.9])
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    axes.transform(np.linalg.inv(extrinsic))  # T_world_camera
    return [frustum, axes]


# --------------------------------------------------------------------------- output


def save_2d_figure(frame, prob, mask, text, path: Path) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.9))
    ax[0].imshow(frame["color"]); ax[0].set_title("RGB")
    im = ax[1].imshow(np.where(frame["depth"] > 0, frame["depth"] / 1000.0, np.nan), cmap="turbo")
    ax[1].set_title("depth [m] (black = missing)"); fig.colorbar(im, ax=ax[1], fraction=0.035)
    ax[2].imshow(prob, cmap="magma", vmin=0, vmax=1); ax[2].set_title("P(target | text)")
    ax[3].imshow(tint(frame["color"], mask)); ax[3].set_title("mask (p > 0.5)")
    for a in ax:
        a.axis("off")
    fig.suptitle(f'"{text}"  —  {frame["name"]}')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_offscreen(geoms, path: Path, front, lookat, up, zoom: float) -> bool:
    """Screenshot via a hidden Open3D window. Returns False if no display is available."""
    try:
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(visible=False, width=1280, height=900):
            return False
        for g in geoms:
            vis.add_geometry(g)
        vc = vis.get_view_control()
        vc.set_front(front); vc.set_lookat(lookat); vc.set_up(up); vc.set_zoom(zoom)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([1.0, 1.0, 1.0])
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(str(path), do_render=True)
        vis.destroy_window()
        return True
    except Exception as e:  # headless machines, missing GL, ...
        print(f"  (offscreen render skipped: {e})")
        return False


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show", action="store_true", help="open the interactive Open3D viewer")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, intr = load_redwood()
    predictor = Predictor(args.run)
    tag = slug(args.text)

    geoms, centroids = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)], []
    fused = o3d.geometry.PointCloud()
    print(f'text: "{args.text}"   intrinsics: fx={intr.intrinsic_matrix[0, 0]:.1f} '
          f"fy={intr.intrinsic_matrix[1, 1]:.1f} cx={intr.intrinsic_matrix[0, 2]:.1f} "
          f"cy={intr.intrinsic_matrix[1, 2]:.1f}  ({intr.width}x{intr.height})")

    for fi in args.frames:
        f = frames[fi]
        prob = predictor.predict(Image.fromarray(f["color"]), args.text)
        mask = prob > args.threshold
        save_2d_figure(f, prob, mask, args.text, OUT_DIR / f"{tag}_{f['name']}_2d.png")

        scene = rgbd_to_pcd(tint(f["color"], mask), f["depth"], intr, f["extrinsic"])
        target = target_cloud(f["depth"], mask, intr, f["extrinsic"])
        valid = (f["depth"] > 0) & mask
        line = (f"frame {fi}: mask {mask.mean():6.1%} of pixels ({valid.sum()} with depth) | "
                f"scene {len(scene.points):,} pts | target {len(target.points):,} pts")
        if len(target.points):
            c = np.asarray(target.points).mean(0)
            centroids.append(c)
            fused += target
            line += f" | centroid (world, m) [{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}]"
            line += f" | box extent {np.round(target.get_axis_aligned_bounding_box().get_extent(), 2)} m"
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            ball.translate(c); ball.paint_uniform_color([1.0, 0.8, 0.0])
            geoms.append(ball)
        print(line)
        geoms += [scene, *camera_geometries(intr, f["extrinsic"])]
        o3d.io.write_point_cloud(str(OUT_DIR / f"{tag}_{f['name']}.ply"), scene)

    if len(fused.points):
        box = fused.get_axis_aligned_bounding_box()
        box.color = (1.0, 0.0, 0.0)
        geoms.append(box)
        print(f"target box (all frames, world axes): center {np.round(box.get_center(), 3)} m, "
              f"extent {np.round(box.get_extent(), 2)} m")

    if len(centroids) > 1:
        C = np.stack(centroids)
        spread = np.linalg.norm(C - C.mean(0), axis=1)
        print(f"multi-view consistency: target centroid spread across {len(C)} frames = "
              f"mean {spread.mean() * 100:.1f} cm, max {spread.max() * 100:.1f} cm "
              "(same object, different camera poses -> should be small if poses and masks agree)")

    # a fixed 3/4 view looking at the scene from above-behind the first camera
    cam_world = np.linalg.inv(frames[args.frames[0]]["extrinsic"])
    lookat = centroids[0] if centroids else cam_world[:3, 3] + cam_world[:3, 2] * 2.0
    front = -cam_world[:3, 2] - 0.6 * cam_world[:3, 1] + 0.5 * cam_world[:3, 0]
    up = -cam_world[:3, 1]
    shot = OUT_DIR / f"{tag}_3d.png"
    if render_offscreen(geoms, shot, front, lookat, up, zoom=0.35):
        print(f"saved {shot.relative_to(PROJECT_ROOT)}")
    print(f"saved 2D figures and .ply clouds to {OUT_DIR.relative_to(PROJECT_ROOT)}/")

    if args.show:
        o3d.visualization.draw_geometries(
            geoms, window_name=f'"{args.text}"', width=1280, height=900,
            front=front.tolist(), lookat=np.asarray(lookat).tolist(), up=up.tolist(), zoom=0.35)


if __name__ == "__main__":
    main()
