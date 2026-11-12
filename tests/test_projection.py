"""Pin down Open3D's back-projection conventions against the pinhole formulas we document.

    Z = d / depth_scale,  X = (u - cx) Z / fx,  Y = (v - cy) Z / fy
    P_world = inverse(extrinsic) @ [X, Y, Z, 1]   (Open3D extrinsic = world -> camera)
"""
import numpy as np
import open3d as o3d

from src.visualize_3d import rgbd_to_pcd

W, H = 64, 48
FX, FY, CX, CY = 50.0, 55.0, 31.5, 23.5


def intrinsic():
    return o3d.camera.PinholeCameraIntrinsic(W, H, FX, FY, CX, CY)


def pinhole(depth_mm: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """Reference implementation, row-major order over pixels with valid depth."""
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u] / scale
    return np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z], axis=1)


def random_depth(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.integers(500, 3500, size=(H, W)).astype(np.uint16)
    d[rng.random((H, W)) < 0.1] = 0  # missing depth must be skipped
    return d


def test_backprojection_matches_pinhole_formula():
    d = random_depth()
    pcd = rgbd_to_pcd(np.zeros((H, W, 3), np.uint8), d, intrinsic(), np.eye(4))
    pts = np.asarray(pcd.points)
    ref = pinhole(d)
    assert pts.shape == ref.shape
    np.testing.assert_allclose(pts, ref, atol=1e-5)


def test_principal_point_lies_on_optical_axis():
    d = np.zeros((H, W), np.uint16)
    # cx, cy = 31.5, 23.5 sit between pixels; the 4 surrounding pixels are symmetric about the axis
    d[23:25, 31:33] = 2000
    pts = np.asarray(rgbd_to_pcd(np.zeros((H, W, 3), np.uint8), d, intrinsic(), np.eye(4)).points)
    np.testing.assert_allclose(pts.mean(0), [0.0, 0.0, 2.0], atol=1e-6)


def test_extrinsic_is_world_to_camera():
    # camera located at world (1, 2, 3), rotated 90 deg about world Y
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T_world_cam[:3, 3] = [1.0, 2.0, 3.0]
    extrinsic = np.linalg.inv(T_world_cam)

    d = random_depth(1)
    pts = np.asarray(rgbd_to_pcd(np.zeros((H, W, 3), np.uint8), d, intrinsic(), extrinsic).points)
    ref_cam = pinhole(d)
    ref_world = (T_world_cam @ np.c_[ref_cam, np.ones(len(ref_cam))].T).T[:, :3]
    np.testing.assert_allclose(pts, ref_world, atol=1e-5)


def test_depth_truncation_drops_far_points():
    d = np.full((H, W), 1000, np.uint16)
    d[:, : W // 2] = 5000  # 5 m, beyond the 4 m default truncation
    pts = np.asarray(rgbd_to_pcd(np.zeros((H, W, 3), np.uint8), d, intrinsic(), np.eye(4)).points)
    assert len(pts) == H * (W - W // 2)
    assert np.allclose(pts[:, 2], 1.0)
