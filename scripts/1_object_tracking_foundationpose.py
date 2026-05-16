"""FoundationPose-based multi-view object 6DoF pose tracking.

Alternative to scripts/1_object_tracking_local.py (silhouette diff-render) and
scripts/1_object_tracking_marker.py (ArUco). For each timestamp we run the
FoundationPose tracker once per camera, compose each per-view (cam_T_obj) into
a world pose using the COLMAP extrinsics, and fuse the per-view world poses
into a single multi-view-consistent pose.

Scale handling (the mesh and the COLMAP world are both up-to-scale).

  We do NOT assume the input mesh is in metric meters. Before handing the
  mesh to FoundationPose we re-center it at the origin and rescale it so its
  longest AABB side equals NORMALIZED_MESH_EXTENT (0.15). FP then operates in
  a "fake-metric" world where 1 unit ≡ 1 meter and the mesh is 0.15 m long.
  To make depth live in the same units we scale COLMAP depth by α, where α is
  the (fake) m/COLMAP-unit ratio. Because the mesh size is now fixed at 0.15,
  the silhouette-vs-depth heuristic
      α = NORMALIZED_MESH_EXTENT * f / (pixel_extent * z_colmap)
  is exactly the scale that makes the normalized mesh's projection match the
  observed silhouette at the observed depth — i.e. it implicitly fits the
  object's absolute COLMAP-world size from the image data, with no metric
  prior on the mesh. α is estimated at frame 0 and then frozen.

Output schema matches scripts/1_object_tracking_local.py exactly so the
downstream axis_align step is unchanged. With the normalized mesh, the saved
quantities collapse to:
    saved_scale     = 1 / α
    pose_save[R]    = R_wo^T          (PyTorch3D row-vector convention)
    pose_save[t]    = t_wo_fake / α
so axis_align reproduces world_pt_colmap = world_pt_fake_metric / α.
"""

import argparse
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "foundationpose"))

import src.utils.colmap_utils as colmap_utils  # noqa: E402

# FP imports — done after sys.path tweak.
import nvdiffrast.torch as dr  # noqa: E402
from foundationpose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor  # noqa: E402


# Target longest AABB side for the (centered) normalized mesh, in "fake-metric"
# meters. Must match utils/axis_align.py:NORMALIZED_MESH_EXTENT so the consumer
# normalizes the same way.
NORMALIZED_MESH_EXTENT = 0.15

SEQ_MASK_SUBPATH = "outputs/sam3"
SEQ_PARSED_SUBPATH = "parsed"
SEQ_CALIB_SUBPATH = "calib"
SEQ_DEPTH_SUBPATH = "outputs/da3"
SEQ_OUTPUT_SUBPATH = "outputs/object_pose/foundationpose"

FP_WEIGHTS_DIR = PROJECT_ROOT / "foundationpose" / "weights"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-view FoundationPose tracking.")
    p.add_argument("--seq-root", type=Path, required=True)
    p.add_argument("--mesh-path", type=Path, required=True,
                   help="Textured .obj/.glb mesh in any consistent units. It is "
                        "centered + rescaled in-place to longest side = "
                        f"{NORMALIZED_MESH_EXTENT} m before FP inference.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--timestamp-start", type=int, default=0)
    p.add_argument("--timestamp-end", type=int, default=-1)
    p.add_argument("--timestamp-step", type=int, default=1)
    p.add_argument("--max-timestamps", type=int, default=-1)
    p.add_argument("--max-views", type=int, default=8,
                   help="Top-K views per frame (sorted by mask area). -1 for all.")
    p.add_argument("--min-mask-pixels", type=int, default=200)
    p.add_argument("--est-refine-iter", type=int, default=5,
                   help="FoundationPose register refine iterations (first frame).")
    p.add_argument("--track-refine-iter", type=int, default=2,
                   help="FoundationPose track_one refine iterations (subsequent frames).")
    p.add_argument("--depth-extr-inv", action="store_true",
                   help="DA3 extrinsics are cam_from_world; invert to get world_from_cam (matches "
                        "scripts/1_object_tracking_local.py).")
    p.add_argument("--alpha-fix", type=float, default=None,
                   help="Skip α estimation, fix to this value. α has units "
                        "of fake-metric-m / COLMAP-unit, where 1 fake-metric m "
                        f"corresponds to {NORMALIZED_MESH_EXTENT} m of the "
                        "normalized mesh's longest side.")
    p.add_argument("--alpha-views-min", type=int, default=4,
                   help="Min views to use when estimating α at frame 0.")
    p.add_argument("--depth-clip-max", type=float, default=4.0,
                   help="Clip depth values larger than this (post-α scaling, in meters).")
    p.add_argument("--down", type=float, default=2.0,
                   help="Downsample factor for collage tiles (rendering uses depth resolution).")
    p.add_argument("--max-collage-tiles", type=int, default=16)
    p.add_argument("--kp3d-reproj-thresh", type=float, default=50.0,
                   help="Exclude cameras whose kp3d_reproj_error in image_confidence.json exceeds this.")
    p.add_argument("--fp-debug", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p


# ── Mask / depth IO ────────────────────────────────────────────────────────────
def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.exists() or mask_path.stat().st_size <= 22:
        return None
    with zipfile.ZipFile(mask_path, "r") as archive:
        keys = sorted(archive.namelist())
        if not keys:
            return None
        masks = [np.load(archive.open(k)).astype(bool) for k in keys]
    return np.any(np.stack(masks, axis=0), axis=0)


_compressed_depth_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]] = {}


def load_compressed_depths(depth_root: Path):
    """Decode all depth_<cam>.mkv into (V, T, H_d, W_d) float32. Cached per depth_root."""
    key = str(depth_root.resolve())
    if key in _compressed_depth_cache:
        return _compressed_depth_cache[key]

    compressed_dir = depth_root / "depths_compressed"
    calib_npz = compressed_dir / "calib.npz"
    cam_names_path = compressed_dir / "cam_names.txt"
    depth_range_path = compressed_dir / "depth_range.npy"
    if not (calib_npz.exists() and cam_names_path.exists() and depth_range_path.exists()):
        raise FileNotFoundError(f"Compressed depth store missing at {compressed_dir}")

    import av  # noqa: WPS433

    calib = np.load(calib_npz)
    intrinsics = np.asarray(calib["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(calib["extrinsics"], dtype=np.float32)
    depth_range = np.load(depth_range_path).astype(np.float32)
    with cam_names_path.open() as f:
        cam_stems = [line.strip() for line in f if line.strip()]

    decoded: List[Optional[np.ndarray]] = []
    for vi, stem in enumerate(cam_stems):
        mkv = compressed_dir / f"depth_{stem}.mkv"
        if not mkv.exists():
            decoded.append(None)
            continue
        d_min, d_max = float(depth_range[vi, 0]), float(depth_range[vi, 1])
        with av.open(str(mkv)) as container:
            frames = [frame.to_ndarray() for frame in container.decode(video=0)]
        if not frames:
            decoded.append(None)
            continue
        arr = np.stack(frames, axis=0).astype(np.float32)
        decoded.append(arr / 4095.0 * (d_max - d_min) + d_min)

    valid = [d for d in decoded if d is not None]
    if not valid:
        raise RuntimeError(f"No depth frames decoded from {compressed_dir}")
    T = max(d.shape[0] for d in valid)
    H, W = valid[0].shape[1:]
    depths = np.zeros((len(cam_stems), T, H, W), dtype=np.float32)
    for vi, d in enumerate(decoded):
        if d is not None:
            depths[vi, : d.shape[0]] = d

    result = (depths, intrinsics, extrinsics, cam_stems)
    _compressed_depth_cache[key] = result
    return result


# ── Camera loading (COLMAP RDF, world-to-cam) ──────────────────────────────────
def load_cameras_rdf(calib_root: Path) -> List[Dict]:
    cameras, images, _ = colmap_utils.read_model(str(calib_root), ext=".bin")
    out = []
    for image in images.values():
        camera = cameras[image.camera_id]
        params = camera.params
        fx, fy, cx, cy = map(float, params[:4])
        dist = np.zeros(5, dtype=np.float64)
        if camera.model == "OPENCV" and len(params) >= 8:
            dist[:4] = params[4:8]
        elif camera.model == "RADIAL" and len(params) >= 5:
            dist[0] = params[3]
            dist[1] = params[4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        R_w2c = colmap_utils.qvec2rotmat(image.qvec).astype(np.float64)
        t_w2c = np.asarray(image.tvec, dtype=np.float64)
        out.append({
            "cam_name": Path(image.name).stem,
            "W": int(camera.width),
            "H": int(camera.height),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "K": K,
            "dist": dist,
            "R_w2c": R_w2c,  # COLMAP convention: P_cam = R × P_world + t
            "t_w2c": t_w2c,  # COLMAP units (potentially up-to-scale)
        })
    return sorted(out, key=lambda info: info["cam_name"])


def filter_bad_cameras(cams: List[Dict], calib_root: Path, threshold: float) -> List[Dict]:
    if threshold <= 0:
        return cams
    img_conf_path = calib_root / "image_confidence.json"
    if not img_conf_path.exists():
        return cams
    with open(img_conf_path, "r", encoding="utf-8") as f:
        img_conf = json.load(f)
    bad = {
        Path(k).stem
        for k, v in img_conf.items()
        if "kp3d_reproj_error" in v and (
            v["kp3d_reproj_error"] < 0 or v["kp3d_reproj_error"] > threshold
        )
    }
    if bad:
        print(f"Excluding {len(bad)} cameras with kp3d_reproj_error > {threshold}px: {sorted(bad)}")
    return [c for c in cams if c["cam_name"] not in bad]


# ── Sequence helpers ───────────────────────────────────────────────────────────
def sorted_timestamps(parsed_root: Path) -> List[str]:
    return sorted(
        [p.name for p in parsed_root.iterdir() if p.is_dir() and p.name.startswith("timestamp_")],
        key=lambda name: int(name.split("_")[-1]),
    )


def read_rgb(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


# ── Mesh normalization ─────────────────────────────────────────────────────────
def normalize_mesh_in_place(mesh: trimesh.Trimesh) -> float:
    """Center mesh at origin and rescale so longest AABB side = NORMALIZED_MESH_EXTENT.

    Returns the longest extent of the *raw* mesh (pre-normalization) so callers can
    log it. After this call, mesh.bounding_box is centered at origin with longest
    side ≈ NORMALIZED_MESH_EXTENT.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if v.size == 0:
        raise ValueError("Mesh has no vertices.")
    aabb_min, aabb_max = v.min(axis=0), v.max(axis=0)
    center = 0.5 * (aabb_min + aabb_max)
    extent_raw = float((aabb_max - aabb_min).max())
    if extent_raw <= 0:
        raise ValueError("Mesh AABB is degenerate (extent <= 0).")
    mesh.apply_translation(-center)
    mesh.apply_scale(NORMALIZED_MESH_EXTENT / extent_raw)
    return extent_raw


# ── α estimation ───────────────────────────────────────────────────────────────
def estimate_alpha_heuristic(
    per_view_obs: List[Dict],
) -> Tuple[float, List[float]]:
    """α (fake-metric m / colmap-unit) ≈ NORMALIZED_MESH_EXTENT × f / (pixel_extent × z_colmap).

    This is purely a silhouette-vs-depth scale estimator: given that the mesh
    has been normalized to longest side = NORMALIZED_MESH_EXTENT, α is the
    fake-metric scale that makes the normalized mesh project to the observed
    silhouette at the observed depth. The pipeline never needs the true metric
    object size; only consistency between mesh and depth in FP's input space.

    per_view_obs: list of dicts with {fx, fy, mask, depth} (depth in COLMAP units).
    Returns (median α, list of per-view estimates).
    """
    estimates: List[float] = []
    for obs in per_view_obs:
        mask = obs["mask"]
        depth = obs["depth"]
        ys, xs = np.nonzero(mask)
        if ys.size < 100:
            continue
        valid = mask & (depth > 0)
        if valid.sum() < 50:
            continue
        z_med = float(np.median(depth[valid]))
        pix_w = int(xs.max() - xs.min())
        pix_h = int(ys.max() - ys.min())
        pix_ext = max(pix_w, pix_h)
        if pix_ext < 4:
            continue
        f = 0.5 * (obs["fx"] + obs["fy"])
        estimates.append(NORMALIZED_MESH_EXTENT * f / (pix_ext * z_med))
    if not estimates:
        return 1.0, estimates
    return float(np.median(estimates)), estimates


# ── World-pose composition + multi-view fusion ─────────────────────────────────
def compose_cam_to_world(R_w2c: np.ndarray, t_w2c_colmap: np.ndarray, alpha: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_cw, t_cw_metric) where world_pt_metric = R_cw × point_cam_metric + t_cw_metric.

    COLMAP gives P_cam = R_w2c × P_world + t_w2c (P_world in COLMAP units, t_w2c in COLMAP units).
    Camera-to-world: P_world = R_w2c^T × (P_cam - t_w2c) = R_w2c^T × P_cam - R_w2c^T × t_w2c.
    Convert to metric world: multiply translation component by α.
    """
    R_cw = R_w2c.T
    t_cw_colmap = -R_cw @ t_w2c_colmap
    return R_cw, alpha * t_cw_colmap


def world_pose_from_view(R_co: np.ndarray, t_co: np.ndarray,
                         R_w2c: np.ndarray, t_w2c: np.ndarray, alpha: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    R_cw, t_cw_metric = compose_cam_to_world(R_w2c, t_w2c, alpha)
    R_wo = R_cw @ R_co
    t_wo_metric = R_cw @ t_co + t_cw_metric
    return R_wo, t_wo_metric


def chordal_mean_rotation(R_list: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted chordal mean: SVD-project of weighted rotation sum."""
    A = np.zeros((3, 3), dtype=np.float64)
    for R, w in zip(R_list, weights):
        A += w * R
    U, _, Vt = np.linalg.svd(A)
    D = np.eye(3)
    D[2, 2] = float(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def fuse_world_poses(
    per_view: List[Dict], weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """per_view dicts must contain 'R_wo' (3,3) and 't_wo_metric' (3,)."""
    n = len(per_view)
    if n == 0:
        raise ValueError("Cannot fuse an empty per-view list.")
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    R_fused = chordal_mean_rotation([v["R_wo"] for v in per_view], weights)
    t_fused = sum(w * v["t_wo_metric"] for w, v in zip(weights, per_view))
    spread = float(np.std(np.stack([v["t_wo_metric"] for v in per_view], 0), axis=0).mean())
    return R_fused, t_fused, spread


# ── Output schema (axis_align consumer) ────────────────────────────────────────
def to_consumer_schema(R_wo_fake_metric: np.ndarray, t_wo_fake_metric: np.ndarray,
                       alpha: float) -> Tuple[np.ndarray, float]:
    """Convert FP's fake-metric world pose (for the already-normalized mesh) into
    (pose_save, scale) so axis_align reconstructs world_pt_colmap via:
        world_pt_colmap = pose_save[:3,:3].T × (scale × normalize_in_place(v_raw)) + pose_save[:3,3]

    Here FP saw verts_norm = (v_raw - center_raw) × NORMALIZED_MESH_EXTENT / extent_raw
    (we centered + rescaled the mesh in main()), and produced:
        world_pt_fake = R_wo × verts_norm + t_wo_fake.
    We want world_pt_colmap = world_pt_fake / α, which matches axis_align's form
    when R_consumer = R_wo, saved_scale = 1/α, t_consumer = t_wo_fake / α.
    (The extent_raw factor inside axis_align's normalize and the one inside
    verts_norm cancel, so we don't need extent_raw or center_raw here.)
    """
    pose_save = np.eye(4, dtype=np.float64)
    pose_save[:3, :3] = R_wo_fake_metric.T
    pose_save[:3, 3] = t_wo_fake_metric / alpha
    saved_scale = float(1.0 / alpha)
    return pose_save, saved_scale


# ── Per-frame view bundle ──────────────────────────────────────────────────────
def collect_view_bundle(
    timestamp_name: str,
    parsed_root: Path,
    mask_root: Path,
    depth_root: Path,
    camera_infos: Sequence[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    """Return per-camera bundle of (rgb, mask, depth_colmap, K_at_depth_res) at this timestamp.

    All channels are resampled to depth resolution so a single K matches them.
    """
    depths_all, da3_intrs, da3_extrs, da3_stems = load_compressed_depths(depth_root)
    try:
        ts_idx = int(timestamp_name.split("_")[-1])
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse timestamp index from '{timestamp_name}'") from exc
    if ts_idx < 0 or ts_idx >= depths_all.shape[1]:
        raise RuntimeError(
            f"Timestamp idx {ts_idx} out of range [0,{depths_all.shape[1]}) for {depth_root}/depths_compressed"
        )
    da3_depth_t = depths_all[:, ts_idx]  # (V, H_d, W_d)

    da3_idx_by_stem = {stem: i for i, stem in enumerate(da3_stems)}

    image_dir = parsed_root / timestamp_name / "images"
    mask_dir = mask_root / timestamp_name / "object_masks"

    bundle: List[Dict] = []
    for cam in camera_infos:
        cam_name = cam["cam_name"]
        rgb_path = image_dir / f"{cam_name}.jpg"
        mask_path = mask_dir / f"{cam_name}.npz"
        if not rgb_path.exists() or not mask_path.exists():
            continue
        if cam_name not in da3_idx_by_stem:
            continue
        di = da3_idx_by_stem[cam_name]
        depth = da3_depth_t[di]  # (H_d, W_d)
        H_d, W_d = depth.shape
        if H_d <= 0 or W_d <= 0:
            continue

        mask = load_mask(mask_path)
        if mask is None:
            continue
        rgb = read_rgb(rgb_path)

        # Resize rgb and mask to depth resolution.
        rgb_d = cv2.resize(rgb, (W_d, H_d), interpolation=cv2.INTER_AREA)
        mask_d = cv2.resize(mask.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST).astype(bool)
        if int(mask_d.sum()) < args.min_mask_pixels:
            continue

        # Scale K from full RGB resolution to depth resolution.
        sx = W_d / max(cam["W"], 1)
        sy = H_d / max(cam["H"], 1)
        K_d = np.array([
            [cam["fx"] * sx, 0.0,            cam["cx"] * sx],
            [0.0,            cam["fy"] * sy, cam["cy"] * sy],
            [0.0,            0.0,            1.0],
        ], dtype=np.float64)

        bundle.append({
            "cam_name": cam_name,
            "rgb": rgb_d,
            "rgb_full": rgb,
            "mask": mask_d,
            "depth_colmap": depth.astype(np.float32),
            "K": K_d,
            "fx": float(K_d[0, 0]),
            "fy": float(K_d[1, 1]),
            "cx": float(K_d[0, 2]),
            "cy": float(K_d[1, 2]),
            "H": H_d, "W": W_d,
            "R_w2c": cam["R_w2c"],
            "t_w2c": cam["t_w2c"],
            "mask_pixels": int(mask_d.sum()),
        })

    bundle.sort(key=lambda b: b["mask_pixels"], reverse=True)
    if args.max_views > 0:
        bundle = bundle[: args.max_views]
    return bundle


# ── FoundationPose driver ──────────────────────────────────────────────────────
class FPDriver:
    """Single FoundationPose estimator shared across views, with per-view pose_last."""

    def __init__(self, mesh: trimesh.Trimesh, args: argparse.Namespace, debug_dir: Path):
        scorer_dir = FP_WEIGHTS_DIR / "2023-10-28-18-33-37"
        refiner_dir = FP_WEIGHTS_DIR / "2024-01-11-20-02-45"
        if not scorer_dir.exists() or not refiner_dir.exists():
            raise FileNotFoundError(
                f"FoundationPose weights missing under {FP_WEIGHTS_DIR}. "
                f"Expected {scorer_dir} and {refiner_dir}."
            )
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        # FP needs vertex normals
        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None:
            mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        self.estimator = FoundationPose(
            model_pts=verts,
            model_normals=normals,
            mesh=mesh,
            scorer=self.scorer, refiner=self.refiner,
            glctx=self.glctx,
            debug=args.fp_debug,
            debug_dir=str(debug_dir),
        )
        self.args = args
        self.pose_last_per_view: Dict[str, torch.Tensor] = {}

    def register(self, view: Dict, depth_metric: np.ndarray) -> Optional[np.ndarray]:
        cam_name = view["cam_name"]
        K = np.asarray(view["K"], dtype=np.float64)
        rgb = view["rgb"].astype(np.uint8)
        mask = view["mask"].astype(np.uint8)
        depth = depth_metric.astype(np.float32)
        try:
            pose = self.estimator.register(
                K=K, rgb=rgb, depth=depth, ob_mask=mask,
                iteration=self.args.est_refine_iter,
            )
        except Exception as exc:
            print(f"[fp.register] {cam_name} failed: {exc}")
            return None
        # estimator.pose_last is set inside register — save per-view.
        self.pose_last_per_view[cam_name] = self.estimator.pose_last.detach().clone()
        return pose

    def track_one(self, view: Dict, depth_metric: np.ndarray) -> Optional[np.ndarray]:
        cam_name = view["cam_name"]
        if cam_name not in self.pose_last_per_view:
            # Not yet registered for this view — skip.
            return None
        # Restore per-view pose_last into the shared estimator.
        self.estimator.pose_last = self.pose_last_per_view[cam_name].clone()
        K = np.asarray(view["K"], dtype=np.float64)
        rgb = view["rgb"].astype(np.uint8)
        depth = depth_metric.astype(np.float32)
        try:
            pose = self.estimator.track_one(
                rgb=rgb, depth=depth, K=K,
                iteration=self.args.track_refine_iter,
            )
        except Exception as exc:
            print(f"[fp.track_one] {cam_name} failed: {exc}")
            return None
        self.pose_last_per_view[cam_name] = self.estimator.pose_last.detach().clone()
        return pose


# ── Visualization ──────────────────────────────────────────────────────────────
def project_world_to_cam(pt_world: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray,
                          K: np.ndarray, dist: np.ndarray) -> Optional[Tuple[float, float]]:
    p_cam = R_w2c @ pt_world + t_w2c
    if p_cam[2] <= 0:
        return None
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.astype(np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(pt_world.reshape(1, 3), rvec, tvec, K, dist)
    u, v = proj[0, 0]
    return float(u), float(v)


BBOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def mesh_aabb_corners(mesh: trimesh.Trimesh) -> np.ndarray:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    return np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ], dtype=np.float64)


def draw_pose_overlay(
    bgr: np.ndarray,
    cam: Dict,
    R_wo_metric: np.ndarray, t_wo_metric: np.ndarray, alpha: float,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    img = bgr.copy()
    bbox_local = mesh_aabb_corners(mesh)  # in metric mesh frame
    bbox_world_metric = (R_wo_metric @ bbox_local.T).T + t_wo_metric  # metric
    # COLMAP world: divide by alpha so it lines up with COLMAP extrinsics
    bbox_world_colmap = bbox_world_metric / alpha
    K = np.asarray(cam["K"], dtype=np.float64)
    R_w2c = cam["R_w2c"]
    t_w2c = cam["t_w2c"]
    dist = np.zeros(5, dtype=np.float64)
    pts_2d: List[Optional[Tuple[int, int]]] = []
    for X in bbox_world_colmap:
        proj = project_world_to_cam(X, R_w2c, t_w2c, K, dist)
        if proj is None:
            pts_2d.append(None)
        else:
            pts_2d.append((int(round(proj[0])), int(round(proj[1]))))
    for a, b in BBOX_EDGES:
        if pts_2d[a] is None or pts_2d[b] is None:
            continue
        cv2.line(img, pts_2d[a], pts_2d[b], (0, 0, 255), 2, cv2.LINE_AA)
    for p in pts_2d:
        if p is not None:
            cv2.circle(img, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
    return img


def annotate_tile(img: np.ndarray, cam_name: str, mask_pixels: int, down: float) -> np.ndarray:
    h, w = img.shape[:2]
    tile = cv2.resize(img, (max(1, int(w / down)), max(1, int(h / down))), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, cam_name, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 255, 30), 2, cv2.LINE_AA)
    cv2.putText(tile, f"mask={mask_pixels}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 30), 2, cv2.LINE_AA)
    return tile


def make_collage(images_bgr: Sequence[np.ndarray]) -> np.ndarray:
    if not images_bgr:
        raise ValueError("Cannot make collage without images.")
    num = len(images_bgr)
    cols = math.ceil(math.sqrt(num))
    rows = math.ceil(num / cols)
    ref_h, ref_w = images_bgr[0].shape[:2]
    normalized = [
        cv2.resize(img, (ref_w, ref_h)) if img.shape[:2] != (ref_h, ref_w) else img
        for img in images_bgr
    ]
    blank = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    padded = normalized + [blank] * (rows * cols - num)
    out_rows = []
    for r in range(rows):
        out_rows.append(np.concatenate(padded[r * cols:(r + 1) * cols], axis=1))
    return np.concatenate(out_rows, axis=0)


def _ensure_even_hw(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[: h - (h % 2), : w - (w % 2)]


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_ensure_even_hw(f) for f in frames]
    prepared = [np.ascontiguousarray(f[:, :, ::-1].astype(np.uint8)) for f in frames]
    try:
        with imageio.get_writer(str(path), fps=fps, codec="libx264",
                                pixelformat="yuv420p", quality=8) as writer:
            for frame in prepared:
                writer.append_data(frame)
    except Exception as exc:
        print(f"Video write failed: {exc}")


# ── Output JSON ────────────────────────────────────────────────────────────────
def save_pose_json(path: Path, names: Sequence[str], poses: Sequence[np.ndarray],
                   losses: Sequence[float], scales: Sequence[float]) -> None:
    payload: Dict[str, Dict] = {}
    for n, p, l, s in zip(names, poses, losses, scales):
        payload[n] = {
            "pose_matrix": p.tolist(),
            "rotation_matrix": p[:3, :3].tolist(),
            "translation": p[:3, 3].tolist(),
            "loss": float(l),
            "scale": float(s),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = build_argparser().parse_args()
    seq_root: Path = args.seq_root
    parsed_root = seq_root / SEQ_PARSED_SUBPATH
    calib_root = seq_root / SEQ_CALIB_SUBPATH
    mask_root = seq_root / SEQ_MASK_SUBPATH
    depth_root = seq_root / SEQ_DEPTH_SUBPATH
    output_root = args.output_root or (seq_root / SEQ_OUTPUT_SUBPATH)
    (output_root / "poses").mkdir(parents=True, exist_ok=True)
    (output_root / "collages").mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    timestamp_names = sorted_timestamps(parsed_root)
    if args.timestamp_end >= 0:
        timestamp_names = timestamp_names[args.timestamp_start:args.timestamp_end:args.timestamp_step]
    else:
        timestamp_names = timestamp_names[args.timestamp_start::args.timestamp_step]
    if args.max_timestamps > 0:
        timestamp_names = timestamp_names[: args.max_timestamps]
    if not timestamp_names:
        raise RuntimeError("No timestamps selected.")

    camera_infos = load_cameras_rdf(calib_root)
    if not camera_infos:
        raise RuntimeError(f"No cameras loaded from {calib_root}.")
    camera_infos = filter_bad_cameras(camera_infos, calib_root, args.kp3d_reproj_thresh)

    print(f"Loading mesh from {args.mesh_path}")
    mesh = trimesh.load(str(args.mesh_path), force="mesh", process=False)
    if mesh.vertices.shape[0] == 0:
        raise RuntimeError(f"Mesh has no vertices: {args.mesh_path}")
    # FoundationPose's make_mesh_tensors expects mesh.visual.material.image
    # (SimpleMaterial). .glb files load with a PBRMaterial whose texture lives
    # in baseColorTexture, so convert it to SimpleMaterial before handing off.
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and \
            isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
        pbr = mesh.visual.material
        tex_image = pbr.baseColorTexture
        if tex_image is None and pbr.baseColorFactor is not None:
            rgb = np.array(pbr.baseColorFactor[:3], dtype=np.uint8).reshape(1, 1, 3)
            tex_image = Image.fromarray(np.tile(rgb, (8, 8, 1)))
        if tex_image is None:
            tex_image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))
        mesh.visual.material = trimesh.visual.material.SimpleMaterial(image=tex_image)
    extent_raw = normalize_mesh_in_place(mesh)
    print(f"  raw mesh longest extent: {extent_raw:.4f} (mesh units); "
          f"normalized to {NORMALIZED_MESH_EXTENT} m (fake-metric).")

    fp_debug_dir = output_root / "fp_debug"
    print("Initializing FoundationPose (loading weights, building rotation grid)...")
    fp = FPDriver(mesh, args, fp_debug_dir)
    print("FoundationPose ready.")

    solved_names: List[str] = []
    solved_poses: List[np.ndarray] = []
    solved_losses: List[float] = []
    solved_scales: List[float] = []
    all_collages: List[np.ndarray] = []
    alpha: Optional[float] = args.alpha_fix
    is_first_frame = True

    ts_pbar = tqdm(timestamp_names, desc="timestamps", dynamic_ncols=True)
    for timestamp_name in ts_pbar:
        bundle = collect_view_bundle(timestamp_name, parsed_root, mask_root, depth_root,
                                     camera_infos, args)
        if not bundle:
            tqdm.write(f"[{timestamp_name}] Skipped — no valid views.")
            continue

        # ── α estimation at first frame ──
        if alpha is None:
            obs_for_alpha = [
                {"fx": v["fx"], "fy": v["fy"], "mask": v["mask"], "depth": v["depth_colmap"]}
                for v in bundle
            ]
            alpha, est_list = estimate_alpha_heuristic(obs_for_alpha)
            tqdm.write(
                f"[{timestamp_name}] α (heuristic) = {alpha:.4f} m/colmap-unit "
                f"(from {len(est_list)} views; range "
                f"[{min(est_list) if est_list else float('nan'):.4f}, "
                f"{max(est_list) if est_list else float('nan'):.4f}])"
            )

        # ── per-view FP ──
        per_view_results: List[Dict] = []
        for view in bundle:
            depth_metric = (view["depth_colmap"] * alpha).astype(np.float32)
            depth_metric[depth_metric > args.depth_clip_max] = 0.0
            depth_metric[depth_metric < 0.0] = 0.0
            if is_first_frame or view["cam_name"] not in fp.pose_last_per_view:
                pose_co = fp.register(view, depth_metric)
            else:
                pose_co = fp.track_one(view, depth_metric)
            if pose_co is None:
                continue
            R_co = np.asarray(pose_co[:3, :3], dtype=np.float64)
            t_co = np.asarray(pose_co[:3, 3], dtype=np.float64)
            R_wo, t_wo_metric = world_pose_from_view(R_co, t_co, view["R_w2c"], view["t_w2c"], alpha)
            per_view_results.append({
                "cam_name": view["cam_name"],
                "R_co": R_co, "t_co": t_co,
                "R_wo": R_wo, "t_wo_metric": t_wo_metric,
                "view": view,
            })

        if not per_view_results:
            tqdm.write(f"[{timestamp_name}] FP failed in all views; skipping.")
            continue

        weights = np.array([v["view"]["mask_pixels"] for v in per_view_results], dtype=np.float64)
        R_wo_fused, t_wo_fused, spread_metric = fuse_world_poses(per_view_results, weights)

        # Save
        pose_save, scale_save = to_consumer_schema(R_wo_fused, t_wo_fused, alpha)
        solved_names.append(timestamp_name)
        solved_poses.append(pose_save.astype(np.float32))
        solved_losses.append(spread_metric)  # multi-view translation spread (metric, in meters)
        solved_scales.append(scale_save)

        tqdm.write(
            f"[{timestamp_name}] views={len(per_view_results)}/{len(bundle)}  "
            f"trans_spread={spread_metric:.4f}m  scale={scale_save:.4f}  α={alpha:.4f}"
        )

        # collage
        tiles = []
        for v in per_view_results[: args.max_collage_tiles]:
            view = v["view"]
            bgr = cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR)
            drawn = draw_pose_overlay(bgr, view, R_wo_fused, t_wo_fused, alpha, mesh)
            tiles.append(annotate_tile(drawn, view["cam_name"], view["mask_pixels"], args.down))
        if tiles:
            collage = make_collage(tiles)
            cv2.putText(
                collage,
                f"{timestamp_name} spread={spread_metric:.4f}m views={len(per_view_results)} α={alpha:.4f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
            )
            all_collages.append(collage)
            cv2.imwrite(str(output_root / "collages" / f"{timestamp_name}.jpg"), collage)

        save_pose_json(
            output_root / "poses" / "optimized_poses.json",
            solved_names, solved_poses, solved_losses, solved_scales,
        )
        is_first_frame = False

    if not solved_poses:
        raise RuntimeError("No timestamps were successfully fit.")

    save_pose_json(
        output_root / "poses" / "optimized_poses.json",
        solved_names, solved_poses, solved_losses, solved_scales,
    )
    np.save(output_root / "poses" / "optimized_poses.npy", np.stack(solved_poses))
    write_video(output_root / "videos" / "multiview_pose_collage.mp4", all_collages, fps=30)

    print(f"Saved {len(solved_poses)} FP poses to "
          f"{output_root / 'poses' / 'optimized_poses.json'}  (α frozen at {alpha:.4f})")


if __name__ == "__main__":
    main()
