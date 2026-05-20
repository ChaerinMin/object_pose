"""ArUco marker-based object 6DoF pose fitting.

Alternative to scripts/1_object_tracking_local.py (which uses differentiable
rendering of object silhouettes). For each timestamp:
  1. Run ArUco detection on every camera's RGB image (full resolution).
  2. For every (marker_id, corner_idx) seen in >= --min-views-per-corner views,
     triangulate the 2D detections into a world-space 3D point.
  3. Run Umeyama (scaled Kabsch) between mesh-local marker corners (after the
     same normalize transform that scripts/1_object_tracking_local.py applies
     to the mesh) and the triangulated world points to recover R, t, scale.

Output schema matches scripts/1_object_tracking_local.py exactly so the
downstream axis_align step is unchanged.

Marker layout file
------------------
Read from <mesh-path>.with_suffix('.json') by default, e.g.
  object_meshes/cylindar.obj  ->  object_meshes/cylindar.json

JSON format:
{
  "_comment": "Coords are in MESH LOCAL frame (same units as the .obj). Corner order
               = OpenCV ArUco order (top-left, top-right, bottom-right, bottom-left)
               when looking AT the printed marker face.",
  "dictionary":     "DICT_4X4_50",
  "marker_length":  0.04,
  "markers": [
    {
      "id": 0,
      "corners": [
        [ x_tl, y_tl, z_tl ],
        [ x_tr, y_tr, z_tr ],
        [ x_br, y_br, z_br ],
        [ x_bl, y_bl, z_bl ]
      ]
    },
    ...
  ]
}
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import trimesh
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

import src.utils.colmap_utils as colmap_utils  # noqa: E402


# Match scripts/1_object_tracking_local.py so output poses live in the same frame.
NORMALIZED_MESH_EXTENT = 0.15
UNIT_SCALE_THRESHOLD = 0.8

SEQ_PARSED_SUBPATH = "parsed"
SEQ_CALIB_SUBPATH = "calib"
SEQ_OUTPUT_SUBPATH = "outputs/object_pose/marker"


def get_aruco_dict_id(name: str) -> int:
    name = name.upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(cv2.aruco, name):
        raise ValueError(
            f"Unknown ArUco dictionary '{name}'. "
            f"Examples: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250."
        )
    return getattr(cv2.aruco, name)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marker-based object 6DoF pose fitting.")
    parser.add_argument("--seq-root", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument(
        "--marker-info", type=Path, default=None,
        help="Marker layout JSON path. Default: <mesh-path>.with_suffix('.json').",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--timestamp-start", type=int, default=0)
    parser.add_argument("--timestamp-end", type=int, default=-1)
    parser.add_argument("--timestamp-step", type=int, default=1)
    parser.add_argument("--max-timestamps", type=int, default=-1)
    parser.add_argument(
        "--min-views-per-corner", type=int, default=2,
        help="Minimum views in which a corner must be detected for triangulation.",
    )
    parser.add_argument(
        "--min-correspondences", type=int, default=4,
        help="Minimum (marker_id, corner) world points required to fit a pose.",
    )
    parser.add_argument(
        "--collage-down", type=float, default=2.0,
        help="Downsample factor for collage tiles.",
    )
    parser.add_argument(
        "--kp3d-reproj-thresh", type=float, default=50.0,
        help="Exclude cameras whose kp3d_reproj_error in image_confidence.json exceeds this.",
    )
    parser.add_argument(
        "--fix-scale-after-first", action="store_true",
        help="Freeze scale to the first solved frame's value for all subsequent frames.",
    )
    return parser


# ── Marker layout ──────────────────────────────────────────────────────────────
def find_marker_info_path(mesh_path: Path, override: Optional[Path]) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--marker-info path does not exist: {override}")
        return override
    candidate = mesh_path.with_suffix(".json")
    if not candidate.exists():
        raise FileNotFoundError(f"Marker layout JSON not found at {candidate}")
    return candidate


def load_marker_info(path: Path) -> Tuple[int, float, Dict[int, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dict_id = get_aruco_dict_id(data.get("dictionary", "DICT_4X4_50"))
    marker_length = float(data.get("marker_length", 0.0))
    layout: Dict[int, np.ndarray] = {}
    for entry in data["markers"]:
        mid = int(entry["id"])
        corners = np.asarray(entry["corners"], dtype=np.float64)
        if corners.shape != (4, 3):
            raise ValueError(
                f"Marker {mid} 'corners' must be 4x3 (tl, tr, br, bl). Got shape {corners.shape}."
            )
        if mid in layout:
            raise ValueError(f"Duplicate marker id {mid} in layout file.")
        layout[mid] = corners
    if not layout:
        raise ValueError(f"No markers found in {path}.")
    return dict_id, marker_length, layout


# ── Mesh-frame transform (must mirror scripts/1_object_tracking_local.py) ──────
def compute_mesh_normalize_factor(mesh_path: Path) -> float:
    suffix = mesh_path.suffix.lower()
    if suffix not in {".obj", ".glb", ".gltf"}:
        raise ValueError(f"Unsupported mesh format '{suffix}'. Use .obj, .glb, or .gltf.")
    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(tm.vertices, dtype=np.float64)
    if verts.size == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    extent = float((verts.max(axis=0) - verts.min(axis=0)).max())
    if extent < UNIT_SCALE_THRESHOLD:
        return 1.0
    return NORMALIZED_MESH_EXTENT / extent


def transform_marker_corners(
    layout: Dict[int, np.ndarray], normalize_factor: float
) -> Dict[int, np.ndarray]:
    """Apply normalize so corners live in the same frame as the post-load mesh."""
    return {mid: corners * normalize_factor for mid, corners in layout.items()}


def compute_mesh_aabb_meshframe(mesh_path: Path, normalize_factor: float) -> np.ndarray:
    """Return (8, 3) cuboid corners of the mesh AABB in the same mesh-local frame
    used by the fitted pose."""
    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(tm.vertices, dtype=np.float64) * normalize_factor
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
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


BBOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
)


# ── Camera loading (COLMAP RDF, world-from-camera) ─────────────────────────────
def load_cameras_rdf(calib_root: Path) -> List[Dict]:
    cameras, images, _ = colmap_utils.read_model(str(calib_root), ext=".bin")
    out = []
    for image in images.values():
        camera = cameras[image.camera_id]
        params = camera.params
        fx, fy, cx, cy = map(float, params[:4])
        dist = np.zeros(5, dtype=np.float64)
        model = camera.model
        if model == "OPENCV" and len(params) >= 8:
            dist[:4] = params[4:8]
        elif model == "FULL_OPENCV" and len(params) >= 12:
            # k1, k2, p1, p2, k3, k4, k5, k6
            dist = np.zeros(8, dtype=np.float64)
            dist[:8] = params[4:12]
        elif model == "RADIAL" and len(params) >= 5:
            dist[0] = params[3]
            dist[1] = params[4]
        elif model == "SIMPLE_RADIAL" and len(params) >= 4:
            dist[0] = params[3]
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
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
            "image_name": image.name,
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


# ── ArUco detection ────────────────────────────────────────────────────────────
def make_detector(dict_id: int):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(aruco_dict, params), None
    # Older API: returns dict + params, caller uses cv2.aruco.detectMarkers
    params = cv2.aruco.DetectorParameters_create()
    return aruco_dict, params


def detect_markers(image_bgr: np.ndarray, detector_or_dict, params_or_none) -> Dict[int, np.ndarray]:
    if hasattr(detector_or_dict, "detectMarkers"):
        corners, ids, _ = detector_or_dict.detectMarkers(image_bgr)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, detector_or_dict, parameters=params_or_none)
    if ids is None:
        return {}
    out: Dict[int, np.ndarray] = {}
    for cid, mid in zip(corners, ids.flatten()):
        out[int(mid)] = cid.reshape(4, 2).astype(np.float64)
    return out


# ── Multi-view triangulation ───────────────────────────────────────────────────
def make_projection_matrix(camera_info: Dict) -> np.ndarray:
    """K [R|t] where (R, t) is world-to-camera in COLMAP RDF."""
    return camera_info["K"] @ np.hstack([camera_info["R_w2c"], camera_info["t_w2c"].reshape(3, 1)])


def undistort_points(uv: np.ndarray, camera_info: Dict) -> np.ndarray:
    """Pixel coords -> ideal-pinhole pixel coords. No-op if no distortion."""
    if not np.any(camera_info["dist"]):
        return uv
    pts = uv.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, camera_info["K"], camera_info["dist"], P=camera_info["K"])
    return und.reshape(-1, 2)


def triangulate_dlt(observations: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """observations: list of (P_3x4, uv_2)."""
    if len(observations) < 2:
        return None
    rows = []
    for P, uv in observations:
        u, v = float(uv[0]), float(uv[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return (X[:3] / X[3]).astype(np.float64)


# ── Umeyama (scaled Kabsch) ────────────────────────────────────────────────────
def umeyama(src: np.ndarray, dst: np.ndarray, fix_scale: Optional[float] = None
           ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Find (R, t, s) minimizing sum ||dst_i - (s R src_i + t)||^2.

    Returns (R, t, s, rmse).
    """
    n = len(src)
    mu_x = src.mean(axis=0)
    mu_y = dst.mean(axis=0)
    cx = src - mu_x
    cy = dst - mu_y
    sigma2_x = (cx ** 2).sum() / n
    Sigma_xy = (cx.T @ cy) / n
    U, D, Vt = np.linalg.svd(Sigma_xy)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1
    R = V @ S @ U.T
    if fix_scale is not None:
        s = float(fix_scale)
    else:
        if sigma2_x < 1e-18:
            s = 1.0
        else:
            s = float(np.trace(np.diag(D) @ S) / sigma2_x)
    t = mu_y - s * (R @ mu_x)
    err = dst - (s * (R @ src.T).T + t)
    rmse = float(np.sqrt((err ** 2).sum(axis=1).mean()))
    return R, t, s, rmse


# ── Per-timestamp pipeline ─────────────────────────────────────────────────────
def _read_and_detect_one(
    image_dir: Path, cam: Dict,
    detector_or_dict, detector_params,
    layout_meshframe: Dict[int, np.ndarray],
) -> Tuple[str, Optional[Dict[int, np.ndarray]]]:
    cam_name = cam["cam_name"]
    img_path = image_dir / f"{cam_name}.jpg"
    if not img_path.exists():
        return cam_name, None
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return cam_name, None
    det = detect_markers(bgr, detector_or_dict, detector_params)
    return cam_name, {mid: c for mid, c in det.items() if mid in layout_meshframe}


def fit_pose_for_timestamp(
    timestamp_name: str,
    parsed_root: Path,
    camera_infos: Sequence[Dict],
    layout_meshframe: Dict[int, np.ndarray],
    detector_or_dict,
    detector_params,
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    fix_scale: Optional[float] = None,
) -> Optional[Dict]:
    image_dir = parsed_root / timestamp_name / "images"
    if not image_dir.exists():
        return None

    detections: Dict[str, Dict[int, np.ndarray]] = {}
    futures = [
        executor.submit(
            _read_and_detect_one, image_dir, cam,
            detector_or_dict, detector_params, layout_meshframe,
        )
        for cam in camera_infos
    ]
    for fut in futures:
        cam_name, det = fut.result()
        if det is None:
            continue
        detections[cam_name] = det

    cam_by_name = {c["cam_name"]: c for c in camera_infos}
    proj_cache = {name: make_projection_matrix(c) for name, c in cam_by_name.items()}

    # (marker_id, corner_idx) -> [(P, uv, cam_name), ...]
    obs_per_corner: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray, str]]] = {}
    for cam_name, det in detections.items():
        cam = cam_by_name[cam_name]
        for mid, corners_uv in det.items():
            corners_und = undistort_points(corners_uv, cam)
            for cidx in range(4):
                obs_per_corner.setdefault((mid, cidx), []).append(
                    (proj_cache[cam_name], corners_und[cidx], cam_name)
                )

    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    used_keys: List[Tuple[int, int]] = []
    for (mid, cidx), obs in obs_per_corner.items():
        if len(obs) < args.min_views_per_corner:
            continue
        X = triangulate_dlt([(P, uv) for (P, uv, _) in obs])
        if X is None:
            continue
        # cheirality check across views
        pos_z = 0
        for _, _, name in obs:
            cam = cam_by_name[name]
            p_cam = cam["R_w2c"] @ X + cam["t_w2c"]
            if p_cam[2] > 0:
                pos_z += 1
        if pos_z < args.min_views_per_corner:
            continue
        src_pts.append(layout_meshframe[mid][cidx])
        dst_pts.append(X)
        used_keys.append((mid, cidx))

    if len(src_pts) < args.min_correspondences:
        return None

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    R, t, s, rmse_world = umeyama(src, dst, fix_scale=fix_scale)
    pose = np.eye(4, dtype=np.float64)
    # Save in PyTorch3D row-vector convention (consumer applies pose[:3,:3].T).
    pose[:3, :3] = R.T
    pose[:3, 3] = t

    reproj_err = compute_reprojection_error(
        layout_meshframe, R, t, s, detections, cam_by_name, used_keys
    )

    return {
        "pose": pose,
        "scale": float(s),
        "rmse_world": float(rmse_world),
        "reproj_loss": float(reproj_err),
        "num_correspondences": int(src.shape[0]),
        "num_views_total": len(detections),
        "num_views_with_marker": sum(1 for d in detections.values() if d),
        "detections": detections,
        "used_keys": used_keys,
    }


def compute_reprojection_error(
    layout_meshframe: Dict[int, np.ndarray],
    R: np.ndarray, t: np.ndarray, s: float,
    detections: Dict[str, Dict[int, np.ndarray]],
    cam_by_name: Dict[str, Dict],
    used_keys: List[Tuple[int, int]],
) -> float:
    used_set = set(used_keys)
    sq_errs = []
    for cam_name, det in detections.items():
        cam = cam_by_name[cam_name]
        rvec, _ = cv2.Rodrigues(cam["R_w2c"])
        tvec = cam["t_w2c"].astype(np.float64).reshape(3, 1)
        for mid, corners_uv in det.items():
            for cidx in range(4):
                if (mid, cidx) not in used_set:
                    continue
                X_local = layout_meshframe[mid][cidx]
                X_world = s * (R @ X_local) + t
                proj, _ = cv2.projectPoints(
                    X_world.reshape(1, 3), rvec, tvec, cam["K"], cam["dist"]
                )
                u, v = proj[0, 0]
                du = u - corners_uv[cidx, 0]
                dv = v - corners_uv[cidx, 1]
                sq_errs.append(du * du + dv * dv)
    if not sq_errs:
        return float("nan")
    return float(np.sqrt(np.mean(sq_errs)))


# ── Visualization ──────────────────────────────────────────────────────────────
def draw_overlay(
    bgr: np.ndarray,
    det: Dict[int, np.ndarray],
    cam: Dict,
    pose: Optional[np.ndarray],
    scale: Optional[float],
    layout_meshframe: Dict[int, np.ndarray],
    bbox_meshframe: Optional[np.ndarray] = None,
) -> np.ndarray:
    img = bgr.copy()
    for mid, c in det.items():
        c_int = c.astype(np.int32)
        cv2.polylines(img, [c_int], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, str(mid), tuple(c_int[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if pose is None or scale is None:
        return img

    R_world_from_obj = pose[:3, :3].T  # consumer convention
    t_world = pose[:3, 3]
    rvec, _ = cv2.Rodrigues(cam["R_w2c"])
    tvec = cam["t_w2c"].astype(np.float64).reshape(3, 1)

    def project_meshframe(points_local: np.ndarray) -> Optional[np.ndarray]:
        """Transform mesh-local points by fitted pose+scale and project to image.
        Returns None if any point is behind the camera."""
        pts_world = (scale * (R_world_from_obj @ points_local.T)).T + t_world
        pts_cam = pts_world @ cam["R_w2c"].T + cam["t_w2c"]
        if np.any(pts_cam[:, 2] <= 0):
            return None
        proj, _ = cv2.projectPoints(
            pts_world.reshape(-1, 1, 3).astype(np.float64),
            rvec, tvec, cam["K"], cam["dist"],
        )
        return proj.reshape(-1, 2)

    for mid, corners_local in layout_meshframe.items():
        proj = project_meshframe(corners_local)
        if proj is None:
            continue
        pts = proj.astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, (0, 0, 255), -1, cv2.LINE_AA)

    if bbox_meshframe is not None:
        proj_bbox = project_meshframe(bbox_meshframe)
        if proj_bbox is not None:
            bbox_pts = proj_bbox.astype(np.int32)
            for i, j in BBOX_EDGES:
                cv2.line(img, tuple(bbox_pts[i]), tuple(bbox_pts[j]),
                         (0, 200, 255), 2, cv2.LINE_AA)
    return img


def annotate_tile(image_bgr: np.ndarray, cam_name: str, n_markers: int, down: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out_w = max(1, int(w / down))
    out_h = max(1, int(h / down))
    tile = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, cam_name, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 255, 30), 2, cv2.LINE_AA)
    cv2.putText(tile, f"markers={n_markers}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 220, 30), 2, cv2.LINE_AA)
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
    rows_out = []
    for r in range(rows):
        rows_out.append(np.concatenate(padded[r * cols:(r + 1) * cols], axis=1))
    return np.concatenate(rows_out, axis=0)


def _ensure_even_hw(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    if new_h == h and new_w == w:
        return frame
    return frame[:new_h, :new_w]


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_ensure_even_hw(f) for f in frames]
    prepared = [np.ascontiguousarray(f[:, :, ::-1].astype(np.uint8)) for f in frames]
    try:
        with imageio.get_writer(
            str(path), fps=fps, codec="libx264", pixelformat="yuv420p", quality=8
        ) as writer:
            for frame in prepared:
                writer.append_data(frame)
    except Exception as exc:
        print(f"Video write failed via imageio: {exc}")
        frame_dir = path.parent / (path.stem + "_frames")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            cv2.imwrite(str(frame_dir / f"{idx:06d}.png"), frame)
        print(f"Wrote frames to {frame_dir} instead of video.")


def sorted_timestamps(root: Path) -> List[str]:
    return sorted(
        [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("timestamp_")],
        key=lambda name: int(name.split("_")[-1]),
    )


def save_pose_json(
    path: Path, names: Sequence[str], poses: Sequence[np.ndarray],
    losses: Sequence[float], scales: Sequence[float],
) -> None:
    payload: Dict[str, Dict] = {}
    for name, pose, loss, scale in zip(names, poses, losses, scales):
        payload[name] = {
            "pose_matrix": pose.tolist(),
            "rotation_matrix": pose[:3, :3].tolist(),
            "translation": pose[:3, 3].tolist(),
            "loss": float(loss),
            "scale": float(scale),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = build_argparser().parse_args()
    seq_root: Path = args.seq_root
    parsed_root = seq_root / SEQ_PARSED_SUBPATH
    calib_root = seq_root / SEQ_CALIB_SUBPATH
    output_root = args.output_root or (seq_root / SEQ_OUTPUT_SUBPATH)
    (output_root / "poses").mkdir(parents=True, exist_ok=True)
    (output_root / "collages").mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)

    marker_path = find_marker_info_path(args.mesh_path, args.marker_info)
    print(f"Loading marker layout from {marker_path}")
    dict_id, marker_length, layout = load_marker_info(marker_path)
    print(f"  dictionary={dict_id}  marker_length={marker_length}  num_markers={len(layout)}")

    normalize_factor = compute_mesh_normalize_factor(args.mesh_path)
    print(f"Mesh normalize factor: {normalize_factor:.6f}")
    layout_meshframe = transform_marker_corners(layout, normalize_factor)
    bbox_meshframe = compute_mesh_aabb_meshframe(args.mesh_path, normalize_factor)

    if marker_length > 0:
        for mid, corners_orig in layout.items():
            edge = float(np.linalg.norm(corners_orig[1] - corners_orig[0]))
            if abs(edge - marker_length) > 0.05 * marker_length:
                print(
                    f"  [warn] Marker {mid} top-edge length {edge:.4f} differs from "
                    f"marker_length {marker_length:.4f} by >5%."
                )

    timestamp_names = sorted_timestamps(parsed_root)
    if args.timestamp_end >= 0:
        timestamp_names = timestamp_names[args.timestamp_start:args.timestamp_end:args.timestamp_step]
    else:
        timestamp_names = timestamp_names[args.timestamp_start::args.timestamp_step]
    if args.max_timestamps > 0:
        timestamp_names = timestamp_names[:args.max_timestamps]
    if not timestamp_names:
        raise RuntimeError("No timestamps selected.")

    camera_infos = load_cameras_rdf(calib_root)
    if not camera_infos:
        raise RuntimeError(f"No cameras loaded from {calib_root}.")
    camera_infos = filter_bad_cameras(camera_infos, calib_root, args.kp3d_reproj_thresh)

    detector_or_dict, detector_params = make_detector(dict_id)

    # ── Phase 1: pure pose tracking. No per-frame visualization or JSON.
    fits: Dict[str, Dict] = {}
    fixed_scale: Optional[float] = None
    n_workers = max(1, min(len(camera_infos), 16))

    track_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        ts_pbar = tqdm(timestamp_names, desc="tracking", dynamic_ncols=True)
        for timestamp_name in ts_pbar:
            result = fit_pose_for_timestamp(
                timestamp_name, parsed_root, camera_infos, layout_meshframe,
                detector_or_dict, detector_params, args,
                executor=executor, fix_scale=fixed_scale,
            )
            if result is None:
                tqdm.write(f"[{timestamp_name}] Skipped (insufficient marker observations).")
                continue
            scale = result["scale"]
            loss = result["reproj_loss"]
            if args.fix_scale_after_first and fixed_scale is None:
                fixed_scale = scale
                tqdm.write(f"[{timestamp_name}] Scale fixed at {fixed_scale:.4f} for subsequent frames.")
            tqdm.write(
                f"[{timestamp_name}] corresps={result['num_correspondences']}  "
                f"reproj={loss:.2f}px  rmse_world={result['rmse_world']:.4f}  "
                f"scale={scale:.4f}  views_with_marker={result['num_views_with_marker']}"
            )
            fits[timestamp_name] = result
    pure_track_seconds = time.perf_counter() - track_t0

    if not fits:
        raise RuntimeError("No timestamps were successfully fit.")

    solved_names = list(fits.keys())
    solved_poses = [fits[n]["pose"].astype(np.float32) for n in solved_names]
    solved_losses = [fits[n]["reproj_loss"] for n in solved_names]
    solved_scales = [fits[n]["scale"] for n in solved_names]

    # ── Phase 2: persist poses + render visualization (re-reads images).
    save_pose_json(
        output_root / "poses" / "optimized_poses.json",
        solved_names, solved_poses, solved_losses, solved_scales,
    )
    np.save(output_root / "poses" / "optimized_poses.npy", np.stack(solved_poses))

    all_collages: List[np.ndarray] = []
    for ts_name in tqdm(solved_names, desc="visualization", dynamic_ncols=True):
        result = fits[ts_name]
        pose = result["pose"].astype(np.float32)
        scale = result["scale"]
        loss = result["reproj_loss"]
        image_dir = parsed_root / ts_name / "images"
        tiles = []
        for cam in camera_infos:
            name = cam["cam_name"]
            img_path = image_dir / f"{name}.jpg"
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if img_path.exists() else None
            det = result["detections"].get(name, {})
            if bgr is None:
                drawn = np.zeros((cam["H"], cam["W"], 3), dtype=np.uint8)
            else:
                drawn = draw_overlay(
                    bgr, det, cam, pose, scale, layout_meshframe,
                    bbox_meshframe=bbox_meshframe,
                )
            tiles.append(annotate_tile(drawn, name, len(det), args.collage_down))
        collage = make_collage(tiles)
        cv2.putText(
            collage,
            f"{ts_name} reproj={loss:.2f}px corresps={result['num_correspondences']} scale={scale:.4f}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(output_root / "collages" / f"{ts_name}.jpg"), collage)
        all_collages.append(collage)

    write_video(output_root / "videos" / "multiview_pose_collage.mp4", all_collages, fps=30)

    n_solved = len(fits)
    fps = n_solved / pure_track_seconds if pure_track_seconds > 0 else float("nan")
    print(
        f"\nPure pose tracking time: {pure_track_seconds:.2f}s for {n_solved} frames "
        f"({fps:.2f} fps, {n_workers} detection workers, {len(camera_infos)} cameras)"
    )
    print(f"Saved {n_solved} marker-fit poses to "
          f"{output_root / 'poses' / 'optimized_poses.json'}")


if __name__ == "__main__":
    main()
