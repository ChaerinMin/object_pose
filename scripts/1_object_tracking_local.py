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
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

import src.utils.colmap_utils as colmap_utils
from src.utils.pytorch3d_utils import setup_renderer, DRModel, check_for_nan_params, visualize_image_list, alpha_blend

from pytorch3d.io import load_objs_as_meshes


DEFAULT_MESH_PATH = Path("/oscar/data/ssrinath/public/brics-mini-copy/obj_assets/3_30_2026.obj")
DEFAULT_MASK_ROOT = Path("/oscar/data/ssrinath/public/brics-mini/2026-04-06/multisequence000001_safe/outputs/sam3")
DEFAULT_PARSED_ROOT = Path("/oscar/data/ssrinath/public/brics-mini/2026-04-06/multisequence000001_safe/parsed")
DEFAULT_CALIB_ROOT = Path("/oscar/data/ssrinath/public/brics-mini/2026-04-06/multisequence000001_safe/calib")
DEFAULT_TRAJ_PATH = Path("/oscar/data/ssrinath/public/brics-mini-copy/obj_traj/object_poses_icp_trackscaled_0142.npz")
DEFAULT_DEPTH_ROOT = Path("/oscar/data/ssrinath/public/brics-mini/2026-04-06/multisequence000001_safe/outputs/da3")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "multiview_object_pose_nvdiffrast"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multiview object pose fitting with PyTorch3D.")
    parser.add_argument("--mesh-path", type=Path, default=DEFAULT_MESH_PATH)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    parser.add_argument("--calib-root", type=Path, default=DEFAULT_CALIB_ROOT)
    parser.add_argument("--traj-path", type=Path, default=DEFAULT_TRAJ_PATH)
    parser.add_argument("--use-traj-init", action="store_true", help="Use trajectory npz for initialization instead of depth.")
    parser.add_argument("--use-depth-init", action="store_true", default=True, help="Initialize translation from DA3 depth (default).")
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--depth-extr-inv", action="store_true", help="Invert DA3 extrinsics if they are cam_from_world.")
    parser.add_argument("--depth-max-points", type=int, default=20000, help="Max depth points to sample for init.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pose-key", type=str, default="global_poses")
    parser.add_argument("--timestamp-start", type=int, default=0)
    parser.add_argument("--timestamp-end", type=int, default=-1)
    parser.add_argument("--timestamp-step", type=int, default=1)
    parser.add_argument("--max-timestamps", type=int, default=-1)
    parser.add_argument("--down", type=float, default=2.0)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--lr-rot", type=float, default=5e-3)
    parser.add_argument("--lr-trans", type=float, default=2e-4)
    parser.add_argument("--min-mask-pixels", type=int, default=200)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--near", type=float, default=1e-3)
    parser.add_argument("--far", type=float, default=10.0)
    parser.add_argument("--shader", type=str, default="mask", choices=["phong", "mask"])
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the mesh (default 1.0 = unit scale).",
    )
    parser.add_argument("--save-per-timestamp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def sorted_timestamps(root: Path) -> List[str]:
    timestamps = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("timestamp_")]
    return sorted(timestamps, key=lambda name: int(name.split("_")[-1]))


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.exists() or mask_path.stat().st_size <= 22:
        return None
    with zipfile.ZipFile(mask_path, "r") as archive:
        keys = sorted(archive.namelist())
        if not keys:
            return None
        masks = [np.load(archive.open(key)).astype(bool) for key in keys]
    return np.any(np.stack(masks, axis=0), axis=0)


def _load_depth_bundle(depth_root: Path, timestamp_name: str):
    npz_path = depth_root / timestamp_name / "depths" / "exports" / "npz" / "results.npz"
    list_path = depth_root / timestamp_name / "depths" / "exports" / "npz" / "results.txt"
    if not npz_path.exists() or not list_path.exists():
        return None
    bundle = np.load(npz_path, allow_pickle=True)
    with list_path.open("r", encoding="utf-8") as f:
        names = [Path(line.strip()).stem for line in f if line.strip()]
    return bundle, names


def depth_init_pose(
    depth_root: Path,
    mask_root: Path,
    timestamp_name: str,
    depth_extr_inv: bool,
    max_points: int,
    camera_infos: Optional[Sequence[Dict]] = None,
) -> Optional[np.ndarray]:
    loaded = _load_depth_bundle(depth_root, timestamp_name)
    if loaded is None:
        return None
    bundle, names = loaded
    depth = bundle["depth"]
    intrs = bundle["intrinsics"]
    extrs = bundle["extrinsics"]

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    cam_info_map = {info["cam_name"]: info for info in (camera_infos or [])}
    mask_dir = mask_root / timestamp_name / "object_masks"
    points_world = []

    for cam_name, idx in name_to_idx.items():
        mask_path = mask_dir / f"{cam_name}.npz"
        mask = load_mask(mask_path)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        u = float(xs.mean())
        v = float(ys.mean())

        dmap = depth[idx]
        dh, dw = dmap.shape
        mh, mw = mask.shape
        sx = dw / max(mw, 1)
        sy = dh / max(mh, 1)
        u_d = int(np.clip(round(u * sx), 0, dw - 1))
        v_d = int(np.clip(round(v * sy), 0, dh - 1))
        z = float(dmap[v_d, u_d])
        if z <= 0:
            continue

        depth_h, depth_w = depth[idx].shape
        cam_info = cam_info_map.get(cam_name)
        if cam_info is not None:
            # Scale calibrated intrinsics to depth resolution.
            scale_x = depth_w / max(cam_info["W"], 1)
            scale_y = depth_h / max(cam_info["H"], 1)
            fx = float(cam_info["fx"]) * scale_x
            fy = float(cam_info["fy"]) * scale_y
            cx = float(cam_info["cx"]) * scale_x
            cy = float(cam_info["cy"]) * scale_y
        else:
            intr = intrs[idx]
            fx, fy = float(intr[0, 0]), float(intr[1, 1])
            cx, cy = float(intr[0, 2]), float(intr[1, 2])
        x = (u_d - cx) * z / fx
        y = (v_d - cy) * z / fy
        pt_cam = np.asarray([x, y, z], dtype=np.float32)

        if depth_extr_inv:
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :4] = extrs[idx]
            world_from_cam = np.linalg.inv(ext)[:3, :4]
        else:
            world_from_cam = extrs[idx]
        R = world_from_cam[:, :3]
        t = world_from_cam[:, 3]
        pt_world = R @ pt_cam + t
        points_world.append(pt_world)

    if not points_world:
        return None
    pts = np.stack(points_world, axis=0)
    if pts.shape[0] > max_points:
        choice = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[choice]
    center = pts.mean(axis=0)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = center.astype(np.float32)
    return pose


def read_rgb(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def resize_mask(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def resize_rgb(image: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def alpha_blend_rgb(base_rgb: np.ndarray, render_rgba: np.ndarray, alpha_scale: float = 0.65) -> np.ndarray:
    render_rgb = np.clip(render_rgba[..., :3], 0.0, 1.0)
    render_alpha = np.clip(render_rgba[..., 3], 0.0, 1.0) * alpha_scale
    base_float = base_rgb.astype(np.float32) / 255.0
    blended = render_rgb * render_alpha[..., None] + base_float * (1.0 - render_alpha[..., None])
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def annotate_tile(image_rgb: np.ndarray, cam_name: str, mask_pixels: int) -> np.ndarray:
    tile = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(tile, cam_name, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 255, 30), 2, cv2.LINE_AA)
    cv2.putText(tile, f"mask={mask_pixels}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 30), 2, cv2.LINE_AA)
    return tile


def make_collage(images_bgr: Sequence[np.ndarray]) -> np.ndarray:
    if not images_bgr:
        raise ValueError("Cannot make collage without images.")
    num_images = len(images_bgr)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    blank = np.zeros_like(images_bgr[0])
    padded = list(images_bgr) + [blank] * (rows * cols - num_images)
    rows_out = []
    for row_idx in range(rows):
        rows_out.append(np.concatenate(padded[row_idx * cols:(row_idx + 1) * cols], axis=1))
    return np.concatenate(rows_out, axis=0)


def quaternion_from_matrix(matrix: np.ndarray) -> torch.Tensor:
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return torch.from_numpy(quat_wxyz).unsqueeze(0)


def quaternion_to_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def resample_poses(poses: np.ndarray, target_length: int) -> np.ndarray:
    if poses.shape[0] == target_length:
        return poses.astype(np.float32)
    src_times = np.linspace(0.0, 1.0, poses.shape[0])
    dst_times = np.linspace(0.0, 1.0, target_length)
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    slerp = Slerp(src_times, rotations)
    out_rot = slerp(dst_times).as_matrix()
    out_trans = np.stack([np.interp(dst_times, src_times, poses[:, dim, 3]) for dim in range(3)], axis=1)
    out = np.tile(np.eye(4, dtype=np.float32), (target_length, 1, 1))
    out[:, :3, :3] = out_rot.astype(np.float32)
    out[:, :3, 3] = out_trans.astype(np.float32)
    return out


def load_camera_infos(calib_root: Path) -> List[Dict]:
    cameras, images, _ = colmap_utils.read_model(str(calib_root), ext=".bin")
    infos = []
    for image in images.values():
        camera = cameras[image.camera_id]
        params = camera.params
        fx, fy, cx, cy = map(float, params[:4])
        # Original implementation: COLMAP RDF -> LUF (matches PyTorch3D helper).
        w2c_rdf = np.eye(4, dtype=np.float32)
        w2c_rdf[:3, :3] = colmap_utils.qvec2rotmat(image.qvec).astype(np.float32)
        w2c_rdf[:3, 3] = np.asarray(image.tvec, dtype=np.float32)
        c2w_rdf = np.linalg.inv(w2c_rdf)
        flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        c2w_luf = c2w_rdf.copy()
        c2w_luf[:3, :3] = c2w_rdf[:3, :3] @ flip
        w2c = np.linalg.inv(c2w_luf).astype(np.float32)
        intrinsics = np.asarray([fx, fy, cx, cy], dtype=np.float32)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        infos.append(
            {
                "cam_name": Path(image.name).stem,
                "W": int(camera.width),
                "H": int(camera.height),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "w2c": torch.from_numpy(w2c),
                "intrinsics": intrinsics,
                "K": K,
                "dist_coeffs": np.zeros(4, dtype=np.float32),
            }
        )
    return sorted(infos, key=lambda info: info["cam_name"])


def load_mesh(mesh_path: Path, device: str, scale: float):
    mesh = load_objs_as_meshes([str(mesh_path)], device=device)
    if abs(scale - 1.0) > 1e-6:
        verts = mesh.verts_list()[0] * scale
        mesh = mesh.update_padded(verts[None])
    return mesh


def collect_frame_views(
    timestamp_name: str,
    parsed_root: Path,
    mask_root: Path,
    camera_infos: Sequence[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    frame_views = []
    image_dir = parsed_root / timestamp_name / "images"
    mask_dir = mask_root / timestamp_name / "object_masks"
    for camera_info in camera_infos:
        cam_name = camera_info["cam_name"]
        image_path = image_dir / f"{cam_name}.jpg"
        mask_path = mask_dir / f"{cam_name}.npz"
        if not image_path.exists() or not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        if mask is None:
            continue
        if int(mask.sum()) < args.min_mask_pixels:
            continue
        rgb = read_rgb(image_path)
        out_hw = (int(camera_info["H"] / args.down), int(camera_info["W"] / args.down))
        rgb_small = resize_rgb(rgb, out_hw)
        mask_small = resize_mask(mask, out_hw)
        frame_views.append(
            {
                "camera": camera_info,
                "cam_name": cam_name,
                "rgb": rgb_small,
                "mask": mask_small,
                "mask_pixels": int(mask_small.sum()),
            }
        )
    frame_views.sort(key=lambda item: item["mask_pixels"], reverse=True)
    if args.max_views > 0:
        frame_views = frame_views[:args.max_views]
    return frame_views


def optimize_pose_for_views(
    mesh,
    frame_views: Sequence[Dict],
    init_pose: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[np.ndarray], float, List[float]]:
    device = mesh.device
    init_quat = quaternion_from_matrix(init_pose[:3, :3]).to(device)
    init_trans = torch.from_numpy(init_pose[:3, 3].astype(np.float32)).unsqueeze(0).to(device)

    renderer_list = []
    image_ref_list = []
    for view in frame_views:
        render_setup = setup_renderer(args, view["camera"], device)
        renderer_list.append(render_setup["renderer"])
        rgb = view["rgb"]
        mask = view["mask"]
        alpha = (mask.astype(np.uint8) * 255)
        image_ref_list.append(np.dstack([rgb, alpha]))

    model = DRModel(
        meshes=mesh,
        renderer_list=renderer_list,
        image_ref_list=image_ref_list,
        anchor_T=init_trans,
        init_R=init_quat,
        lambda_rgb=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.mesh_rotation], "lr": args.lr_rot},
            {"params": [model.mesh_translation], "lr": args.lr_trans},
        ]
    )

    best_loss = float("inf")
    best_pose = init_pose.astype(np.float32).copy()
    best_renders: List[np.ndarray] = []
    loss_history: List[float] = []

    for it in range(args.iters):
        optimizer.zero_grad()
        loss_pixels_list, image_list, _ = model(it)
        if not loss_pixels_list:
            break
        loss = torch.stack([loss.mean() for loss in loss_pixels_list]).mean()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        with torch.no_grad():
            model.mesh_rotation.copy_(torch.nn.functional.normalize(model.mesh_rotation, dim=-1))

        if loss.item() < best_loss and not check_for_nan_params(model):
            best_loss = float(loss.item())
            quat = torch.nn.functional.normalize(model.mesh_rotation.detach(), dim=-1)
            rot = quaternion_to_matrix_wxyz(quat)[0].detach().cpu().numpy()
            trans = model.mesh_translation.detach()[0].cpu().numpy()
            best_pose = np.eye(4, dtype=np.float32)
            best_pose[:3, :3] = rot.astype(np.float32)
            best_pose[:3, 3] = trans.astype(np.float32)
            best_renders = []
            for img in image_list:
                rgba = img.detach().squeeze(0).cpu().numpy()
                # Render silhouette as black with alpha.
                rgba[..., :3] = 0.0
                best_renders.append(rgba)

    return best_pose, best_renders, best_loss, loss_history


def save_loss_plot(path: Path, losses: Sequence[float]) -> None:
    if not losses:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Optimization Loss per Iteration")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _ensure_even_hw(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    if new_h == h and new_w == w:
        return frame
    return frame[:new_h, :new_w]


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int = 10, frames_are_bgr: bool = True) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_ensure_even_hw(f) for f in frames]
    prepared = []
    for frame in frames:
        out = frame
        if frames_are_bgr and frame.ndim == 3 and frame.shape[2] == 3:
            out = frame[:, :, ::-1]
        prepared.append(np.ascontiguousarray(out.astype(np.uint8)))

    # Prefer imageio/ffmpeg with yuv420p for broad compatibility.
    try:
        with imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        ) as writer:
            for frame in prepared:
                writer.append_data(frame)
        return
    except Exception as exc:
        print(f"Video write failed via imageio: {exc}")

    # Final fallback: dump frames so nothing is lost.
    frame_dir = path.parent / (path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"{idx:06d}.png"), frame[:, :, ::-1])
    print(f"Wrote frames to {frame_dir} instead of video.")


def save_pose_json(path: Path, timestamp_names: Sequence[str], poses: Sequence[np.ndarray], losses: Sequence[float]) -> None:
    payload = {}
    for timestamp_name, pose, loss in zip(timestamp_names, poses, losses):
        payload[timestamp_name] = {
            "pose_matrix": pose.tolist(),
            "rotation_matrix": pose[:3, :3].tolist(),
            "translation": pose[:3, 3].tolist(),
            "loss": float(loss),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_root,
        "collages": output_root / "collages",
        "overlays": output_root / "overlays",
        "poses": output_root / "poses",
        "videos": output_root / "videos",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    args = build_argparser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    timestamp_names = sorted_timestamps(args.parsed_root)
    if args.timestamp_end >= 0:
        timestamp_names = timestamp_names[args.timestamp_start:args.timestamp_end:args.timestamp_step]
    else:
        timestamp_names = timestamp_names[args.timestamp_start::args.timestamp_step]
    if args.max_timestamps > 0:
        timestamp_names = timestamp_names[:args.max_timestamps]
    if not timestamp_names:
        raise RuntimeError("No timestamps selected.")

    camera_infos = load_camera_infos(args.calib_root)
    if not camera_infos:
        raise RuntimeError(f"No cameras loaded from {args.calib_root}.")

    init_poses = None
    if args.use_traj_init:
        pose_data = np.load(args.traj_path, allow_pickle=True)
        if args.pose_key not in pose_data.files:
            raise KeyError(f"{args.pose_key} not found in {args.traj_path}. Available keys: {pose_data.files}")
        init_poses = resample_poses(pose_data[args.pose_key], len(timestamp_names))

    mesh = load_mesh(args.mesh_path, args.device, args.mesh_scale)
    output_dirs = ensure_output_dirs(args.output_root)

    all_collages: List[np.ndarray] = []
    solved_poses: List[np.ndarray] = []
    solved_losses: List[float] = []
    solved_timestamp_names: List[str] = []
    previous_pose: Optional[np.ndarray] = None

    for idx, timestamp_name in enumerate(tqdm(timestamp_names, desc="Optimizing timestamps")):
        frame_views = collect_frame_views(timestamp_name, args.parsed_root, args.mask_root, camera_infos, args)
        if not frame_views:
            tqdm.write(f"Skipping {timestamp_name}: no valid masked views.")
            continue

        if previous_pose is not None:
            init_pose = previous_pose
        elif not args.use_traj_init:
            init_pose = depth_init_pose(
                args.depth_root,
                args.mask_root,
                timestamp_name,
                args.depth_extr_inv,
                args.depth_max_points,
                camera_infos,
            )
            if init_pose is None:
                init_pose = np.eye(4, dtype=np.float32)
        else:
            init_pose = init_poses[idx]
        pose, renders, loss, loss_history = optimize_pose_for_views(
            mesh, frame_views, init_pose, args
        )
        previous_pose = pose
        solved_timestamp_names.append(timestamp_name)
        solved_poses.append(pose)
        solved_losses.append(loss)
        save_loss_plot(output_dirs["poses"] / "loss_plots" / f"{timestamp_name}.png", loss_history)

        tiles = []
        for view, render_rgba in zip(frame_views, renders):
            overlay = alpha_blend_rgb(view["rgb"], render_rgba)
            tile = annotate_tile(overlay, view["cam_name"], view["mask_pixels"])
            tiles.append(tile)
            if args.save_per_timestamp:
                camera_dir = output_dirs["overlays"] / timestamp_name
                camera_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(camera_dir / f"{view['cam_name']}.jpg"), tile)

        collage = make_collage(tiles)
        cv2.putText(collage, f"{timestamp_name} loss={loss:.6f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        all_collages.append(collage)
        if args.save_per_timestamp:
            cv2.imwrite(str(output_dirs["collages"] / f"{timestamp_name}.jpg"), collage)

    if not solved_poses:
        raise RuntimeError("No timestamps were successfully optimized.")

    save_pose_json(output_dirs["poses"] / "optimized_poses.json", solved_timestamp_names, solved_poses, solved_losses)
    np.save(output_dirs["poses"] / "optimized_poses.npy", np.stack(solved_poses))
    write_video(output_dirs["videos"] / "multiview_pose_collage.mp4", all_collages, fps=10, frames_are_bgr=True)

    print(f"Saved {len(solved_poses)} optimized poses to {output_dirs['poses'] / 'optimized_poses.json'}")
    print(f"Saved collage video to {output_dirs['videos'] / 'multiview_pose_collage.mp4'}")


if __name__ == "__main__":
    main()
