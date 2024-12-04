import os
import sys
import numpy as np

sys.path.append("../build")  # build folder of instant-NGP
sys.path.append("../scripts")
import time
import json
import trimesh
import pandas as pd
from copy import deepcopy
import cv2
import math
from tqdm import tqdm, trange
import imageio.v2 as imageio
import struct
from natsort import natsorted
import glob
import pickle
import csv
import shutil
from sklearn.cluster import DBSCAN
from collections import Counter

def add_npg_parser(parser):
    parser.add_argument("--network", type=str, required=False, default='./data/nerf_base.json')
    parser.add_argument("--action", type=str, default="")
    parser.add_argument("--cam_traj_path", type=str, default="")
    parser.add_argument("--separate_calib", action="store_true")
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32512)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--overwrite_segmentation", action="store_true")
    parser.add_argument("--aabb_scale", type=int, default=1)
    parser.add_argument("--num_objects", type=int, default=1)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--pad", nargs="+", default=[0.05, 0.05, 0.05], type=float)

    parser.add_argument(
        "--mesh_path",
        default="mesh.ply",
        help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.",
    )
    parser.add_argument(
        "--marching_cubes_res",
        default=256,
        type=int,
        help="Sets the resolution for the marching cubes grid.",
    )

    parser.add_argument("--optimize_extrinsics", default=True, action="store_true")
    parser.add_argument("--optimize_focal_length", action="store_true")
    parser.add_argument("--optimize_distortion", action="store_true")
    parser.add_argument("--save_segmented_images", action="store_true")
    parser.add_argument("--save_raw_density", action="store_true")
    parser.add_argument("--align_bounding_box", default=True, action="store_true")
    parser.add_argument(
        "--face_to_cam_path", type=str, default="./metadata/faceToCam.json"
    )
    parser.add_argument("--downscale_factor", type=float, default=0.45)
    parser.add_argument("--face_to_cam", action="store_true")
    parser.add_argument("--cam_faces_path", type=str, default="./data/faces_v2.json")
    parser.add_argument("--save_dir_name", type=str, default="camera_check")


def sort_frames_by_mask_area(all_view_masks, num_frames):
    frame_sums = []

    # Loop over each frame ID
    for frame_id in range(num_frames):
        total_mask_sum = 0

        # Sum up valid mask areas across all views for this frame
        for view_masks in all_view_masks.values():
            mask = view_masks[frame_id]
            if mask is not None:
                total_mask_sum += mask.sum()
        
        # Only consider frames with a non-zero mask area sum
        if total_mask_sum > 0:
            frame_sums.append((frame_id, total_mask_sum))

    # Sort frames by the total mask area sum in descending order
    sorted_frames = sorted(frame_sums, key=lambda x: x[1], reverse=True)

    # Extract just the frame IDs in the sorted order
    sorted_frame_ids = [frame_id for frame_id, _ in sorted_frames]

    return sorted_frame_ids
    

def get_aabb_min_max(aabb_min, aabb_max, downscale_factor):
    roi = np.array([0.5, 0.5, 0.5])
    size = 1 - downscale_factor
    mx = np.minimum(1.0 * np.ones(3), ((np.ones(3) * roi) + size / 2))
    mn = np.maximum(0.0 * np.ones(3), ((np.ones(3) * roi) - size / 2))
    
    aabb_min = np.max([mn, aabb_min], axis = 0)
    aabb_max = np.min([mx, aabb_max], axis = 0)
    return aabb_min, aabb_max

def connected_components(th):
    num_labels, labels_im = cv2.connectedComponents(th)
    s = 0
    ind = 1
    if num_labels > 2:
        for nl in range(1, num_labels):
            temp = np.sum((labels_im == nl).astype(np.uint8))
            if temp > s:
                s = temp
                ind = nl
    mask = (labels_im == ind) * 255
    return mask

def process_mask(
    alpha,
    opening_kernel,
    thresh,
    con_comp=False,
    use_alpha=False,
    dilate_mask=False,
    contours=False,
    contour_thresh=100,
):
    mask = alpha.copy()
    mask = (mask > thresh) * 255
    mask = mask.astype(np.uint8)

    # mask = cv2.morphologyEx(
    #     (mask * 255).astype(np.uint8), cv2.MORPH_OPEN, opening_kernel
    # )

    if con_comp:
        mask = connected_components((mask * 255).astype(np.uint8))

    if dilate_mask:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), opening_kernel, iterations=1)

    if contours:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_TC89_L1,
        )

        # Choose contours with are greater than the given threshold
        chosen = []
        for contour in contours:
            if cv2.contourArea(contour) > contour_thresh:
                chosen.append(contour)
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, chosen, -1, 255, -1)
        mask = new_mask

    if use_alpha:
        mask = mask * alpha

    return mask
    
def save_mask(testbed, imgs, img_names, params, output_path, num_objects = 1):
    testbed.nerf.render_with_lens_distortion = False
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    width= 1280
    height = 720
    
    for idx in tqdm(range(len(imgs))):
        testbed.set_camera_to_training_view(idx)
        frame = testbed.render(width, height, 1, True)
        alpha = frame[:, :, 3]

        con_comp = True if num_objects == 1 else False 
        mask = process_mask( alpha, opening_kernel, 0.8, con_comp, False, False, False, None)
        mask = mask.astype(np.float32) 
        img = deepcopy(imgs[idx])
        img = img * 255
        img[..., 3] = mask

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        output_dir = os.path.join(output_path, params[idx]["cam_name"])
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(
                output_dir, f"{img_names[idx]}.png"
            ),
            img,
        )

def save_raw_density_npy(res, testbed, path, filter_density, sigma_thresh, frame_id=0):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.shape[0] % (res * res * res) != 0:
        print("InstantNGP Density Format failed")
        return False, None
    raw_sigma = raw.reshape(res, res, res, -1)[:, :, :, -1]

    s = 1.0
    xmin, xmax = 0, s
    ymin, ymax = 0, s
    zmin, zmax = 0, s
    xdists = np.linspace(xmin, xmax, res)
    ydists = np.linspace(ymin, ymax, res)
    zdists = np.linspace(zmin, zmax, res)
    coords = np.stack(
        np.meshgrid(xdists, ydists, zdists, indexing="ij"), axis=-1
    ).astype(np.float32)
    

    samples = coords.reshape((res, res, res, 3))
    scale = (testbed.render_aabb.max - testbed.render_aabb.min)
    offset = testbed.render_aabb.min
    
    samples = samples * scale + offset
    
    
    sigma = 1 - np.exp(-1.0 * raw_sigma * raw_sigma)
    sigma = np.clip(sigma, 0, 1)
    pts = samples[sigma >= sigma_thresh, :]
    volume_dir = os.path.dirname(path)
    filter_path = os.path.join(volume_dir, f"{str(frame_id).zfill(6)}_filtered_samples.ply")
    point_cloud = trimesh.PointCloud(pts)
    point_cloud.export(filter_path)

    if pts.shape[0] > 0: # if there are high density regions.
        center = np.array([0.5, 0.5, 0.5])
        point_cloud.vertices -= center
        # Define the rotation matrices
        rotation_z_180 = np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1]
        ])

        rotation_y_90 = np.array([
            [ 0,  0,  1],
            [ 0,  1,  0],
            [-1,  0,  0]
        ])

        # Combine the two rotations by multiplying the matrices
        combined_rotation = rotation_y_90 @ rotation_z_180
        point_cloud.vertices = point_cloud.vertices @ combined_rotation.T
        point_cloud.vertices += center

        # Define the filtering conditions
        x_min, x_max = 0.15, 0.80
        y_min, y_max = 0.33, 0.70
        z_min, z_max = 0.10, 0.90
        # Create a mask for vertices that meet the conditions
        vertex_mask = (point_cloud.vertices[:, 1] >= y_min) & (point_cloud.vertices[:, 1] <= y_max) & (point_cloud.vertices[:, 2] >= z_min) & (point_cloud.vertices[:, 2] <= z_max)  & (point_cloud.vertices[:, 0] >= x_min) & (point_cloud.vertices[:, 0] <= x_max)
        filtered_vertices = point_cloud.vertices[vertex_mask]

        if filtered_vertices.shape[0] > 0: # if there are enough points in the bounded region
            # Create a new PointCloud with the filtered vertices
            filtered_point_cloud = trimesh.PointCloud(filtered_vertices)
            output_path = os.path.join(volume_dir, f"{str(frame_id).zfill(6)}_filtered_bounded_samples.ply")
            filtered_point_cloud.export(output_path)
            points = filtered_vertices

            # Apply DBSCAN clustering
            # Adjust 'eps' and 'min_samples' as needed based on the point cloud structure
            dbscan = DBSCAN(eps=0.02, min_samples=10).fit(points)
            labels = dbscan.labels_
            cluster_counts = Counter(labels[labels != -1])
            if not cluster_counts:
                print("No dense clusters found in the InstantNGP point cloud.")
                return False, None
            else:
                # Find the densest cluster (cluster with the most points)
                densest_cluster_label = max(cluster_counts, key=cluster_counts.get)
                densest_cluster_points = points[labels == densest_cluster_label]

                # Print information about the densest cluster
                print(f"Densest cluster label: {densest_cluster_label}")
                print(f"Number of points in the densest cluster: {len(densest_cluster_points)}")
                center = np.mean(densest_cluster_points, axis=0)

                # Optionally, save the densest cluster as a new point cloud file
                densest_cluster_cloud = trimesh.PointCloud(densest_cluster_points)
                output_path = os.path.join(volume_dir, f"{str(frame_id).zfill(6)}_filtered_bounded_clutered_samples.ply")
                densest_cluster_cloud.export(output_path)

                return True, densest_cluster_points
        else:
            print("InstantNGP reconstruction failed in bounded range")
            return False, None
    else:
        print("InstantNGP reconstruction failed")
        return False, None

        
            
            

def save_raw_density(testbed, res, output_file, frame_id):
    resolution = np.array([res, res, res])
    
    volume_dir = os.path.join(output_file, "volume_raw")
    os.makedirs(volume_dir, exist_ok=True)
    testbed.save_raw_density_grid(
        filename =volume_dir,
        resolution=resolution,
        aabb_min=testbed.render_aabb.min,
        aabb_max=testbed.render_aabb.max,
    )
    volume_name = f"{res}x{res}x{res}_0.bin"
    volume_path = os.path.join(volume_dir, volume_name)
    shutil.copy(f'volume_raw/{volume_name}', volume_path)
    
    success, vertices =save_raw_density_npy(
        res, testbed, volume_path, filter_density=True, sigma_thresh=0.99999999, frame_id=frame_id
    )
    return success, vertices
    
def generate_test_video(testbed, args, output_dir, cam_traj_path, aabb_min, aabb_max):
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    
    testbed.snap_to_pixel_centers = True
    spp =1

    # testbed.nerf.render_min_transmittance = 1e-4
    testbed.shall_train = False

    f = open(cam_traj_path)
    test_params = json.load(f)
    f.close()

    n_frames = len(test_params)

    rendered_frames = []
    cache_size = 100  # Looks good for 2080 Ti
    step = math.ceil(n_frames / cache_size)
    start = 0
    
    for sp in trange(step):
        rendered_frames.extend(
            get_test_renders(
                args,
                testbed,
                test_params,
                start=start,
                cache_size=cache_size,
                max_frames=n_frames,
                spp=spp,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
            )
        )
        start = start + cache_size

    for idx, image in tqdm(enumerate(rendered_frames)):
        alpha = image[..., 3:4]
        image = linear_to_srgb(image[..., 0:3])
        final_image = np.concatenate([image, alpha], axis=-1) * 255
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(output_dir, f"{idx:04d}.png"), final_image)
        
    # frameSize = (1280, 720)
    # fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # output_path = os.path.join(output_dir, "gt_contacts.mp4")
    # out = cv2.VideoWriter(output_path, fourcc, 30.0, frameSize)

    # for image in tqdm(rendered_frames):
    #     image = linear_to_srgb(image[..., 0:3])
    #     image = image[..., ::-1] * 255.0
    #     out.write(image.astype(np.uint8))
    # out.release()




def write_csv(csv_path, row, include_header=False):
    with open(csv_path, "w+") as csvfile:
        # Add a new row to the end of the file
        writer = csv.writer(csvfile, delimiter=",")
        if include_header:
            writer.writerow(
                [
                    "name",
                    "psnr",
                    "min-psnr",
                    "max-psnr",
                    "ssim",
                    "lpips",
                    "training_time",
                    "rendering_time",
                    "device_name",
                    "val_id",
                    "skip_cams",
                    "train_cams",
                ]
            )
        writer.writerow(row)
        csvfile.close()


def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)



def write_image_imageio(img_file, img, quality):
    img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    kwargs = {}
    if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
        if img.ndim >= 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        kwargs["quality"] = quality
        kwargs["subsampling"] = 0
    imageio.imwrite(img_file, img, **kwargs)


def write_image(file, img, quality=95):
    if os.path.splitext(file)[1] == ".bin":
        if img.shape[2] < 4:
            img = np.dstack(
                (img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]]))
            )
        with open(file, "wb") as f:
            f.write(struct.pack("ii", img.shape[0], img.shape[1]))
            f.write(img.astype(np.float16).tobytes())
    else:
        if img.shape[2] == 4:
            img = np.copy(img)
            # Unmultiply alpha
            img[..., 0:3] = np.divide(
                img[..., 0:3],
                img[..., 3:4],
                out=np.zeros_like(img[..., 0:3]),
                where=img[..., 3:4] != 0,
            )
            img[..., 0:3] = linear_to_srgb(img[..., 0:3])
        else:
            img = linear_to_srgb(img)
        write_image_imageio(file, img, quality)


def read_pickle_file(path):
    objects = []
    with open(path, "rb") as fp:
        while True:
            try:
                obj = pickle.load(fp)
                objects.append(obj)

            except EOFError:
                break

    return objects


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg


def load_dataset(directory):
    # print(directory)

    cam_data_path = os.path.join(directory, "cam_data.pkl")
    cam_data = read_pickle_file(cam_data_path)[0]
    cams = {"width": 1280, "height": 720}

    imgs = {}
    image_dir = os.path.join(directory, "render/")
    images = glob.glob(image_dir + "**/*.png", recursive=True)
    images.sort()

    mask_dir = os.path.join(directory, "mask/")
    depth_dir = os.path.join(directory, "depth/")

    for i in range(len(images)):
        image_current = images[i]
        image_id = os.path.basename(image_current).split(".")[0]
        image_parent_dir = image_current.split("/")[-2]
        # import pdb; pdb.set_trace()
        cam = cam_data[image_id]["K"]
        [cams["fx"], cams["fy"], cams["cx"], cams["cy"]] = cam
        pose = cam_data[image_id]["extrinsics_blender"]
        # pose = np.vstack([pose, np.array([0, 0, 0, 1])])
        # pose = np.linalg.inv(pose)
        # print(pose)

        imgs[i] = {
            "camera_id": image_id,
            "t": pose[:3, 3].reshape(3, 1),
            "R": pose[:3, :3],
            "path": images[i],
            "pose": pose,
            "fx": cams["fx"],
            "fy": cams["fy"],
            "cx": cams["cx"],
            "cy": cams["cy"],
        }

        imgs[i]["mask_path"] = os.path.join(
            mask_dir, "%s/%s_seg.png" % (image_parent_dir, image_id)
        )
        imgs[i]["depth_path"] = os.path.join(
            depth_dir, "%s/%s_depth.npz" % (image_parent_dir, image_id)
        )

    return imgs


def get_test_renders(
    args,
    testbed,
    test_params,
    start=0,
    cache_size=100,
    max_frames=250,
    spp=32,
    aabb_min=[0.2, 0.2, 0.2],
    aabb_max=[0.8, 0.8, 0.8],
):
    end = min(start + cache_size, max_frames)
    cache_size = end - start
    testbed.create_empty_nerf_dataset(n_images=cache_size, aabb_scale=args.aabb_scale)
    names = []
    testbed.nerf.training.n_images_for_training = cache_size

    testbed.render_aabb.min = aabb_min
    testbed.render_aabb.max = aabb_max

    cached_frames = []
    for idx, _ in tqdm(enumerate(range(cache_size))):
        trans = test_params[start + idx]["transform_matrix"]
        trans = np.asarray(trans)
        trans = np.concatenate([trans, np.array([[0, 0, 0, 1]])])
        
        fl_x = test_params[start + idx]["fl_x"]
        fl_y = test_params[start + idx]["fl_y"]
        cx = test_params[start + idx]["cx"]
        cy = test_params[start + idx]["cy"]
        height = int(test_params[start + idx]["h"])
        width = int(test_params[start + idx]["w"])

        img = np.zeros((height, width, 4))
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        testbed.nerf.training.set_image(idx, img, depth_img)

        scale = testbed.nerf.training.dataset.scale
        offset = testbed.nerf.training.dataset.offset
        
        trans = trans[:3, :4].tolist()
        
        testbed.nerf.training.set_camera_extrinsics(idx, trans, convert_to_ngp=False)

        testbed.nerf.training.set_camera_intrinsics(
            idx, fx=fl_x, fy=fl_y, cx=cx, cy=cy, k1=0, k2=0, p1=0, p2=0
        )
        testbed.set_camera_to_training_view(idx)
        image = testbed.render(width, height, 1, True)
        cached_frames.append(image)
        
    # testbed.init_window(1920, 1080)
    # while testbed.frame():
    #     if testbed.want_repl():
    #         ipdb.set_trace()
    return cached_frames


def clean_mesh(mesh_path, largest_component=True):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    # ms.remove_isolated_pieces_wrt_diameter() #for pymeshlab 0.2.1
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=100)
    if largest_component:
        ms.generate_splitting_by_connected_components()
        max_face_num = max(ms[i].face_number() for i in range(1, ms.number_meshes()))
        for i in range(1, ms.number_meshes()):
            if ms[i].face_number() == max_face_num:
                ms.set_current_mesh(i)
    ms.save_current_mesh(mesh_path)


def clean_density(density_path):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(density_path)
    # ms.surface_reconstruction_ball_pivoting(clustering = 40) #for pymeshlab 0.2.1
    ms.generate_surface_reconstruction_ball_pivoting(clustering=40)
    ms.save_current_mesh(density_path)


def save_mesh(res, mesh_path, testbed, downscale_factor = 0.0, pad=[0.07, 0.07, 0.07], aabb_mn = None, aabb_mx = None, delete=False, refine = False, num_objects = 1):
    print(
        f"Generating mesh via marching cubes and saving to {mesh_path}. Resolution=[{res},{res},{res}]"
    )
    
    if aabb_mn is not None and aabb_mx is not None:
        testbed.render_aabb.min = aabb_mn
        testbed.render_aabb.max = aabb_mx
        res = 128
        
        testbed.compute_and_save_marching_cubes_mesh(mesh_path, [res, res, res], thresh=0.9999)
        largest_component = True if num_objects == 1 else False
        clean_mesh(mesh_path, largest_component=largest_component)
        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        scale = testbed.nerf.training.dataset.scale
        offset = testbed.nerf.training.dataset.offset
        mesh.vertices = mesh.vertices * scale + offset
        mesh.export(mesh_path)
        return 
    else:  
        coarse_mesh_path = mesh_path.replace(".ply", "_coarse.ply") 
        testbed.compute_and_save_marching_cubes_mesh(coarse_mesh_path, [res, res, res], thresh=0.9999)
        largest_component = True if num_objects == 1 else False
        clean_mesh(coarse_mesh_path, largest_component=largest_component)

    mesh = trimesh.load(coarse_mesh_path, process=False, maintain_order=True)
    if not mesh.is_empty:
        vertices = mesh.vertices
        scale = testbed.nerf.training.dataset.scale
        offset = testbed.nerf.training.dataset.offset
        vertices = vertices * scale + offset
        mesh.vertices = vertices
        mesh.export(coarse_mesh_path)

        pad = np.array(pad)
        min_vert = np.asarray(vertices.min(0) - pad)
        max_vert = np.asarray(vertices.max(0) + pad)
        aabb_min, aabb_max = get_aabb_min_max(min_vert, max_vert, downscale_factor)
        testbed.render_aabb.min = aabb_min 
        testbed.render_aabb.max = aabb_max
        # os.remove(coarse_mesh_path)

        if refine:
            ## Generate the fine mesh now
            res = 128  ##Use a low resolution because boundary is tighter
            testbed.compute_and_save_marching_cubes_mesh(mesh_path, [res, res, res], thresh=0.9999)
            largest_component = True if num_objects == 1 else False
            clean_mesh(mesh_path, largest_component=largest_component)
            mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
            mesh.vertices = mesh.vertices * scale + offset
            mesh.export(mesh_path)

        if delete:
            os.remove(coarse_mesh_path)
            os.remove(mesh_path)
    else:
        min_vert = np.ones((3)) * 0.2
        max_vert = np.ones((3)) * 0.8
        testbed.render_aabb.min = min_vert
        testbed.render_aabb.max = max_vert

    return mesh.vertices, mesh.faces