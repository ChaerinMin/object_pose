import os
import numpy as np
import src.utils.params as param_utils

from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

SIDE_CAMERAS = ["brics-odroid-004_cam0", "brics-odroid-005_cam1"] # camera at the left and right of the hand panel
BOTTOM_SIDE_CAMERAS = [ # camera at the bottom
                  "brics-odroid-003_cam0",
                  "brics-odroid-003_cam1",
                  "brics-odroid-004_cam0"
                  "brics-odroid-008_cam0",
                  "brics-odroid-008_cam1",
                  "brics-odroid-009_cam0",
                  "brics-odroid-013_cam0",
                  "brics-odroid-013_cam1",
                  "brics-odroid-014_cam0",
                  "brics-odroid-018_cam0",
                  "brics-odroid-018_cam1",
                  "brics-odroid-019_cam0",
                ]

BOTTOM_BOTTOM_CAMERAS = [ # camera at the bottom
                  "brics-odroid-026_cam0",
                  "brics-odroid-026_cam1",
                  "brics-odroid-027_cam0",
                  "brics-odroid-027_cam1",
                  "brics-odroid-028_cam0",
                  "brics-odroid-029_cam0",
                  "brics-odroid-029_cam1",
                  "brics-odroid-030_cam0",
                  "brics-odroid-030_cam1",
                ]
# # Note: For text data only. Since partial camera is not functional well.
# IGNORE_CAMERAS = [
#     "brics-odroid-009_cam1",
#     "brics-odroid-010_cam1",
#     "brics-odroid-024_cam1",
#     "brics-odroid-025_cam1",
# ]

def removed_cameras(remove_side=False, remove_bottom=False, ignored_cameras=None):
    to_remove = []
    if ignored_cameras:
        IGNORE_CAMERAS = ignored_cameras
    else:
        IGNORE_CAMERAS = []
    if remove_side:
        to_remove = to_remove + SIDE_CAMERAS + BOTTOM_SIDE_CAMERAS + IGNORE_CAMERAS
    if remove_bottom:
        to_remove = to_remove + SIDE_CAMERAS + BOTTOM_BOTTOM_CAMERAS + IGNORE_CAMERAS
    return to_remove

def map_camera_names(base_dir, name_list):
    """
    Maps each name in the name_list to a subdirectory in base_dir that starts with the name.

    :param base_dir: The directory to search for subdirectories.
    :param name_list: A list of names to map to subdirectories.
    :return: A dictionary mapping each name in name_list to a matching subdirectory in base_dir.
    """
    # Find all subdirectories in the base directory
    subdirs = [d.split('.')[0] for d in os.listdir(base_dir)]

    # Create a dictionary to map names in name_list to subdirectories
    name_map = {}

    for name in name_list:
        # Find a subdirectory that starts with the name
        matched_subdir = next((subdir for subdir in subdirs if subdir.startswith(name)), None)
        
        if matched_subdir:
            name_map[name] = matched_subdir

    return name_map

def get_projections(args, params, cam_names, cam_mapper, easymocap_format=False):
    # Gets the projection matrices and distortion parameters
    projs = []
    intrs = []
    dist_intrs = []
    dists = []
    rot = []
    trans = []

    for param in params:
        if (param["cam_name"] in cam_names) and (param["cam_name"] in cam_mapper):
            extr = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            r, t = param_utils.get_rot_trans(param)

            rot.append(r)
            trans.append(t)

            intrs.append(intr.copy())
            
            dist_intrs.append(intr.copy())

            projs.append(intr @ extr)
            dists.append(dist)
    if easymocap_format:
        # Easy Mocap for 3D keypoints
        cameras = { 
            'K': np.asarray(intrs),
            'R': np.asarray(rot), 
            'T': np.asarray(trans),
            'dist': np.asarray(dists),
            'P': np.asarray(projs) }
    elif args.undistort:
        cameras = { 'K': np.asarray(dist_intrs),
                    'R': np.asarray(rot), 
                    'T': np.asarray(trans) }
    else:
        cameras = { 'K': np.asarray(intrs),
                    'R': np.asarray(rot), 
                    'T': np.asarray(trans) }
    
    return intrs, np.asarray(projs), dist_intrs, dists, cameras

def get_ngp_cameras(args, params, cam_names, cam_mapper, faces):
    cam2idx = {}
    pos = []
    rot = []
    intrs = []
    dists = []
    c2ws = []

    for idx, param in enumerate(params):
        if (param["cam_name"] in cam_names) and (param["cam_name"] in cam_mapper):
            w2c = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            w2c = np.vstack((w2c, np.asarray([[0, 0, 0, 1]])))
            c2w = np.linalg.inv(w2c)
            cam2idx[param["cam_name"]] = idx
            intrs.append(intr)
            dists.append(dist)
            pos.append(c2w[:3, 3])
            rot.append(c2w[:3, :3])
            c2ws.append(c2w)
    extrs = np.array(c2ws)
    # pos = np.stack(pos)
    # rot = np.stack(rot)
    # center = pos.mean(axis=0)
    # max_dist = cdist(pos, pos).max()

    # # Move center of scene to [0, 0, 0]
    # pos -= center

    # axs = np.zeros((3, 3))

    # # Rotate to align bounding box
    # for idx, dir_ in enumerate(
    #     [
    #         ["1 0 0", "-1 0 0"],
    #         ["0 1 0", "0 -1 0"],
    #         ["0 0 1", "0 0 -1"],
    #     ]
    # ):
    #     avg1 = []
    #     for camera in faces[dir_[0]]["cameras"]:
    #         try:
    #             avg1.append(pos[cam2idx[camera]])
    #         except:
    #             pass

    #     avg2 = []
    #     for camera in faces[dir_[1]]["cameras"]:
    #         try:
    #             avg2.append(pos[cam2idx[camera]])
    #         except:
    #             pass

    #     axs[idx] = np.asarray(avg1).mean(axis=0) - np.asarray(avg2).mean(axis=0)
    #     axs[idx] /= np.linalg.norm(axs[idx])

    # # Get closest orthormal basis
    # u, _, v = np.linalg.svd(axs)
    # orth_axs = u @ v

    # new_pos = (orth_axs @ pos.T).T
    # new_rot = orth_axs @ rot

    # # Scale to fit diagonal in unity cube
    # scale_factor = np.sqrt(2) / max_dist * args.camera_scale
    # new_pos *= scale_factor

    # # Move center of scene to [0.5, 0.5, 0.5]
    # new_pos += 0.5

    # extrs = np.zeros((new_pos.shape[0], 4, 4))
    # extrs[:, :3, :3] = new_rot
    # extrs[:, :3, 3] = new_pos
    # extrs[:, 3, 3] = 1

    return intrs, extrs, dists