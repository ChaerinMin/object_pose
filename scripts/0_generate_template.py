import os
import argparse
import pytorch3d

import sys
sys.path.append(".")
from src.utils.pytorch3d_utils import setup_renderer, uniform_quaternions
import src.utils.colmap_utils as colmap_utils
import src.utils.dinov2_utils as dinov2_utils
from src.utils.template_util import crop_image, images_to_template_reps, match_image_with_template_reps

sys.path.append("./thirdparty/dinov2")

import json
from PIL import Image
import numpy as np
import torch
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesAtlas,
    TexturesVertex,
    BlendParams
)
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply, save_obj
from pytorch3d.transforms import Rotate, Translate, axis_angle_to_matrix, quaternion_to_axis_angle, matrix_to_axis_angle


def batch_render_loader(args, cameras, device):
    renderer_list = []
    for i, camera in enumerate(cameras[:]): # only render the first camera
        render_setup = setup_renderer(args, camera, device)
        renderer_list.append(render_setup)
    return renderer_list



parser = argparse.ArgumentParser()
parser.add_argument('--session', type=str, default='2024-07-01-02-action-instruments')
parser.add_argument('--save_dir', default='test/ukelele/template_02', help='')
parser.add_argument('--down', default=4, type=float, help='downsample image size')
parser.add_argument('--camera_type', default='perspective', help='')
parser.add_argument('--texture_type', default='ply', help='single|multi|ply')
parser.add_argument('--shader', default='phong', help='phong|mask')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--normalize_verts', default=False, action='store_true')
args = parser.parse_args()

scanned_mesh_path = "/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-09-21_session_snapshot-object-instruments/scans/ukelele_scan/AR-Code-Object-Capture-app-1727202177-simplified.obj"
# scanned_mesh_path = "/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-09-21_session_snapshot-object-instruments/scans/keyboard/keyboard-simplified.obj"
input_path = f'/users/rfu7/ssrinath/brics/non-pii/brics-mini/{args.session}'
out_dir = f'/users/rfu7/ssrinath/datasets/Action/brics-mini/{args.session}'
params_path = os.path.join(out_dir, "optim_params.txt")

save_json_folder = os.path.join(args.save_dir, 'metadata')
save_image_folder = os.path.join(args.save_dir, 'imgs')
os.makedirs(save_json_folder, exist_ok = True)
os.makedirs(save_image_folder, exist_ok = True)

device = "cuda" if torch.cuda.is_available() else "cpu"

""" load camera """
render_cameras = colmap_utils.read_cameras_from_txt(params_path)


""" STEP 1: Generate Templates """
""" load anchor mesh """
""" initialize T  """
verts, faces = load_ply(f"test/ukelele/{args.session}_0028/mesh/ngp_mesh/000000.ply")
textures = TexturesVertex(verts_features=torch.zeros_like(verts)[None])
mesh = Meshes([verts], [faces], textures=textures).to("cuda")
mesh_translation = torch.mean(torch.stack(mesh.verts_list()), dim=1).to(device) #[1, 3]

""" load scanned mesh """
template_mesh = load_objs_as_meshes([scanned_mesh_path], device='cuda')

""" load renders """
renderer_list = batch_render_loader(args, render_cameras, device)

# """ initialize random views """
num_initializations = 50
random_quats = uniform_quaternions(num_initializations)
rotation_matrices = quaternion_to_axis_angle(random_quats)

""" initialize random views """
for mid, mesh_rotation in enumerate(rotation_matrices):

    # Render the image using the updated camera position. Based on the new position of the 
    # camera we calculate the rotation and translation matrices
    # R = Rotate(euler_angles_to_matrix(self.mesh_rotation, convention='XYZ'))
    R = Rotate(axis_angle_to_matrix(mesh_rotation.to(device)))
    T = Translate(torch.clamp(mesh_translation, min=-0.2, max=1.2))   # (1, 3)
    transform  = R.compose(T)
    tverts = transform.transform_points(template_mesh.verts_list()[0])
    faces = template_mesh.faces_list()[0]
    tmesh = Meshes(
        verts=[tverts],   
        faces=[faces],
        textures = mesh.textures,
    )

    json_file_path = os.path.join(save_json_folder, str(mid).zfill(6) + '.json')
    image_folder = os.path.join(save_image_folder, str(mid).zfill(6))
    os.makedirs(image_folder, exist_ok = True)
    
    mid_info = {'mesh_rotation': mesh_rotation.cpu().numpy().tolist()}
    for rid, render_info in enumerate(renderer_list):
        cam_name =render_info['cam_name']
        image_tensor, fragments = render_info['renderer'](meshes_world=tmesh)
        
        image_save_path = os.path.join(image_folder, cam_name + '.png')
        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np, 'RGBA')
        # image.save(image_save_path)
        mid_info[cam_name] = image_save_path
    # with open(json_file_path, 'w') as json_file:
    #     json.dump(mid_info, json_file, indent=4) 



""" STEP 2: Generate Representations """
all_init_scores = [0] * num_initializations
all_init_freq = [0] * num_initializations
all_init_dist = [0] * num_initializations
extractor = dinov2_utils.DinoFeatureExtractor(model_name="dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1").to(device)
best_view_score = []
for camera in render_cameras:
    anchor_cam_name = camera['cam_name']
    """ initialize random views """
    template_img_paths = []
    for mid in range(num_initializations):
        json_file_path = os.path.join(save_json_folder, str(mid).zfill(6) + '.json')
        image_folder = os.path.join(save_image_folder, str(mid).zfill(6))
        image_save_path = os.path.join(image_folder, anchor_cam_name + '.png')
        template_img_paths.append(image_save_path)

    valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs = images_to_template_reps(template_img_paths, extractor, device)
    if valid_template_ids is None:
        continue
    # Load and crop instance image
    instance_image_path = f'test/ukelele/{args.session}_0028/segmentation/{anchor_cam_name}/000000.png'
    if not os.path.isfile(instance_image_path):
        continue
    
    template_scores, sorted_template_ids, valid_template_ids = match_image_with_template_reps(instance_image_path, extractor, valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs, device)
    for s, sid in zip(template_scores, sorted_template_ids):
        tid = valid_template_ids[sid]
        if s > 0.2:
            all_init_scores[tid] += s
            all_init_freq[tid] += 1

all_init_scores = torch.tensor(all_init_scores)
sorted_indices = torch.argsort(all_init_scores)
sorted_scores = all_init_scores[sorted_indices]
print("Sorted indices:", sorted_indices)
print("Sorted scores:", sorted_scores)
all_init_freq = torch.tensor(all_init_freq)
sorted_indices = torch.argsort(all_init_freq)
sorted_freq = all_init_freq[sorted_indices]
print("Sorted indices:", sorted_indices)
print("Sorted freq:", sorted_freq)

# for view, score in best_view_score:
#     print(view, score)