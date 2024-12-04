import logging
import argparse
import io
import os
import sys
import contextlib
import cv2
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import img_as_ubyte
import imageio

import torch
import torch.nn as nn
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply, save_obj

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

from pytorch3d.transforms import Rotate, Translate, matrix_to_quaternion, quaternion_to_matrix, euler_angles_to_matrix, axis_angle_to_matrix

def setup_renderer(args, camera, device, to_load_extr=True):
    # Initialize a camera.
    # print(camera)
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    cam_name = camera['cam_name']
    c2w = torch.inverse(camera['w2c']) # to c2w
    R, T = c2w[:3, :3], c2w[:3, 3:]
    R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to LUF for Rotation

    new_c2w = torch.cat([R, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
    R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
    R = R[None] # batch 1 for rendering
    T = T[None] # batch 1 for rendering
    if not to_load_extr:
        R = torch.eye(3)[None]
        T = torch.zeros(3)[None]

    """ Downsample images size for faster rendering """
    H, W = camera['H'], camera['W']
    H, W = int(H / args.down), int(W / args.down)
    K, dist_coeffs = camera['K'], camera['dist_coeffs']
    intrinsics = camera['intrinsics'] / args.down

    image_size = ((H, W),)  # (h, w)
    fcl_screen = ((intrinsics[0], intrinsics[1]),)  # fcl_ndc * min(image_size) / 2
    prp_screen = ((intrinsics[2], intrinsics[3]), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
    cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, R=R, T=T, device=device)

    # Define the settings for rasterization and shading.
    blend_params = BlendParams(sigma=1e-8, gamma=1e-8, background_color=(0.0, 0.0, 0.0))

    raster_settings = RasterizationSettings(
        image_size=image_size[0],
        blur_radius=np.log(1. / 1e-8 - 1.) * blend_params.sigma, 
        faces_per_pixel=50,
        max_faces_per_bin=50 * 1000,
        perspective_correct=False
    )

    lights = AmbientLights(device=device)
    #lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

    # Create a Phong renderer by composing a rasterizer and a shader.
    if args.shader == 'mask':
        shader = pytorch3d.renderer.SoftSilhouetteShader(blend_params=blend_params)
        print('Use mask SoftSilhouetteShader shader')
    else:
        shader = SoftPhongShader(
            blend_params=blend_params,
            device=device,
            cameras=cameras,
            lights=lights
        )

    renderer = MeshRendererWithFragments(
        rasterizer = rasterizer,
        shader=shader
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'lights': lights,
            'rasterizer': rasterizer, 'renderer': renderer, 'camera_R': R, 'camera_T': T, 'K': K, 'dist_coeffs': dist_coeffs, 'cam_name': cam_name}

    return render_setup

def visualize_image_list(image_list, save_alpha=False):
    # Calculate the number of rows and columns
    num_images = len(image_list)
    row_num = int(math.floor(math.sqrt(num_images)))  # Largest integer <= sqrt
    col_num = math.ceil(num_images / row_num)  # Calculate the column number

    # Find the size of the first image and resize others to match (assume all images have the same size)
    if len(image_list[0].shape) == 4:
        _, img_height, img_width, channels = image_list[0].shape
    elif len(image_list[0].shape) == 3 and image_list[0].shape[-1] == 4:
        img_height, img_width, channels = image_list[0].shape
    else:
        _, img_height, img_width = image_list[0].shape
        channels = 1

    # If images are not the same size, resize them
    resized_images = [cv2.resize(img.detach().squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else img, (img_width, img_height)) for img in image_list]

    # Add blank images (black) if the number of images is not a perfect grid
    num_blanks = row_num * col_num - num_images
    blank_image = np.zeros((img_height, img_width), dtype=np.uint8) if channels == 1 else np.zeros((img_height, img_width, channels), dtype=np.uint8)
    resized_images.extend([blank_image] * num_blanks)

    # Initialize an empty list to hold rows of concatenated images
    rows = []
    for i in range(row_num):
        # Concatenate images in each row
        row_images = resized_images[i * col_num:(i + 1) * col_num]
        row_concat = np.concatenate(row_images, axis=1)  # Concatenate along width (axis=1)
        rows.append(row_concat)

    # Concatenate all rows to form the final large image
    large_image = np.concatenate(rows, axis=0)  # Concatenate along height (axis=0)
    if len(image_list[0].shape) == 4 and not save_alpha:
        return large_image[:, :, :3]
    return large_image

def alpha_blend(render_image, ref_image):
    """
    Overlay render_image on top of ref_image using the alpha channels of both images.
    
    Args:
    - render_image (np.ndarray): RGBA image of shape (H, W, 4).
    - ref_image (np.ndarray): RGBA image of shape (H, W, 4).
    
    Returns:
    - blended_image (np.ndarray): Final overlayed image of shape (H, W, 4).
    """
    
    # Extract RGB and alpha channels for both images
    render_rgb = render_image[:, :, :3] * 255
    render_bgr = render_rgb[:, :, ::-1] # reverse RGB TO GBR for cv2 video writting.
    render_alpha = render_image[:, :, 3] # already to [0, 1]
    
    # ref_rgb = ref_image[:, :, [2, 1, 0]]
    ref_rgb = ref_image[:, :, :3]
    ref_bgr = ref_rgb[:, :, ::-1]
    ref_alpha = np.ones_like(ref_image[:, :, 3])  # Normalize alpha to [0, 1]
    
    # Compute the combined alpha of the two images
    combined_alpha = render_alpha + ref_alpha * (1 - render_alpha)
    
    # Blend the RGB channels based on the alpha channel
    blended_bgr = (render_bgr * render_alpha[:, :, None] + 
                   ref_bgr * ref_alpha[:, :, None] * (1 - render_alpha[:, :, None])) / combined_alpha[:, :, None]
    
    # Combine the blended RGB and the combined alpha channel
    blended_image = np.dstack((blended_bgr, combined_alpha * 255)).astype(np.uint8)
    
    return blended_image

# Function to apply transformation to the mesh vertices
def apply_transformation_to_mesh(mesh, rotation, translation):
    R = Rotate(rotation)
    T = Translate(translation)   # (1, 3)
    transform  = R.compose(T)
    tverts = transform.transform_points(mesh.verts_list()[0])
    faces = mesh.faces_list()[0]
    tmesh = Meshes(
        verts=[tverts],   
        faces=[faces],
        textures =mesh.textures,
    )

    return tmesh

def save_transformed_mesh(transformed_mesh, mesh, save_path):
    # Get the components of the transformed mesh
    transformed_verts = transformed_mesh.verts_packed()
    faces = transformed_mesh.faces_packed()
    verts_uvs = mesh.textures.verts_uvs_list()[0]
    faces_uvs = mesh.textures.faces_uvs_list()[0]
    texture_map = transformed_mesh.textures.maps_padded()[0]  # Single texture map

    # Save the transformed mesh as an OBJ file with texture information
    save_obj(
        save_path, 
        verts=transformed_verts, 
        faces=faces,
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        texture_map=texture_map
    )

def random_quaternions(n):
    rand = torch.randn(n, 4)  # Generate random numbers
    rand = rand / rand.norm(dim=1, keepdim=True)  # Normalize to unit quaternions
    return rand

def uniform_quaternions(n):
    """
    Generate n uniformly distributed quaternions.
    """
    u1 = torch.rand(n)  # Uniform random numbers in [0, 1]
    u2 = torch.rand(n)  # Uniform random numbers in [0, 1]
    u3 = torch.rand(n)  # Uniform random numbers in [0, 1]

    # Azimuthal angle: theta in [0, 2 * pi]
    theta = 2 * math.pi * u1

    # Polar angle: phi in [0, pi]
    phi = torch.acos(1 - 2 * u2)  # Uniform distribution on sphere

    # Radius for the unit quaternion
    r = torch.sqrt(1 - u3)

    # Components of the quaternion
    w = r * torch.cos(theta)
    x = r * torch.sin(theta)
    y = torch.sqrt(u3) * torch.cos(phi)
    z = torch.sqrt(u3) * torch.sin(phi)

    # Stack into a tensor of shape (n, 4) representing n quaternions
    quaternions = torch.stack([w, x, y, z], dim=1)
    return quaternions

def batch_render_loader(args, cameras, device):
    renderer_list = []
    for i, camera in enumerate(cameras[:]): # only render the first camera
        render_setup = setup_renderer(args, camera, device)
        renderer_list.append(render_setup)
    return renderer_list
    
def batch_render_ref_loader(args, idx, cameras, all_view_images, all_view_masks, device):
    image_ref_list = []
    renderer_list = []
    total_mask_pixels = 0
    for i, camera in enumerate(cameras[:]): # only render the first camera

        render_setup = setup_renderer(args, camera, device)

        """ load captured image """
        # img_path = os.path.join(args.image_dir, camera['cam_name'], str(idx).zfill(6) + '.png')

        # if os.path.isfile(img_path):
        img = all_view_images[camera['cam_name']][idx]
        masks = all_view_masks[camera['cam_name']][idx]
        if masks is not None:
            combined_mask = np.any(masks, axis=0)
            alpha_mask = (combined_mask * 255).astype(np.uint8)
            image_undist = np.dstack((img, alpha_mask))
            image_ref = cv2.resize(image_undist, (image_undist.shape[1]//args.down, image_undist.shape[0]//args.down))
            image_ref_list.append(image_ref)
            total_mask_pixels += (image_ref[:, :, -1] / 255).sum()

            renderer = render_setup['renderer']
            renderer_list.append(renderer)
    return image_ref_list, renderer_list, total_mask_pixels

def check_for_nan_params(model):
    has_nan_in_rotation = torch.isnan(model.mesh_rotation).any()
    has_nan_in_translation = torch.isnan(model.mesh_translation).any()

    if has_nan_in_rotation:
        print("NaN values detected in model.mesh_rotation")
    if has_nan_in_translation:
        print("NaN values detected in model.mesh_translation")

    return has_nan_in_rotation or has_nan_in_translation


# In[55]:
class DRModel(nn.Module):
    def __init__(self, meshes, renderer_list, image_ref_list, anchor_T=[[0.0, 0.0, 0.0]], init_R=torch.ones((1, 4), dtype=torch.float32), lambda_mask = 1.0, lambda_rgb = 0.0):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer_list = renderer_list
        self.lambda_mask = lambda_mask
        self.lambda_rgb = lambda_rgb

        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_mask_ref_stack = torch.stack([torch.from_numpy((image_ref[:, :, 3] > 0).astype(np.float32)) for image_ref in image_ref_list])
        # image_mask_ref_stack = torch.stack([image_ref for image_ref in image_ref_list])
        self.register_buffer('image_mask_ref_stack', image_mask_ref_stack)
        
        # Get the colored reference RGB image by finding all non-white pixel values. 
        if self.lambda_rgb > 0.0:
            image_rgb_ref_stack = torch.stack([
                torch.from_numpy(
                    np.where(
                        np.expand_dims(image_ref[:, :, 3] > 0, axis=-1),
                        (image_ref[:, :, [2, 1, 0]] / 255.).astype(np.float32),  # Use normalized RGB if alpha > 0
                        np.zeros_like(image_ref[:, :, :3], dtype=np.float32)  # Use zeros if alpha == 0
                    )
                )
                for image_ref in image_ref_list
            ])
            self.register_buffer('image_rgb_ref_stack', image_rgb_ref_stack)

        # Create an optimizable parameter for the translation and rotation of the mesh. 
        self.mesh_rotation = nn.Parameter(init_R.to(dtype=torch.float32, device=meshes.device))
        self.mesh_translation = nn.Parameter(anchor_T.to(dtype=torch.float32, device=meshes.device)) if isinstance(anchor_T, torch.Tensor) else nn.Parameter(torch.tensor(anchor_T, dtype=torch.float32).to(meshes.device))

    def forward(self, it):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        # R = Rotate(euler_angles_to_matrix(self.mesh_rotation, convention='XYZ'))
        # quaternions = torch.nn.functional.normalize(self.mesh_rotation, dim=-1)
        R = Rotate(quaternion_to_matrix(self.mesh_rotation), orthogonal_tol=1e-3)
        T = Translate(torch.clamp(self.mesh_translation, min=-0.2, max=1.2))   # (1, 3)
        transform  = R.compose(T)
        # print('transform nan', torch.isnan(transform).any())
        tverts = transform.transform_points(self.meshes.verts_list()[0])
        faces = self.meshes.faces_list()[0]
        tmesh = Meshes(
            verts=[tverts],   
            faces=[faces],
            textures = self.meshes.textures,
        )

        pixel_loss_list = []
        image_list = []
        
        for rid, render in enumerate(self.renderer_list[:]):
            image, fragments = render(meshes_world=tmesh)

            # Calculate the silhouette loss
            loss_mask = (image[..., 3] - self.image_mask_ref_stack[rid]) ** 2
            # loss_mask = (depth - self.image_mask_ref_stack[rid]) ** 2

            if self.lambda_rgb > 0:
                loss_rgb = (image[..., :3] - self.image_rgb_ref_stack[rid]) ** 2

            loss = torch.cat([loss_rgb, loss_mask.unsqueeze(-1)], dim=-1) if self.lambda_rgb > 0 else loss_mask

            image_list.append(image)
            pixel_loss_list.append(loss)

        return pixel_loss_list, image_list, tmesh