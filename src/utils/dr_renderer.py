import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader
)
from scipy.spatial.transform import Rotation
import trimesh
from PIL import Image

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_params(params_path):
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ]
    )
    params = np.sort(params, order="cam_name")

    return params

class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref, init_R, init_T):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_ref = torch.from_numpy((image_ref[:, :, 3] > 0).astype(np.float32))

        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_rotation = nn.Parameter(init_R).to(meshes.device)
        self.camera_translation = nn.Parameter(init_T).to(meshes.device)

    def forward(self, it):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        R = quaternion_to_matrix(self.camera_rotation)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        T = self.camera_translation   # (1, 3)
        
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        loss = (image[..., 3] - self.image_ref) ** 2
        return loss, image

if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    path = "/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-09-21_session_snapshot-object-instruments"
    params_path = os.path.join(path, "optim_params.txt")
    params = read_params(params_path)

    param = params[0]
    # Camera parameters from COLMAP
    height = int(param["height"])
    width = int(param["width"])
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])

    R = qvec2rotmat(-qvec)
    T = np.array(tvec)

    # Rt = np.zeros((4, 4))
    # Rt[:3, :3] = R.transpose()
    # Rt[:3, 3] = T
    # Rt[3, 3] = 1.0

    # W2C = np.linalg.inv(Rt)
    # T = W2C[:3, 3]
    # R = W2C[:3, :3]
    
    # Given R and T (world2camera)
    R = torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0)
    T = torch.tensor(T, dtype=torch.float32, device=device).unsqueeze(0)

    fx = param["fx"] / width
    fy = param["fy"] / height
    cx = param["cx"] / width
    cy = param["cy"] / height
    k1, k2, p1, p2 = -0.4007544787858172, 0.14385189120881453, 0.002595185177401478, 0.0017625355487786755

    fov_y = 2 * np.arctan(height / (2 * fy)) * (180.0 / np.pi)

    # # Step 3: Intrinsics - Normalize focal length and principal point
    # focal_length = torch.tensor([[focal_length_x / width, focal_length_y / height]], dtype=torch.float32)  # Normalized focal length
    # principal_point = torch.tensor([[cx / width, cy / height]], dtype=torch.float32)  # Normalized principal point
    
    # K = np.eye(4)
    # K[0, 0] = param["fx"] #/ width
    # K[1, 1] = param["fy"] #/ height
    # K[0, 2] = param["cx"] / width
    # K[1, 2] = param["cy"] / height

    # K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
    # # rot = torch.tensor(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]), dtype=torch.float32, device=device)
    # height, width = 720, 1280
    
    # Load template mesh
    # mesh = load_objs_as_meshes(["/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-09-21_session_snapshot-object-instruments/scans/ukelele_scan/AR-Code-Object-Capture-app-1727202177.obj"], device=device)
    verts, faces = load_ply("/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-09-21_session_snapshot-object-instruments/mesh/ngp_mesh/ukelele.ply")
    textures = TexturesVertex(verts_features=torch.zeros_like(verts)[None])
    mesh = Meshes([verts], [faces], textures=textures).to("cuda")

    
    # print(vertex_colors.shape)
    # exit(0)
    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(
        znear=0.1, 
        device=device)
    # cameras = PerspectiveCameras(
    #     focal_length=((fx, fy),),
    #     principal_point=((cx, cy),),
    #     image_size=[(height, width)],
    #     device=device
    # )

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-8, gamma=1e-8)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=np.log(1. / 1e-8 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # Select the viewpoint using spherical angles  
    distance = 1   # distance from camera to the object
    elevation = 50.0   # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

    # Get the position of the camera based on the spherical angles
    _, _ = look_at_view_transform(distance, elevation, azimuth, device=device)
    print(R.shape)
    print(T.shape)
    # Render the teapot providing the values of R and T. 
    silhouette = silhouette_renderer(meshes_world=mesh, R=R, T=T)
    # image_ref = phong_renderer(meshes_world=mesh, R=R, T=T)

    silhouette = silhouette.cpu().numpy()
    # image_ref = image_ref.cpu().numpy()

    image_path = "/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-09-21_session_snapshot-object-instruments/images/segmented_sam/brics-odroid-001_cam0/00000000.png"
    image = Image.open(image_path)
    image_ref = np.array(image)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze())
    plt.grid(False)

    plt.savefig("output_figure.png")
    # exit(0)
    # We will save images periodically and compose them into a GIF.
    filename_output = "./teapot_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
    filename_output = "./teapot_optimization_loss.gif"
    loss_writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    # Initialize a model using the renderer, mesh and reference image
    R = matrix_to_quaternion(R)
    T = T
    model = Model(meshes=mesh, renderer=silhouette_renderer, image_ref=image_ref, init_R=R, init_T=T).to(device)
    print(model)
    exit(0)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loop = tqdm(range(100))
    for i in loop:
        optimizer.zero_grad()
        loss_pixels, _ = model(i)

        loss = torch.sum(loss_pixels)
        loss.backward()
        optimizer.step()
        
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        
        if loss.item() < 10:
            break
        
        # Save outputs to create a GIF. 
        if i % 10 == 0:
            R = quaternion_to_matrix(model.camera_rotation)
            T = model.camera_translation
            image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
            loss_writer.append_data(img_as_ubyte(loss_pixels.detach().cpu().numpy()))
            
            plt.figure()
            plt.imshow(image[..., :3])
            plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            plt.axis("off")
        
    writer.close()
    loss_writer.close()
    exit(0)