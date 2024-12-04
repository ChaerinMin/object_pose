import open3d as o3d
import numpy as np

# Define paths for the mesh and point cloud
scanned_mesh_path = "/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/scans/ukelele_scan/ukelele-simplified.obj"
raw_init_path = "/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/ukelele/2024-06-18_0029/mesh/ngp_mesh/volume_raw/000000_filtered_bounded_clutered_samples.ply"

# Load the scanned mesh and raw initialization point cloud
scanned_mesh = o3d.io.read_triangle_mesh(scanned_mesh_path)
raw_init_pcd = o3d.io.read_point_cloud(raw_init_path)

# Convert the scanned mesh to a point cloud by sampling points
scanned_pcd = scanned_mesh.sample_points_poisson_disk(10000)  # Sample 10,000 points from the mesh

# Downsample point clouds for faster processing
scanned_pcd = scanned_pcd.voxel_down_sample(voxel_size=0.01)
raw_init_pcd = raw_init_pcd.voxel_down_sample(voxel_size=0.01)

# Estimate normals (important for ICP alignment)
scanned_pcd.estimate_normals()
raw_init_pcd.estimate_normals()

# Define parameters for ICP
threshold = 0.02  # Distance threshold for ICP convergence
trans_init = np.eye(4)  # Initial alignment guess (identity matrix)

# Apply ICP to align the scanned point cloud to the raw initialization point cloud
icp_result = o3d.pipelines.registration.registration_icp(
    source=scanned_pcd,           # Point cloud of the mesh to align
    target=raw_init_pcd,          # Reference point cloud
    max_correspondence_distance=threshold,
    init=trans_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Print the transformation matrix
print("Transformation Matrix from ICP:")
print(icp_result.transformation)

# Apply the transformation to the original scanned mesh (not the sampled points)
scanned_mesh.transform(icp_result.transformation)

# Save the aligned mesh
o3d.io.write_triangle_mesh("/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/ukelele_scan_transformed.obj", scanned_mesh)
