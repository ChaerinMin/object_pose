import pymeshlab

input_path = "/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/scans/instruments/ukelele_scan/ukelele.obj"
# Step 1: Create a new MeshLab script session
ms = pymeshlab.MeshSet()

# Step 2: Load your mesh (replace with the path to your .obj, .ply, etc.)
ms.load_new_mesh(input_path)

# # Step 3: Apply the simplification filter
# # You can specify the target number of faces or percentage of faces to retain
ms.meshing_decimation_quadric_edge_collapse(
    targetperc=0.5,                # Retain 20% of the original faces
    preserveboundary=True,          # Preserve boundary edges
    qualitythr=0.3,             # Threshold to retain better-quality faces
)

# Step 4: Center the mesh to make it zero-centered
ms.get_geometric_measures()  # Calculate geometric properties, including the center
center = ms.current_mesh().bounding_box().center()

# ms.compute_matrix_from_translation(axisx =-center[0], axisy =-center[1], axisz =-center[2])  # Translate to the origin

ms.compute_matrix_from_scaling_or_normalization(axisx = 1.2, axisy = 1.2, axisz = 1.2)
# Step 4: Save the simplified mesh
ms.save_current_mesh(input_path.replace('.obj', '-simplified1_2.obj'))