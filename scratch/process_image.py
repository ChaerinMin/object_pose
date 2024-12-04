from PIL import Image

view_name = 'brics-odroid-006_cam0' # 'brics-odroid-025_cam1' #'brics-odroid-012_cam0' #'brics-odroid-006_cam0' 
# Define the input file path
input_path = f'/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/ukelele/2024-06-18_0030/segmentation/{view_name}/000120.png'

# Define the output file path
output_path = f'/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/ukelele/2024-06-18_0030/segmentation/{view_name}/000120_rgb_mask.png'

# # Open the RGBA image
# image = Image.open(input_path)

# # Convert the image to RGB (ignoring alpha channel)
# rgb_image = image.convert('RGB')

# # Save the RGB image
# rgb_image.save(output_path)

# print(f"RGB image saved at: {output_path}")

# # Open the RGBA image
# image = Image.open(input_path)

# # Split the RGBA channels
# r, g, b, alpha = image.split()

# # Create an RGB version of the image and make it fully opaque
# rgb_image = Image.merge("RGBA", (r, g, b, Image.new("L", alpha.size, 255)))

# # Create the mask image
# # Masked area (alpha > 0) is opaque (255), other areas are transparent (0)
# mask_transparency = alpha.point(lambda p: 128 if p > 0 else 0)
# red_mask = Image.merge("RGBA", (alpha, Image.new("L", alpha.size, 0), Image.new("L", alpha.size, 0), mask_transparency))

# # Overlay the RGB image with the mask
# final_image = Image.alpha_composite(rgb_image, red_mask)

# # Save the resulting image
# final_image.save(output_path)

# print(f"Final overlay image saved at: {output_path}")


from PIL import Image

# Define file paths
rgb_path = '/users/rfu7/ssrinath/datasets/Action/brics-mini-objects/ukelele/2024-06-18_0030/segmentation/brics-odroid-006_cam0/000120_rgb.png'
rgba_path = '/users/rfu7/data/code/24Text2Action/data_analysis/video_visualization/render_video/v1_hands.png'
output_path = '/users/rfu7/data/code/24Text2Action/data_analysis/video_visualization/render_video/v1_composed_hands.png'

# Open the RGB and RGBA images
rgb_image = Image.open(rgb_path).convert("RGBA")  # Ensure it has an alpha channel
rgba_image = Image.open(rgba_path)

# Extract the alpha channel from the RGBA image
r_overlay, g_overlay, b_overlay, alpha_overlay = rgba_image.split()

# Create the overlay image from RGBA (using its RGB with transparency)
overlay_image = Image.merge("RGBA", (r_overlay, g_overlay, b_overlay, alpha_overlay))

# Ensure the base RGB image is fully opaque
r, g, b, _ = rgb_image.split()
rgb_fully_opaque = Image.merge("RGBA", (r, g, b, Image.new("L", rgb_image.size, 255)))

# Overlay the RGB image with the second image using transparency
final_image = Image.alpha_composite(rgb_fully_opaque, overlay_image)

# Save the resulting image
final_image.save(output_path)

print(f"Composed image saved at: {output_path}")
