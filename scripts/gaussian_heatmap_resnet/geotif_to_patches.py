import rasterio
from rasterio.windows import Window
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform # Required for resizing

# --- Configuration ---
# Update these paths to match your file locations
image_folder = "../../images/orthophoto/jojo/agri_tech_centre/may_2025/"
geotiff_file_path = os.path.join(image_folder, "JOJO's_RGB_transparent_mosaic_group1_modified.tif")
output_folder = "../../images/orthophoto/jojo/agri_tech_centre/may_2025/vine_row_patches_800x800_scaled_grid"

# Define the final output image size in pixels
PATCH_SIZE = 800

# Define the scaling factor for the initial crop
# A scale of 2.0 means we will tile the image in 1600x1600 sections
# and then resize each one down to the final 800x800 size.
PATCH_SCALE = 2.0

# --- Main Script ---

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Calculate the size of the grid cells to read from the source image
scaled_patch_size = int(PATCH_SIZE * PATCH_SCALE)

try:
    # 1. Open the GeoTIFF file
    with rasterio.open(geotiff_file_path) as src:
        width = src.width
        height = src.height
        nodata_value = src.nodata if src.nodata is not None else 0

        print(f"Tiling image of size {width}x{height} into {scaled_patch_size}x{scaled_patch_size} sections.")
        print(f"Each section will be resized to {PATCH_SIZE}x{PATCH_SIZE}.")

        # 2. Iterate over the image in a grid of scaled_patch_size
        for y in range(0, height, scaled_patch_size):
            for x in range(0, width, scaled_patch_size):
                
                # 3. Define the window for the large (scaled) patch
                window = Window(x, y, scaled_patch_size, scaled_patch_size)

                # 4. Read the large patch data
                large_patch = src.read(window=window)
                
                # 5. Skip empty patches
                if np.all(large_patch == nodata_value):
                    continue

                # 6. Resize the patch
                # Reshape for processing: (bands, height, width) -> (height, width, bands)
                image_to_resize = np.moveaxis(large_patch, 0, -1)

                # Resize the large patch down to the target patch size
                resized_image = transform.resize(
                    image_to_resize,
                    (PATCH_SIZE, PATCH_SIZE),
                    anti_aliasing=True
                )
                
                # The output of resize is a float array (0.0 to 1.0), ready for saving.
                if resized_image.shape[2] == 1:
                    resized_image = resized_image.squeeze(axis=2)
                
                # 7. Generate filename based on grid coordinates
                x_index = x // scaled_patch_size
                y_index = y // scaled_patch_size
                output_filename = os.path.join(output_folder, f"{x_index}_{y_index}.png")

                # 8. Save the final, resized patch
                plt.imsave(output_filename, resized_image, cmap='gray' if len(resized_image.shape) == 2 else None)
                print(f"  - Saved resized patch to {output_filename}")


except FileNotFoundError:
    print(f"Error: The file {geotiff_file_path} was not found. Please check the path and filename.")
except Exception as e:
    print(f"An error occurred: {e}")

print("\nProcessing complete.")