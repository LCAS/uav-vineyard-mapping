import os
import numpy as np
import rasterio
from rasterio.windows import Window, bounds
from rasterio.features import rasterize
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from PIL import Image
import geopandas as gpd
import cv2
from shapely.geometry import box

# --- Config ---
PATCH_SIZE = 800
PATCH_SCALE = 4.0
scaled_patch_size = int(PATCH_SIZE * PATCH_SCALE)

# geotiff_path = "../../images/orthophoto/jojo/agri_tech_centre/winter_2022/Vineyard_RGB_transparent_mosaic_group1.tif"
# geojson_path = "../../ground_truth/jojo/jojo_pole_locations.geojson"
# output_root = "dataset/orthophoto/jojo/winter_2022/"

geotiff_path = "../../images/orthophoto/riseholme/july_2025/100_feet/University-of-Lincoln-Riseholme-Campus-01-07-2025-100ft-orthophoto.tif"
geojson_path = "../../ground_truth/riseholme/riseholme_pole_locations_with_linestrings.geojson"
output_root = "dataset/orthophoto/riseholme/july_2025/100_feet/"

# --- Output folders ---
patch_dir = os.path.join(output_root, "patches")
row_mask_dir = os.path.join(output_root, "row_masks")
post_mask_dir = os.path.join(output_root, "post_masks")
annotated_dir = os.path.join(output_root, "annotated_images")
for d in [patch_dir, row_mask_dir, post_mask_dir, annotated_dir]:
    os.makedirs(d, exist_ok=True)

# --- Load geometries ---
gdf = gpd.read_file(geojson_path)
print(f"GeoJSON CRS before reprojection: {gdf.crs}")

with rasterio.open(geotiff_path) as src:
    print(f"Raster CRS: {src.crs}")
    gdf = gdf.to_crs(src.crs)
    print(f"GeoJSON CRS after reprojection: {gdf.crs}")

    rows = gdf[gdf.geometry.type == "LineString"]
    posts = gdf[gdf.geometry.type == "Point"]

    width, height = src.width, src.height
    total_x = (width + scaled_patch_size - 1) // scaled_patch_size
    total_y = (height + scaled_patch_size - 1) // scaled_patch_size
    total_patches = total_x * total_y
    patch_counter = 0

    for y in range(0, height, scaled_patch_size):
        for x in range(0, width, scaled_patch_size):
            patch_counter += 1
            print(f"Processing patch {patch_counter} of {total_patches} at x={x}, y={y}...")

            window = Window(x, y, scaled_patch_size, scaled_patch_size)
            transform = src.window_transform(window)

            # Read patch
            patch = src.read(window=window)
            if np.all(patch == src.nodata):
                print("  - Skipping empty patch")
                continue
            image_patch = np.moveaxis(patch, 0, -1)
            image_patch_resized = resize(image_patch, (PATCH_SIZE, PATCH_SIZE), anti_aliasing=True)
            image_patch_uint8 = (image_patch_resized * 255).astype(np.uint8)

            window_bounds = bounds(window, src.transform)
            window_polygon = box(*window_bounds)

            # Filter geometries intersecting this patch
            rows_in_patch = rows[rows.intersects(window_polygon)].geometry
            posts_in_patch = posts[posts.intersects(window_polygon)].geometry
            print(f"    rows_in_patch={len(rows_in_patch)}, posts_in_patch={len(posts_in_patch)}")

            h, w = int(window.height), int(window.width)

            # --- Rasterize ---
            row_mask = rasterize([(geom, 1) for geom in rows_in_patch], out_shape=(h, w), transform=transform, fill=0, dtype=float)
            post_mask = rasterize([(geom, 1) for geom in posts_in_patch], out_shape=(h, w), transform=transform, fill=0, dtype=float)

            # --- Apply Gaussian blur to create heatmaps ---
            row_mask = gaussian_filter(row_mask, sigma=30) # bigger sigma → thicker row lines
            post_mask = gaussian_filter(post_mask, sigma=50) # bigger sigma → larger dots

            # --- Normalize masks to 0-1 before resizing ---
            row_mask = row_mask / (row_mask.max() + 1e-8)
            post_mask = post_mask / (post_mask.max() + 1e-8)

            # post_mask *= 2.0  # brighten / amplify heatmap
            # post_mask = np.clip(post_mask, 0, 1)

            # --- Resize to PATCH_SIZE ---
            row_mask = resize(row_mask, (PATCH_SIZE, PATCH_SIZE), anti_aliasing=False)
            post_mask = resize(post_mask, (PATCH_SIZE, PATCH_SIZE), anti_aliasing=False)

            # --- Convert to uint8 for saving ---
            row_mask_uint8 = np.clip(row_mask * 255, 0, 255).astype(np.uint8)
            post_mask_uint8 = np.clip(post_mask * 255, 0, 255).astype(np.uint8)

            # --- Filenames ---
            ix, iy = x // scaled_patch_size, y // scaled_patch_size
            fname = f"{ix}_{iy}.png" if not (np.all(post_mask_uint8 == 0) and np.all(row_mask_uint8 == 0)) else f"{ix}_{iy}_d.png"

            # --- Save original patch and masks ---
            Image.fromarray(image_patch_uint8).save(os.path.join(patch_dir, fname))
            Image.fromarray(row_mask_uint8).save(os.path.join(row_mask_dir, fname))
            Image.fromarray(post_mask_uint8).save(os.path.join(post_mask_dir, fname))

            # --- Create annotated overlay using colormaps ---
            patch_rgb = image_patch_uint8[:, :, :3].astype(np.float32) / 255.0

            # Apply colormaps
            row_rgb = cv2.applyColorMap(row_mask_uint8, cv2.COLORMAP_JET)
            post_rgb = cv2.applyColorMap(post_mask_uint8, cv2.COLORMAP_COOL)

            row_rgb = cv2.cvtColor(row_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            post_rgb = cv2.cvtColor(post_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

            # Alpha blending using normalized masks
            row_alpha = np.expand_dims(row_mask * 0.5, axis=2)   # increase alpha for visibility
            post_alpha = np.expand_dims(post_mask * 0.9, axis=2)

            # Step 1: overlay rows
            overlay = patch_rgb * (1 - row_alpha) + row_rgb * row_alpha

            # Step 2: overlay posts on top of rows
            overlay = overlay * (1 - post_alpha) + post_rgb * post_alpha

            overlay = np.clip(overlay, 0, 1)

            overlay_img = Image.fromarray((overlay*255).astype(np.uint8))
            overlay_img.save(os.path.join(annotated_dir, fname))


            print(f"  - Saved patch, masks, and annotated image: {fname}")

print("✅ Done! All patches, masks, and annotated images saved.")
