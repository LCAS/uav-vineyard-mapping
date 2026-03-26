import os
import pandas as pd
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import pyproj
from pyproj import Transformer

def extract_vine_patches(geotiff_path, csv_path, output_dir, size_meters=1, handle_edges=True):
    """
    Extract patches from a GeoTIFF file centered on vine locations from a CSV file.
    Handles coordinate conversion from OSGB36 (BNG) to the GeoTIFF's CRS.
    
    Args:
        geotiff_path (str): Path to the GeoTIFF file
        csv_path (str): Path to the CSV file containing vine locations in BNG (Easting/Northing)
        output_dir (str): Directory to save the extracted patches
        size_meters (float): Size of the patch in meters
        handle_edges (bool): If True, handle vines near the edges by padding
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} vine locations from {csv_path}")
    
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Get the CRS of the GeoTIFF
        dst_crs = src.crs
        print(f"GeoTIFF CRS: {dst_crs}")
        print(f"GeoTIFF bounds: {src.bounds}")
        
        # Create coordinate transformer from BNG (EPSG:27700) to the GeoTIFF's CRS
        transformer = Transformer.from_crs(
            "EPSG:27700",  # British National Grid (OSGB36)
            dst_crs,       # GeoTIFF's CRS
            always_xy=True
        )
        
        # Get the resolution of the GeoTIFF
        res_x = src.transform[0]  # x resolution in CRS units
        res_y = -src.transform[4]  # y resolution (negative due to origin at top-left)
        
        print(f"GeoTIFF resolution: {res_x} x {res_y} meters per pixel")
        
        # Calculate size in pixels based on the resolution
        size_pixels_x = int(size_meters / res_x)
        size_pixels_y = int(size_meters / res_y)
        
        print(f"Extracting {size_pixels_x} x {size_pixels_y} pixel patches (approx. {size_meters}m x {size_meters}m)")
        
        # Track processed vines
        processed_count = 0
        edge_count = 0
        skipped_count = 0
        
        # Process each vine location
        for i, row in df.iterrows():
            vine_id = row['Vine ID']
            easting = row['Easting']
            northing = row['Northing']
            
            # Print the original coordinates for debugging
            print(f"Original BNG coordinates for {vine_id}: Easting={easting}, Northing={northing}")
            
            # Convert BNG coordinates to the GeoTIFF's CRS
            x, y = transformer.transform(easting, northing)
            print(f"Transformed coordinates: x={x}, y={y}")
            
            # Convert coordinates to pixel coordinates in the GeoTIFF
            row_offset, col_offset = ~src.transform * (x, y)
            
            # Convert to integers for pixel indices
            px, py = int(col_offset), int(row_offset)
            print(f"Pixel coordinates: px={px}, py={py}")
            
            # Check if the pixel is within the bounds of the GeoTIFF
            if px < 0 or px >= src.width or py < 0 or py >= src.height:
                print(f"Warning: Vine {vine_id} at ({easting}, {northing}) -> ({x}, {y}) -> ({px}, {py}) is outside the GeoTIFF bounds {(0, 0, src.width, src.height)}. Skipping.")
                skipped_count += 1
                continue
            
            # Calculate window boundaries (centered on the vine)
            row_start = int(py - size_pixels_y // 2)
            col_start = int(px - size_pixels_x // 2)
            
            # Check if window is outside the GeoTIFF boundaries
            if not handle_edges and (row_start < 0 or col_start < 0 or 
                row_start + size_pixels_y > src.height or 
                col_start + size_pixels_x > src.width):
                print(f"Warning: Vine {vine_id} at ({easting}, {northing}) is too close to the edge of the GeoTIFF. Skipping.")
                skipped_count += 1
                continue
            
            # Handle edge cases when requested
            if handle_edges:
                # Create a blank canvas for our output image
                output_data = np.zeros((3, size_pixels_y, size_pixels_x), dtype=np.uint8)
                
                # Calculate the valid region to read from the source
                valid_row_start = max(0, row_start)
                valid_col_start = max(0, col_start)
                valid_row_end = min(src.height, row_start + size_pixels_y)
                valid_col_end = min(src.width, col_start + size_pixels_x)
                
                # Skip if the vine is completely outside the GeoTIFF
                if valid_row_end <= valid_row_start or valid_col_end <= valid_col_start:
                    print(f"Warning: Vine {vine_id} at ({easting}, {northing}) is completely outside the GeoTIFF. Skipping.")
                    skipped_count += 1
                    continue
                
                # Calculate offsets for placing the data in the output array
                out_row_start = valid_row_start - row_start
                out_col_start = valid_col_start - col_start
                out_row_end = out_row_start + (valid_row_end - valid_row_start)
                out_col_end = out_col_start + (valid_col_end - valid_col_start)
                
                # Read the valid portion of data
                window = Window(
                    valid_col_start, 
                    valid_row_start, 
                    valid_col_end - valid_col_start, 
                    valid_row_end - valid_row_start
                )
                
                data = src.read(window=window)
                
                # Handle different band counts
                if data.shape[0] == 1:
                    # For single band, duplicate to all three RGB channels
                    for i in range(3):
                        output_data[i, out_row_start:out_row_end, out_col_start:out_col_end] = data[0]
                elif data.shape[0] >= 3:
                    # For 3+ bands, use the first three
                    output_data[:3, out_row_start:out_row_end, out_col_start:out_col_end] = data[:3]
                
                # Note that we're handling an edge case
                edge_count += 1
                
                # Normalize data
                for i in range(3):
                    band = output_data[i]
                    if np.any(band):  # Only normalize if there's actual data
                        min_val = np.percentile(band[band > 0], 2)
                        max_val = np.percentile(band[band > 0], 98)
                        if max_val > min_val:  # Prevent division by zero
                            output_data[i] = np.clip((band - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
                
                # Convert to RGB
                rgb_data = np.transpose(output_data, (1, 2, 0))
                
            else:
                # Original handling for non-edge cases
                window = Window(col_start, row_start, size_pixels_x, size_pixels_y)
                data = src.read(window=window)
                
                # Convert to RGB
                if data.shape[0] == 1:
                    rgb_data = np.stack([data[0], data[0], data[0]], axis=2)
                elif data.shape[0] == 3:
                    rgb_data = np.transpose(data, (1, 2, 0))
                elif data.shape[0] > 3:
                    rgb_data = np.transpose(data[0:3], (1, 2, 0))
                
                # Normalize data
                if rgb_data.dtype != np.uint8:
                    for i in range(rgb_data.shape[2]):
                        band = rgb_data[:, :, i]
                        min_val = np.percentile(band, 2)
                        max_val = np.percentile(band, 98)
                        rgb_data[:, :, i] = np.clip((band - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
            
            # Create and save the image
            img = Image.fromarray(rgb_data)
            output_path = os.path.join(output_dir, f"{vine_id}.png")
            img.save(output_path)
            processed_count += 1
            
            if processed_count % 10 == 0 or i == len(df) - 1:
                print(f"Processed {processed_count}/{len(df)} vines ({edge_count} edge cases handled, {skipped_count} skipped)")

# Alternative function that you can use if the vineyard lat/long values are available
def extract_vine_patches_from_latlong(geotiff_path, csv_path, output_dir, size_meters=1, handle_edges=True):
    """
    Extract patches from a GeoTIFF file centered on vine locations from a CSV file.
    Handles coordinate conversion from WGS84 (lat/long) to the GeoTIFF's CRS.
    
    Args:
        geotiff_path (str): Path to the GeoTIFF file
        csv_path (str): Path to the CSV file containing vine locations in lat/long
        output_dir (str): Directory to save the extracted patches
        size_meters (float): Size of the patch in meters
        handle_edges (bool): If True, handle vines near the edges by padding
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} vine locations from {csv_path}")
    
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Get the CRS of the GeoTIFF
        dst_crs = src.crs
        print(f"GeoTIFF CRS: {dst_crs}")
        print(f"GeoTIFF bounds: {src.bounds}")
        
        # Create coordinate transformer from WGS84 to the GeoTIFF's CRS
        transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (latitude/longitude)
            dst_crs,      # GeoTIFF's CRS
            always_xy=True  # Ensure lon/lat order for input
        )
        
        # Continue with the same logic as above
        # ...
        # (The rest of the function remains the same as extract_vine_patches,
        # just make sure to use lat/long columns instead of easting/northing)

if __name__ == "__main__":
    geotiff_path = "../images/orthophoto/39_feet/odm_orthophoto.tif"
    # geotiff_path = "../images/orthophoto/65_feet/odm_orthophoto.tif"
    # geotiff_path = "../images/orthophoto/100_feet/odm_orthophoto.tif"
    csv_path = "../data/riseholme_vineyard_easting_northing_vines.csv"  # BNG OSGB36, EPSG:27700 coordinates
    # Alternatively use lat/long if available
    # csv_path = "../data/riseholme_vineyard_lat_long_vines.csv"
    output_dir = "../data/vine_images_from_orthophoto"
    patch_size_meters = 2.0 
    
    # Use with BNG coordinates
    extract_vine_patches(geotiff_path, csv_path, output_dir, patch_size_meters)
    
    # Or use with lat/long coordinates
    # extract_vine_patches_from_latlong(geotiff_path, csv_path, output_dir, patch_size_meters)