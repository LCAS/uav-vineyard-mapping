import json
import os
import math
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFilter
import cv2
import re

# --- 1. User Configuration ---
# Adjust these paths and values for your project.

# Input/Output Folders
image_folder = "../../images/riseholme/march_2025/39_feet"
geojson_file_path = "../../ground_truth/riseholme/riseholme_vine_locations.geojson"
output_folder = "./heatmap_masks/riseholme/march_2025/39_feet/vines/"
overlay_output_folder = "./annotated_images/riseholme/march_2025/39_feet/vines/"

# Flag to save annotated overlay images for verification
SAVE_OVERLAY_IMAGE = True

# Parameters for vine trunks
# 39 feet altitude image
VINE_RADIUS_BASE = 40   # Base radius in pixels for a vine trunk at the image center
BLUR_RADIUS_VINE = 30   # Gaussian blur radius for the heatmap effect

# # 65 feet altitude image
# VINE_RADIUS_BASE = 25   # Base radius in pixels for a vine trunk at the image center
# BLUR_RADIUS_VINE = 15   # Gaussian blur radius for the heatmap effect

# # 100 feet altitude image
# VINE_RADIUS_BASE = 20   # Base radius in pixels for a vine trunk at the image center
# BLUR_RADIUS_VINE = 10   # Gaussian blur radius for the heatmap effect

# Camera specifications (e.g., from DJI Zenmuse H20 Series)
camera_specs = {
    "focal_length_mm": 4.5,
    "sensor_width_mm": 6.17,
    "sensor_height_mm": 4.55
}

# --- 2. Helper Functions ---
def dms_to_decimal(dms_str):
    """Converts Degrees, Minutes, Seconds string to decimal degrees."""
    parts = dms_str.split()
    degrees = float(parts[0])
    minutes = float(parts[2].replace("'", ""))
    seconds = float(parts[3].replace('"', ""))
    direction = parts[4]
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_number(value):
    """Extracts a numeric value from a string."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        if match:
            return float(match.group(0))
    return None

def extract_exif(image_path):
    """Extracts key EXIF data from a drone image using ExifTool."""
    try:
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]
        return {
            "flight_yaw_degree": extract_number(metadata.get('FlightYawDegree')),
            "gimbal_yaw_degree": extract_number(metadata.get('GimbalYawDegree', 0)),
            "gps_latitude": dms_to_decimal(metadata['GPSLatitude']),
            "gps_longitude": dms_to_decimal(metadata['GPSLongitude']),
            "altitude_above_ground": extract_number(metadata.get('RelativeAltitude')),
            "image_width": metadata.get('ImageWidth'),
            "image_height": metadata.get('ImageHeight'),
        }
    except Exception as e:
        print(f"Error extracting EXIF from {os.path.basename(image_path)}: {e}")
        return None

def get_pixel_from_gps(latitude, longitude, flight_degree, gimbal_degree,
                       image_width, image_height,
                       gsd_x, gsd_y,
                       gps_lat_decimal, gps_lon_decimal):
    """Projects GPS coordinates to pixel coordinates using GSD."""
    lat_change = (latitude - gps_lat_decimal) * 111320
    lon_to_m = 111320 * math.cos(math.radians(gps_lat_decimal))
    lon_change = (longitude - gps_lon_decimal) * lon_to_m

    gimbal_radians = math.radians(float(gimbal_degree))

    corrected_lon_change = lon_change * math.cos(gimbal_radians) - lat_change * math.sin(gimbal_radians)
    corrected_lat_change = lon_change * math.sin(gimbal_radians) + lat_change * math.cos(gimbal_radians)

    corrected_pixel_x = corrected_lon_change / gsd_x
    corrected_pixel_y = corrected_lat_change / gsd_y

    pixel_x = (image_width / 2) + corrected_pixel_x
    pixel_y = (image_height / 2) - corrected_pixel_y

    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return int(pixel_x), int(pixel_y)
    return None, None

# --- 3. Main Label Generation Loop ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if SAVE_OVERLAY_IMAGE and not os.path.exists(overlay_output_folder):
    os.makedirs(overlay_output_folder)

with open(geojson_file_path, 'r') as f:
    geojson_data = json.load(f)

# Filter for Point features, which represent the vine trunks
vine_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'Point']
print(f"Loaded {len(vine_features)} vine trunk locations.")

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_folder, image_name)
    exif_data = extract_exif(image_path)
    
    required_keys = ['image_width', 'image_height', 'altitude_above_ground', 
                     'gps_latitude', 'gps_longitude', 'flight_yaw_degree']
    if not exif_data or any(exif_data.get(k) is None for k in required_keys):
        print(f"Skipping {image_name} due to missing essential metadata.")
        continue
    
    altitude = exif_data['altitude_above_ground']
    if altitude <= 0:
        print(f"Skipping {image_name} due to invalid altitude ({altitude}m).")
        continue

    print(f"Processing image: {image_name}")

    image_size = (exif_data['image_width'], exif_data['image_height'])
    
    # Calculate Ground Sample Distance (GSD) for this image
    gsd_x = (camera_specs["sensor_width_mm"] * altitude) / (camera_specs["focal_length_mm"] * image_size[0])
    gsd_y = (camera_specs["sensor_height_mm"] * altitude) / (camera_specs["focal_length_mm"] * image_size[1])

    # Create a single mask for the vine trunks
    vine_mask = Image.new('L', image_size, 0)
    draw_vine = ImageDraw.Draw(vine_mask)

    image_center_x, image_center_y = image_size[0] / 2, image_size[1] / 2
    max_distance = math.sqrt(image_center_x**2 + image_center_y**2)

    for vine in vine_features:
        vine_lon, vine_lat = vine['geometry']['coordinates']
        pixel_x, pixel_y = get_pixel_from_gps(
            vine_lat, vine_lon,
            exif_data['flight_yaw_degree'], exif_data['gimbal_yaw_degree'],
            image_size[0], image_size[1],
            gsd_x, gsd_y,
            exif_data['gps_latitude'], exif_data['gps_longitude']
        )
        
        if pixel_x is not None:
            # Scale radius based on distance from image center to simulate perspective
            distance_from_center = math.sqrt((pixel_x - image_center_x)**2 + (pixel_y - image_center_y)**2)
            scaling_factor = 1 + ((distance_from_center / max_distance) * 2)
            
            # Draw the individual vine trunk
            vine_radius = int(VINE_RADIUS_BASE * scaling_factor)
            draw_vine.ellipse((pixel_x - vine_radius, pixel_y - vine_radius, 
                               pixel_x + vine_radius, pixel_y + vine_radius), fill=255)

    # Apply blur to create heatmap effect and save the mask
    vine_mask_blurred = vine_mask.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS_VINE))
    vine_mask_blurred.save(os.path.join(output_folder, f"vine_mask_{image_name}"))

    # Create and save overlay image for verification
    if SAVE_OVERLAY_IMAGE:
        original_img = Image.open(image_path).convert("RGB")
        vine_mask_np = np.array(vine_mask_blurred)

        # Apply a color map (like JET or VIRIDIS) to the mask
        heatmap_cv = cv2.applyColorMap(vine_mask_np, cv2.COLORMAP_JET)
        heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB))
        
        # Blend the original image with the heatmap
        blended_img = Image.blend(original_img, heatmap_pil, alpha=0.5)
        blended_img.save(os.path.join(overlay_output_folder, f"overlay_{image_name}"))
        print(f" - Saved overlay image to {os.path.join(overlay_output_folder, f'overlay_{image_name}')}")
    
    print(f" - Generated vine mask for {image_name}")
    print("-" * 20)