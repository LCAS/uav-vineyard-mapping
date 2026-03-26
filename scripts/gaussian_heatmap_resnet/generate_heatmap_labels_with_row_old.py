import json
import os
import math
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFilter
import cv2

# --- User-Provided Information ---
image_folder = "../../images/agri_tech_centre/jojo/"
geojson_file_path = "../../ground_truth/jojo/jojo_pole_locations.geojson"
output_folder = "./heatmap_masks/agri_tech_centre/jojo/"

# --- Flag to save mask images overlaid on input images ---
SAVE_OVERLAY_IMAGE = True
overlay_output_folder = "./annotated_images/agri_tech_centre/jojo/"

# 100 feet altitude image
# --- Base values for mask sizes at the image center ---
POST_RADIUS_BASE = 30 # Radius for post markers at the center
ROW_WIDTH_BASE = 60 # Width of the rows at the center
BLUR_RADIUS_POST = 15 # Base blur radius for post heatmaps
BLUR_RADIUS_ROW = 30 # Base blur radius for row heatmaps

# # Camera specifications from DJI Zenmuse H20 Series for Riseholme images
# camera_specs = {
#     "focal_length_mm": 4.5,
#     "sensor_width_mm": 6.17,
#     "sensor_height_mm": 4.55
# }

# Camera specifications Agri tech centre drone P1 camera # https://enterprise.dji.com/zenmuse-p1
camera_specs = {
    "focal_length_mm": 35.0,  # * 0.12
    "fov_deg": 63.5,
    "sensor_width_mm": 35.9,
    "sensor_height_mm": 24.0
}

# --- Helper Functions ---
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
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        if match:
            return float(match.group(0))
    return None

def extract_exif(image_path):
    """Extracts key EXIF data from a drone image using ExifTool."""
    try:
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        metadata = json.loads(result.stdout)
        if metadata:
            metadata_dict = metadata[0]

            flight_yaw_degree = metadata_dict.get('FlightYawDegree', None)
            flight_pitch_degree = metadata_dict.get('FlightPitchDegree', None)
            flight_roll_degree = metadata_dict.get('FlightRollDegree', None)
            gimbal_yaw_degree = metadata_dict.get('GimbalYawDegree', 0)
            gimbal_pitch_degree = metadata_dict.get('GimbalPitchDegree', 0)
            gimbal_roll_degree = metadata_dict.get('GimbalRollDegree', 0)

            gps_latitude_dms = metadata_dict.get('GPSLatitude', None)
            gps_longitude_dms = metadata_dict.get('GPSLongitude', None)

            gps_latitude = dms_to_decimal(gps_latitude_dms) if gps_latitude_dms else None
            gps_longitude = dms_to_decimal(gps_longitude_dms) if gps_longitude_dms else None

            altitude_above_ground = metadata_dict.get('RelativeAltitude', None)

            if flight_yaw_degree is None:
                flight_yaw_degree = metadata_dict.get('Yaw', None)
            if flight_pitch_degree is None:
                flight_pitch_degree = metadata_dict.get('Pitch', None)
            if flight_roll_degree is None:
                flight_roll_degree = metadata_dict.get('Roll', None)

            image_height = metadata_dict.get('ImageHeight', None)
            image_width = metadata_dict.get('ImageWidth', None)

            print(f"Extracted EXIF from {os.path.basename(image_path)}: "
                    f"Yaw: {flight_yaw_degree}, Pitch: {flight_pitch_degree}, Roll: {flight_roll_degree}, "
                    f"Gimbal Yaw: {gimbal_yaw_degree}, Gimbal Pitch: {gimbal_pitch_degree}, Gimbal Roll: {gimbal_roll_degree}, "
                    f"Lat: {gps_latitude}, Lon: {gps_longitude}, Altitude: {altitude_above_ground}, "
                    f"Width: {image_width}, Height: {image_height}")

            return {
                "flight_yaw_degree": extract_number(flight_yaw_degree),
                "flight_pitch_degree": extract_number(flight_pitch_degree),
                "flight_roll_degree": extract_number(flight_roll_degree),
                "gimbal_yaw_degree": extract_number(gimbal_yaw_degree),
                "gimbal_pitch_degree": extract_number(gimbal_pitch_degree),
                "gimbal_roll_degree": extract_number(gimbal_roll_degree),
                "gps_latitude": gps_latitude,
                "gps_longitude": gps_longitude,
                "altitude_above_ground": extract_number(altitude_above_ground),
                "image_width": image_width,
                "image_height": image_height,
            }
        else:
            return None
    except Exception as e:
        print(f"Error extracting EXIF: {e}")
        return None

def get_pixel_from_gps(latitude, longitude, flight_degree, gimbal_degree,
                       image_width, image_height,
                       gsd_x, gsd_y,
                       gps_lat_decimal, gps_lon_decimal):

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

# --- Main Label Generation Loop ---
if not os.path.exists(image_folder):
    print(f"Error: Image folder not found at '{image_folder}'")
    exit()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if SAVE_OVERLAY_IMAGE and not os.path.exists(overlay_output_folder):
    os.makedirs(overlay_output_folder)

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"Error: No image files found in '{image_folder}'")
    exit()

try:
    with open(geojson_file_path, 'r') as f:
        geojson_data = json.load(f)
except FileNotFoundError:
    print(f"Error: GeoJSON file not found at '{geojson_file_path}'")
    exit()

# Separate features based on geometry type
post_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'Point']
row_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'LineString']


for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)

    exif_data = extract_exif(image_path)
    if not exif_data or exif_data['image_width'] is None or exif_data['image_height'] is None:
        print(f"Skipping {image_name} due to missing metadata.")
        continue

    image_size = (exif_data['image_width'], exif_data['image_height'])

    altitude_above_ground = exif_data['altitude_above_ground']

    if altitude_above_ground is None or altitude_above_ground <= 0:
        print(f"Skipping {image_name} due to invalid altitude data.")
        continue

    gsd_x = (camera_specs["sensor_width_mm"] * altitude_above_ground) / (camera_specs["focal_length_mm"] * exif_data["image_width"])
    gsd_y = (camera_specs["sensor_height_mm"] * altitude_above_ground) / (camera_specs["focal_length_mm"] * exif_data["image_height"])

    print(f"Processing image: {image_name}")

    post_mask = Image.new('L', image_size, 0)
    row_mask = Image.new('L', image_size, 0)
    draw_post = ImageDraw.Draw(post_mask)
    draw_row = ImageDraw.Draw(row_mask)

    image_center_x, image_center_y = image_size[0] / 2, image_size[1] / 2
    max_distance = math.sqrt(image_center_x**2 + image_center_y**2)
    
    # Process Point features (posts)
    for post in post_features:
        post_lon, post_lat = post['geometry']['coordinates']

        pixel_x, pixel_y = get_pixel_from_gps(
            post_lat, post_lon,
            exif_data['flight_yaw_degree'], exif_data['gimbal_yaw_degree'],
            image_size[0], image_size[1],
            gsd_x, gsd_y,
            exif_data['gps_latitude'], exif_data['gps_longitude']
        )

        if pixel_x is not None:
            distance_from_center = math.sqrt((pixel_x - image_center_x)**2 + (pixel_y - image_center_y)**2)
            scaling_factor = 1 + ((distance_from_center / max_distance) * 2)

            post_radius = int(POST_RADIUS_BASE * scaling_factor)
            draw_post.ellipse((pixel_x - post_radius, pixel_y - post_radius, pixel_x + post_radius, pixel_y + post_radius), fill=255)
    
    # Process LineString features (rows)
    for row in row_features:
        row_coordinates = row['geometry']['coordinates']
        pixel_coords = []
        for lon, lat in row_coordinates:
            pixel_x, pixel_y = get_pixel_from_gps(
                lat, lon,
                exif_data['flight_yaw_degree'], exif_data['gimbal_yaw_degree'],
                image_size[0], image_size[1],
                gsd_x, gsd_y,
                exif_data['gps_latitude'], exif_data['gps_longitude']
            )
            if pixel_x is not None:
                pixel_coords.append((pixel_x, pixel_y))

        if len(pixel_coords) > 1:
            distance_from_center = math.sqrt((pixel_coords[0][0] - image_center_x)**2 + (pixel_coords[0][1] - image_center_y)**2)
            scaling_factor = 1 + ((distance_from_center / max_distance) * 2)
            row_width = int(ROW_WIDTH_BASE * scaling_factor)
            draw_row.line(pixel_coords, fill=255, width=row_width)


    post_mask_blurred = post_mask.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS_POST))
    row_mask_blurred = row_mask.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS_ROW))

    post_mask_blurred.save(os.path.join(output_folder, f"posts_mask_{image_name}"))
    row_mask_blurred.save(os.path.join(output_folder, f"rows_mask_{image_name}"))

    if SAVE_OVERLAY_IMAGE:
        original_img = Image.open(image_path).convert("RGB")
        
        # Convert blurred masks to numpy arrays for color mapping
        post_mask_np = np.array(post_mask_blurred)
        row_mask_np = np.array(row_mask_blurred)

        # Apply jet colormap
        post_heatmap = cv2.applyColorMap(post_mask_np, cv2.COLORMAP_JET)
        row_heatmap = cv2.applyColorMap(row_mask_np, cv2.COLORMAP_JET)
        
        # Convert back to Pillow images
        post_heatmap_pil = Image.fromarray(cv2.cvtColor(post_heatmap, cv2.COLOR_BGR2RGB))
        row_heatmap_pil = Image.fromarray(cv2.cvtColor(row_heatmap, cv2.COLOR_BGR2RGB))
        
        # Create an empty overlay image and paste the heatmaps with the mask as alpha
        overlay_img = Image.new('RGB', image_size, (0, 0, 0))
        overlay_img.paste(row_heatmap_pil, (0, 0), row_mask_blurred)
        overlay_img.paste(post_heatmap_pil, (0, 0), post_mask_blurred)
        
        blended_img = Image.blend(original_img, overlay_img, alpha=0.5)
        blended_img.save(os.path.join(overlay_output_folder, f"overlay_{image_name}"))
        print(f" - Saved overlay image to {os.path.join(overlay_output_folder, f'overlay_{image_name}')}")
    
    print(f" - Generated post and row masks for {image_name}")
    print("-" * 20)