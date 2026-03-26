import json
import os
import math
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFilter
import cv2

# --- User-Provided Information ---
# image_folder = "../../images/agri_tech_centre/jojo/"
# geojson_file_path = "../../ground_truth/jojo/jojo_pole_locations.geojson"
# output_folder = "./heatmap_masks/agri_tech_centre/jojo/"

# image_folder = "../../images/agri_tech_centre/coolhurst/"
# geojson_file_path = "../../ground_truth/coolhurst/coolhurst_pole_locations.geojson"
# output_folder = "./heatmap_masks/agri_tech_centre/coolhurst/"

# image_folder = "../../images/agri_tech_centre/arun_1/"
# geojson_file_path = "../../ground_truth/arun_valley/arun_valley_pole_locations.geojson"
# output_folder = "./heatmap_masks/agri_tech_centre/arun_1/"

# image_folder = "../../images/agri_tech_centre/arun_2/"
# geojson_file_path = "../../ground_truth/arun_valley/arun_valley_pole_locations.geojson"
# output_folder = "./heatmap_masks/agri_tech_centre/arun_2/"

# image_folder = "../../images/riseholme/august_2024/39_feet"
# geojson_file_path = "../../ground_truth/riseholme/riseholme_pole_locations.geojson"
# output_folder = "./heatmap_masks/riseholme/testing"

image_folder = "../../images/jojo/riccardo/DJI_202507301654_030_jojo"
geojson_file_path = "../../ground_truth/jojo/jojo_pole_locations.geojson"
output_folder = "./heatmap_masks/jojo/riccardo/"

# --- Flag to save mask images overlaid on input images ---

SAVE_OVERLAY_IMAGE = True
overlay_output_folder = "./annotated_images/jojo/riccardo/"

# overlay_output_folder = "./annotated_images/riseholme/testing/"

# --- Base values for mask sizes at the image center ---
POST_RADIUS_BASE = 30
ROW_WIDTH_BASE = 60
BLUR_RADIUS_POST = 15
BLUR_RADIUS_ROW = 30

# --- Helper Functions ---
def build_camera_matrix(img_width, img_height, hfov_deg):
    """
    Build a camera intrinsic matrix using horizontal FOV.
    """
    hfov = math.radians(hfov_deg)
    f_x = (img_width / 2) / math.tan(hfov / 2)
    f_y = f_x  # Assume square pixels
    c_x = img_width / 2
    c_y = img_height / 2
    K = np.array([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1  ]
    ])
    return K

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

def rotation_matrix(yaw, pitch, roll):
    """
    Compute a 3D rotation matrix from yaw, pitch, and roll angles (degrees).
    """
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))

    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [ 0,   0, 1]])
    Ry = np.array([[cp, 0, sp],  [ 0,  1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],   [0, cr,-sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def extract_exif(image_path):
    """Extracts key EXIF data from a drone image using ExifTool."""
    try:
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]
        return {
            "flight_yaw_degree": extract_number(metadata.get('FlightYawDegree')),
            "flight_pitch_degree": extract_number(metadata.get('FlightPitchDegree')),
            "flight_roll_degree": extract_number(metadata.get('FlightRollDegree')),
            "gimbal_yaw_degree": extract_number(metadata.get('GimbalYawDegree')),
            "gimbal_pitch_degree": extract_number(metadata.get('GimbalPitchDegree')),
            "gimbal_roll_degree": extract_number(metadata.get('GimbalRollDegree')),
            "gps_latitude": dms_to_decimal(metadata['GPSLatitude']),
            "gps_longitude": dms_to_decimal(metadata['GPSLongitude']),
            "altitude_above_ground": extract_number(metadata.get('RelativeAltitude')),
            "image_width": metadata.get('ImageWidth'),
            "image_height": metadata.get('ImageHeight'),
            "fov_deg": extract_number(metadata.get('FOV'))
        }
    except Exception as e:
        print(f"Error extracting EXIF from {os.path.basename(image_path)}: {e}")
        return None

def get_pixel_from_gps(lat, lon,
                       flight_yaw, flight_pitch, flight_roll,
                       gimbal_yaw, gimbal_pitch, gimbal_roll,
                       gps_lat, gps_lon, altitude,
                       K, image_width, image_height,
                       convention="dji", debug=False):
    """
    Project a ground point into image coordinates using different
    ENU→camera optical frame conventions.
    """

    # Convert GPS differences to meters in ENU (East, North, Up)
    lat_change = (lat - gps_lat) * 111320.0
    lon_change = (lon - gps_lon) * 111320.0 * math.cos(math.radians(gps_lat))
    world_vec = np.array([lon_change, lat_change, -float(altitude)])

    # Normalize yaw: DJI 0° = North, ENU 0° = East
    flight_yaw = (flight_yaw - 90) % 360

    # Handle nadir ambiguity
    if abs(gimbal_pitch + 90) < 1.0:  # Nadir
        if abs(gimbal_roll) > 90:     # Roll-flip case
            gimbal_yaw = (gimbal_yaw - 180) % 360
            gimbal_roll = 0.0
        effective_yaw = flight_yaw
    else:
        effective_yaw = (flight_yaw + gimbal_yaw) % 360

    # Build rotations
    R_yaw = rotation_matrix(effective_yaw, 0, 0)
    R_pitch_roll = rotation_matrix(0, gimbal_pitch, gimbal_roll)
    R_total = R_pitch_roll @ R_yaw
    vec_in_gimbal_frame = R_total @ world_vec


    # --- Try different ENU → Camera conventions ---
    if convention == "paper":
        # From the paper you shared
        gimbal_to_optical = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
    elif convention == "dji":
        # Your original attempt
        gimbal_to_optical = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
    elif convention == "alt1":
        gimbal_to_optical = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    elif convention == "alt2":
        gimbal_to_optical = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
    else:
        raise ValueError(f"Unknown convention: {convention}")

    cam_vec = gimbal_to_optical @ vec_in_gimbal_frame

    if debug:
        print(f"[{convention}] cam_vec:", cam_vec)

    # Projection
    p_homogeneous = K @ cam_vec
    w = p_homogeneous[2]

    if w <= 1e-6:
        return None, None

    pixel_x = p_homogeneous[0] / w
    pixel_y = p_homogeneous[1] / w

    pixel_y = image_height - pixel_y
    pixel_x = image_width - pixel_x

    if debug:
        print(f"[{convention}] pixel (x,y): {pixel_x:.1f}, {pixel_y:.1f}")

    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return int(round(pixel_x)), int(round(pixel_y))
    return None, None

# --- Main Label Generation Loop (no changes below) ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
if SAVE_OVERLAY_IMAGE and not os.path.exists(overlay_output_folder):
    os.makedirs(overlay_output_folder)

with open(geojson_file_path, 'r') as f:
    geojson_data = json.load(f)

post_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'Point']
row_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'LineString']

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    image_path = os.path.join(image_folder, image_name)
    exif_data = extract_exif(image_path)

    print(f"Extracted EXIF for {image_name}: {exif_data}")
    
    required_keys = ['image_width', 'image_height', 'altitude_above_ground', 'fov_deg',
                     'gps_latitude', 'gps_longitude', 'flight_yaw_degree']
    if not exif_data or any(exif_data.get(k) is None for k in required_keys):
        print(f"Skipping {image_name} due to missing essential metadata.")
        continue

    print(f"Processing image: {image_name}")
    
    K = build_camera_matrix(
        exif_data['image_width'],
        exif_data['image_height'],
        exif_data['fov_deg']
    )

    image_size = (exif_data['image_width'], exif_data['image_height'])
    post_mask = Image.new('L', image_size, 0)
    row_mask = Image.new('L', image_size, 0)
    draw_post = ImageDraw.Draw(post_mask)
    draw_row = ImageDraw.Draw(row_mask)
    image_center_x, image_center_y = image_size[0] / 2, image_size[1] / 2
    max_distance = math.sqrt(image_center_x**2 + image_center_y**2)

    flight_pitch = exif_data.get('flight_pitch_degree', 0) or 0
    flight_roll = exif_data.get('flight_roll_degree', 0) or 0

    for post in post_features:
        post_lon, post_lat = post['geometry']['coordinates']
        pixel_x, pixel_y = get_pixel_from_gps(
            post_lat, post_lon,
            exif_data['flight_yaw_degree'], flight_pitch, flight_roll,
            exif_data['gimbal_yaw_degree'], exif_data['gimbal_pitch_degree'], exif_data['gimbal_roll_degree'],
            exif_data['gps_latitude'], exif_data['gps_longitude'],
            exif_data['altitude_above_ground'],
            K, image_size[0], image_size[1],
            debug=False
        )

        if pixel_x is not None:
            distance_from_center = math.sqrt((pixel_x - image_center_x)**2 + (pixel_y - image_center_y)**2)
            scaling_factor = 1 + ((distance_from_center / max_distance) * 2)
            post_radius = int(POST_RADIUS_BASE * scaling_factor)
            draw_post.ellipse((pixel_x - post_radius, pixel_y - post_radius, pixel_x + post_radius, pixel_y + post_radius), fill=255)

    for row in row_features:
        pixel_coords = []
        for lon, lat in row['geometry']['coordinates']:
            pixel_x, pixel_y = get_pixel_from_gps(
                lat, lon,
                exif_data['flight_yaw_degree'], flight_pitch, flight_roll,
                exif_data['gimbal_yaw_degree'], exif_data['gimbal_pitch_degree'], exif_data['gimbal_roll_degree'],
                exif_data['gps_latitude'], exif_data['gps_longitude'],
                exif_data['altitude_above_ground'],
                K, image_size[0], image_size[1],
                debug=False
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
        post_mask_np = np.array(post_mask_blurred)
        row_mask_np = np.array(row_mask_blurred)
        post_heatmap = cv2.applyColorMap(post_mask_np, cv2.COLORMAP_JET)
        row_heatmap = cv2.applyColorMap(row_mask_np, cv2.COLORMAP_JET)
        post_heatmap_pil = Image.fromarray(cv2.cvtColor(post_heatmap, cv2.COLOR_BGR2RGB))
        row_heatmap_pil = Image.fromarray(cv2.cvtColor(row_heatmap, cv2.COLOR_BGR2RGB))
        overlay_img = Image.new('RGB', image_size, (0, 0, 0))
        overlay_img.paste(row_heatmap_pil, (0, 0), row_mask_blurred)
        overlay_img.paste(post_heatmap_pil, (0, 0), post_mask_blurred)
        blended_img = Image.blend(original_img, overlay_img, alpha=0.5)
        blended_img.save(os.path.join(overlay_output_folder, f"overlay_{image_name}"))
        print(f" - Saved overlay image to {os.path.join(overlay_output_folder, f'overlay_{image_name}')}")

    print(f" - Generated post and row masks for {image_name}")
    print("-" * 20)