import json
import os
import math
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFilter
import cv2
import rasterio
from pyproj import Transformer

# --- User-Provided Information ---
dsm_file_path = "../../images/riseholme/august_2024/dem/39_feet/dsm.tif"
image_folder = "../../images/riseholme/august_2024/39_feet"
geojson_file_path = "../../ground_truth/riseholme/riseholme_pole_locations.geojson"
output_folder = "./heatmap_masks/riseholme/39_feet/"
overlay_output_folder = "./annotated_images/riseholme/39_feet/"

# --- Flag to save mask images overlaid on input images ---
SAVE_OVERLAY_IMAGE = True

# --- Base values for mask sizes at the image center ---
POST_RADIUS_BASE = 30
ROW_WIDTH_BASE = 60
BLUR_RADIUS_POST = 15
BLUR_RADIUS_ROW = 30

debug_output_folder = "./annotated_images/riseholme/39_feet/debug_projections/"
os.makedirs(debug_output_folder, exist_ok=True)

# --- Load DSM ---
dsm_dataset = rasterio.open(dsm_file_path)
transformer = Transformer.from_crs("EPSG:4326", dsm_dataset.crs, always_xy=True)

def get_dsm_elevation(lat, lon):
    """Return DSM elevation at a given lat/lon."""
    x, y = transformer.transform(lon, lat)
    row, col = dsm_dataset.index(x, y)
    row = max(0, min(row, dsm_dataset.height - 1))
    col = max(0, min(col, dsm_dataset.width - 1))
    elevation = dsm_dataset.read(1)[row, col]
    return float(elevation)

def save_debug_projection(image_path, projected_points, output_path):
    """Draw projected points on the original image and save to disk."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for px, py in projected_points:
        r = 5
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 0, 0))
    img.save(output_path)

# --- Camera / projection helpers ---
def build_camera_matrix(img_width, img_height, hfov_deg):
    hfov = math.radians(hfov_deg)
    f_x = (img_width / 2) / math.tan(hfov / 2)
    f_y = f_x
    c_x = img_width / 2
    c_y = img_height / 2
    return np.array([[f_x, 0, c_x],[0, f_y, c_y],[0, 0, 1]])

def dms_to_decimal(dms_str):
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
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        if match:
            return float(match.group(0))
    return None

def rotation_matrix(yaw, pitch, roll):
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def extract_exif(image_path):
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

def get_pixel_from_gps(lon, lat,
                       flight_yaw, flight_pitch, flight_roll,
                       gimbal_yaw, gimbal_pitch, gimbal_roll,
                       gps_lon, gps_lat, altitude_above_ground,
                       K, image_width, image_height,
                       debug=False):

    # Convert GPS differences to meters (ENU)
    lat_change = (lat - gps_lat) * 111320.0
    lon_change = (lon - gps_lon) * 111320.0 * math.cos(math.radians(gps_lat))

    # Elevation adjustment using DSM
    target_elevation = get_dsm_elevation(lat, lon)
    drone_elevation = get_dsm_elevation(gps_lat, gps_lon)
    drone_alt_msl = altitude_above_ground + drone_elevation
    z_rel = -(drone_alt_msl - target_elevation)  # camera looks toward target

    world_vec = np.array([lon_change, lat_change, z_rel])

    # DJI convention adjustment
    flight_yaw = (flight_yaw - 90) % 360
    if abs(gimbal_pitch + 90) < 1.0 and abs(gimbal_roll) > 90:
        gimbal_yaw = (gimbal_yaw - 180) % 360
        gimbal_roll = 0.0
        effective_yaw = flight_yaw
    else:
        effective_yaw = (flight_yaw + gimbal_yaw) % 360

    # Rotate world vector to camera/gimbal frame
    R_total = rotation_matrix(0, gimbal_pitch, gimbal_roll) @ rotation_matrix(effective_yaw, 0, 0)
    vec_in_gimbal_frame = R_total @ world_vec

    # Correct DJI ENU → optical frame
    gimbal_to_optical = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    cam_vec = gimbal_to_optical @ vec_in_gimbal_frame

    # Project to image
    p_homogeneous = K @ cam_vec
    w = p_homogeneous[2]
    if w <= 1e-6:
        if debug:
            print(f"[DEBUG] Point behind camera: lat={lat}, lon={lon}, z_rel={z_rel}")
        return None, None

    pixel_x = p_homogeneous[0] / w
    pixel_y = image_height - (p_homogeneous[1] / w)  # flip Y

    if debug:
        print(f"[DEBUG] Projected pixel: lat={lat}, lon={lon}, x={pixel_x:.1f}, y={pixel_y:.1f}")

    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return int(round(pixel_x)), int(round(pixel_y))
    return None, None
    
def get_pixel_from_gps(lon, lat,
                       flight_yaw, flight_pitch, flight_roll,
                       gimbal_yaw, gimbal_pitch, gimbal_roll,
                       gps_lon, gps_lat, altitude_above_ground,
                       K, image_width, image_height,
                       debug=False):

    # ENU coordinates
    lat_change = (lat - gps_lat) * 111320.0
    lon_change = (lon - gps_lon) * 111320.0 * math.cos(math.radians(gps_lat))

    target_elevation = get_dsm_elevation(lat, lon)
    drone_elevation = get_dsm_elevation(gps_lat, gps_lon)
    drone_alt_msl = altitude_above_ground + drone_elevation
    z_rel = -(drone_alt_msl - target_elevation)

    world_vec = np.array([lon_change, lat_change, z_rel])

    # DJI convention adjustment
    flight_yaw = (flight_yaw - 90) % 360
    effective_yaw = (flight_yaw + gimbal_yaw) % 360

    # Rotation: yaw first, then gimbal pitch/roll
    R_total = rotation_matrix(effective_yaw, 0, 0) @ rotation_matrix(0, gimbal_pitch, gimbal_roll)
    vec_in_gimbal_frame = R_total @ world_vec

    # Correct ENU → camera optical frame (nadir down)
    gimbal_to_optical = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    cam_vec = gimbal_to_optical @ vec_in_gimbal_frame

    # Project to image
    p_homogeneous = K @ cam_vec
    w = p_homogeneous[2]
    if w <= 1e-6:
        if debug:
            print(f"[DEBUG] Point behind camera: lat={lat}, lon={lon}, z_rel={z_rel}")
        return None, None

    pixel_x = p_homogeneous[0] / w
    pixel_y = image_height - (p_homogeneous[1] / w)

    if debug:
        print(f"[DEBUG] Projected pixel: lat={lat}, lon={lon}, x={pixel_x:.1f}, y={pixel_y:.1f}")

    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return int(round(pixel_x)), int(round(pixel_y))
    return None, None



# --- Main loop ---
os.makedirs(output_folder, exist_ok=True)
if SAVE_OVERLAY_IMAGE:
    os.makedirs(overlay_output_folder, exist_ok=True)

with open(geojson_file_path, 'r') as f:
    geojson_data = json.load(f)

post_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'Point']
row_features = [f for f in geojson_data['features'] if f['geometry']['type'] == 'LineString']

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    image_path = os.path.join(image_folder, image_name)
    exif_data = extract_exif(image_path)
    required_keys = ['image_width', 'image_height', 'altitude_above_ground', 'fov_deg',
                     'gps_latitude', 'gps_longitude', 'flight_yaw_degree']
    if not exif_data or any(exif_data.get(k) is None for k in required_keys):
        print(f"Skipping {image_name} due to missing metadata.")
        continue

    K = build_camera_matrix(exif_data['image_width'], exif_data['image_height'], exif_data['fov_deg'])
    image_size = (exif_data['image_width'], exif_data['image_height'])
    post_mask = Image.new('L', image_size, 0)
    row_mask = Image.new('L', image_size, 0)
    draw_post = ImageDraw.Draw(post_mask)
    draw_row = ImageDraw.Draw(row_mask)
    image_center_x, image_center_y = image_size[0] / 2, image_size[1] / 2
    max_distance = math.sqrt(image_center_x**2 + image_center_y**2)

    flight_pitch = exif_data.get('flight_pitch_degree', 0) or 0
    flight_roll = exif_data.get('flight_roll_degree', 0) or 0

    # Draw posts
    projected_post_points = []
    for post in post_features:
        post_lon, post_lat = post['geometry']['coordinates']
        pixel_x, pixel_y = get_pixel_from_gps(
            post_lon, post_lat,
            exif_data['flight_yaw_degree'], flight_pitch, flight_roll,
            exif_data['gimbal_yaw_degree'], exif_data['gimbal_pitch_degree'], exif_data['gimbal_roll_degree'],
            exif_data['gps_longitude'], exif_data['gps_latitude'],
            exif_data['altitude_above_ground'],
            K, image_size[0], image_size[1]
        )
        if pixel_x is not None:
            projected_post_points.append((pixel_x, pixel_y))
            distance_from_center = math.sqrt((pixel_x - image_center_x)**2 + (pixel_y - image_center_y)**2)
            scaling_factor = 1 + ((distance_from_center / max_distance) * 2)
            post_radius = int(POST_RADIUS_BASE * scaling_factor)
            draw_post.ellipse((pixel_x - post_radius, pixel_y - post_radius, pixel_x + post_radius, pixel_y + post_radius), fill=255)

    # Save debug projection
    save_debug_projection(image_path, projected_post_points,
                          os.path.join(debug_output_folder, f"debug_posts_{image_name}"))

    # Draw rows
    for row in row_features:
        pixel_coords = []
        for lon, lat in row['geometry']['coordinates']:
            pixel_x, pixel_y = get_pixel_from_gps(
                lon, lat,
                exif_data['flight_yaw_degree'], flight_pitch, flight_roll,
                exif_data['gimbal_yaw_degree'], exif_data['gimbal_pitch_degree'], exif_data['gimbal_roll_degree'],
                exif_data['gps_longitude'], exif_data['gps_latitude'],
                exif_data['altitude_above_ground'],
                K, image_size[0], image_size[1]
            )
            if pixel_x is not None:
                pixel_coords.append((pixel_x, pixel_y))
        if len(pixel_coords) > 1:
            distance_from_center = math.sqrt((pixel_coords[0][0] - image_center_x)**2 + (pixel_coords[0][1] - image_center_y)**2)
            scaling_factor = 1 + ((distance_from_center / max_distance) * 2)
            row_width = int(ROW_WIDTH_BASE * scaling_factor)
            draw_row.line(pixel_coords, fill=255, width=row_width)

    # Blur and save masks
    post_mask_blurred = post_mask.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS_POST))
    row_mask_blurred = row_mask.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS_ROW))
    post_mask_blurred.save(os.path.join(output_folder, f"posts_mask_{image_name}"))
    row_mask_blurred.save(os.path.join(output_folder, f"rows_mask_{image_name}"))

    # Overlay visualization
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

    print(f" - Generated post and row masks for {image_name}")
