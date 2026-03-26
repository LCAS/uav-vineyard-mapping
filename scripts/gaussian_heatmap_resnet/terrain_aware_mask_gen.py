"""
Script Name: terrain_aware_mask_gen_final.py
Description: Production version of the mask generator.
             Uses empirically tuned Focal Length (34.82mm) which outperformed
             Altitude Corrections.
"""

import json
import os
import math
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFilter
import cv2
import rasterio
from pyproj import Transformer

# ==========================================
#      USER CONFIGURATION
# ==========================================

# 1. DRONE PROFILE
# Options: 
# H20 Risehome / Riccardo's Drone
# P1 Agiri Tech Centre
# M3M Oufields DJI Mavic 3 Multispectral 20MP RGB Wide Camera (4/3 CMOS Sensor)
# RX1RM2 Agri Tech Centre fixed wing UAV
CURRENT_PROFILE = 'H20' 

# # 2. DATA PATHS
image_folder = "../../images/jojo/riccardo/DJI_202507311147_029_jojo3-120"
output_folder = "./terrain_aware_mask_gen/heatmap_masks/riccardo/jojo/DJI_202507311147_029_jojo3-120"

# image_folder = "../../images/outfields/jojo/topdown"
# output_folder = "./terrain_aware_mask_gen/heatmap_masks/outfields/jojo/topdown"
geojson_file_path = "../../ground_truth/jojo/jojo_pole_locations.geojson"
dem_path = "../../images/agri_tech_centre/dsm/DSM_JOJO's.tif"

# Convert relative paths to absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, image_folder)
output_folder = os.path.join(script_dir, output_folder)
geojson_file_path = os.path.join(script_dir, geojson_file_path)
dem_path = os.path.join(script_dir, dem_path)

# 3. TUNING (Locked to "Golden" Values)
# These are set to 0.0 because the 34.82mm focal length fix solved the drift.
YAW_OFFSET_DEG = 0.0
EAST_OFFSET_METERS = 0.0
NORTH_OFFSET_METERS = 0.0
ALT_OFFSET_METERS = 0.0

# Output Settings
SAVE_OVERLAY_IMAGE = True
overlay_output_folder = os.path.join(output_folder, "overlays")

# POST_RADIUS_BASE = 100
# ROW_WIDTH_BASE = 100
# BLUR_RADIUS_POST = 50
# BLUR_RADIUS_ROW = 50

POST_RADIUS_BASE = 60
ROW_WIDTH_BASE = 60
BLUR_RADIUS_POST = 30
BLUR_RADIUS_ROW = 30

# POST_RADIUS_BASE = 50
# ROW_WIDTH_BASE = 50
# BLUR_RADIUS_POST = 25
# BLUR_RADIUS_ROW = 25

# ==========================================
#      CAMERA PROFILES
# ==========================================
DRONE_PROFILES = {
    'H20': { 
        "sensor_width_mm": 6.17, 
        "sensor_height_mm": 4.55, 
        "focal_length_mm": 4.5 
    },
    'P1': { 
        "sensor_width_mm": 35.9, 
        "sensor_height_mm": 24.0, 
        "focal_length_mm": 35.0
    },
    'M3M': { 
        "sensor_width_mm": 17.3, 
        "sensor_height_mm": 13.0, 
        "focal_length_mm": 12.3
    },
    'RX1RM2': { 
        "sensor_width_mm": 35.9, 
        "sensor_height_mm": 24.0, 
        "focal_length_mm": 35.0
    }
}

# ==========================================
#      LOGIC
# ==========================================

def get_elevation_from_dem(lat, lon, dem_dataset):
    """Samples DEM elevation at lat/lon."""
    if not dem_dataset: return 0
    try:
        if dem_dataset.crs.is_geographic:
            vals = dem_dataset.sample([(lon, lat)])
        else:
            transformer = Transformer.from_crs("EPSG:4326", dem_dataset.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            vals = dem_dataset.sample([(x, y)])
        
        val = next(vals)[0]
        # Filter out NoData values
        if val < -500 or val > 9000: return 0
        return val
    except:
        return 0 

def extract_exif(image_path):
    """Extracts metadata using ExifTool."""
    try:
        cmd = ['exiftool', '-json', '-n', image_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not result.stdout: return None
        meta = json.loads(result.stdout)[0]
        
        # RX1RM2 uses different metadata field names
        # Camera is FIXED to drone, pointing straight down
        if CURRENT_PROFILE == 'RX1RM2':
            abs_alt = meta.get('GPSAltitude')  # Direct GPS altitude
            # Use drone orientation directly (no gimbal)
            yaw = meta.get('Yaw') or 0
            pitch = -90.0  # Fixed pointing straight down
            roll = meta.get('Roll') or 0
            width = meta.get('SonyImageWidth') or meta.get('ImageWidth')
            height = meta.get('SonyImageHeight') or meta.get('ImageHeight')
        else:
            # For other drones (H20, P1, M3M) with gimbals
            abs_alt = meta.get('AbsoluteAltitude') or meta.get('GPSAltitude')
            
            yaw = meta.get('GimbalYawDegree') or meta.get('GimbalYaw')
            if yaw is None: yaw = meta.get('FlightYawDegree') or meta.get('Yaw') or 0
            
            pitch = meta.get('GimbalPitchDegree') or meta.get('GimbalPitch')
            if pitch is None: pitch = meta.get('FlightPitchDegree') or meta.get('Pitch') or -90
            
            roll = meta.get('GimbalRollDegree') or meta.get('GimbalRoll') or 0
            width = meta.get('ImageWidth')
            height = meta.get('ImageHeight')

        # Get Focal Length from Profile
        profile_focal = DRONE_PROFILES[CURRENT_PROFILE]['focal_length_mm']
        
        return {
            "lat": float(meta.get('GPSLatitude')),
            "lon": float(meta.get('GPSLongitude')),
            "alt": float(abs_alt),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "focal_length": profile_focal,
            "width": int(width),
            "height": int(height)
        }
    except Exception as e:
        print(f"EXIF Error: {e}")
        return None

def get_rotation_matrix_ned_to_body(yaw_deg, pitch_deg, roll_deg):
    """Constructs 3D Rotation Matrix."""
    y = np.radians(yaw_deg + YAW_OFFSET_DEG)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    
    # Rz (Yaw)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], 
                   [np.sin(y),  np.cos(y), 0], 
                   [0,          0,         1]])
    # Ry (Pitch)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], 
                   [0,         1, 0], 
                   [-np.sin(p), 0, np.cos(p)]])
    # Rx (Roll)
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(r), -np.sin(r)], 
                   [0, np.sin(r),  np.cos(r)]])
    
    return Rz @ Ry @ Rx

def compute_projection(exif, post_lat, post_lon, post_alt, K):
    """
    Projects 3D GPS point to 2D Pixel.
    """
    # 1. Convert to UTM (Meters)
    utm_zone = int((exif['lon'] + 180) / 6) + 1
    crs_utm = f"EPSG:326{utm_zone}" if exif['lat'] >= 0 else f"EPSG:327{utm_zone}"
    transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)

    cam_e, cam_n = transformer.transform(exif['lon'], exif['lat'])
    post_e, post_n = transformer.transform(post_lon, post_lat)
    
    # Manual Offsets (if any remain)
    cam_e -= EAST_OFFSET_METERS 
    cam_n -= NORTH_OFFSET_METERS
    
    # Vector: Camera -> Post
    dE = post_e - cam_e
    dN = post_n - cam_n
    dU = post_alt - (exif['alt'] + ALT_OFFSET_METERS)
    
    # ENU -> NED Conversion
    P_ned = np.array([dN, dE, -dU])
    
    # Rotation (NED -> Body)
    R_ned_body = get_rotation_matrix_ned_to_body(exif['yaw'], exif['pitch'], exif['roll'])
    P_body = R_ned_body.T @ P_ned
    
    # Body -> Camera Frame
    x_cam = P_body[1]
    y_cam = P_body[2]
    z_cam = P_body[0]
    
    if z_cam <= 0: return None

    # Project
    # Note: K1/K2 Distortion removed as 34.82mm fixed the linear scale issues
    pts, _ = cv2.projectPoints(np.array([[x_cam, y_cam, z_cam]]), 
                               np.zeros(3), np.zeros(3), K, None)
    
    return int(pts[0][0][0]), int(pts[0][0][1])

def compute_camera_matrix(exif, profile):
    """Calculates Intrinsic Matrix K."""
    fx = exif['focal_length'] * (exif['width'] / profile['sensor_width_mm'])
    fy = exif['focal_length'] * (exif['height'] / profile['sensor_height_mm'])
    cx = exif['width'] / 2
    cy = exif['height'] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# ==========================================
#      MAIN EXECUTION
# ==========================================
print(f"--- Mask Gen Final (Production) ---")
print(f"Profile: {CURRENT_PROFILE} (Focal Length: {DRONE_PROFILES[CURRENT_PROFILE]['focal_length_mm']}mm)")

if not os.path.exists(output_folder): os.makedirs(output_folder)
if SAVE_OVERLAY_IMAGE and not os.path.exists(overlay_output_folder): os.makedirs(overlay_output_folder)

dem_ds = None
if os.path.exists(dem_path):
    dem_ds = rasterio.open(dem_path)
else:
    print("WARNING: DEM not found. Calculations will assume flat earth.")

# Load GeoJSON
with open(geojson_file_path, 'r') as f:
    geojson_data = json.load(f)

# Process Images
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    exif = extract_exif(img_path)
    
    if not exif:
        print(f"Failed to extract EXIF from {img_name}")
        continue
    
    print(f"Processing {img_name}...")
    print(f"  Yaw: {exif['yaw']}, Pitch: {exif['pitch']}, Roll: {exif['roll']}")
    
    K = compute_camera_matrix(exif, DRONE_PROFILES[CURRENT_PROFILE])
    
    post_mask = Image.new('L', (exif['width'], exif['height']), 0)
    row_mask = Image.new('L', (exif['width'], exif['height']), 0)
    draw_post = ImageDraw.Draw(post_mask)
    draw_row = ImageDraw.Draw(row_mask)
    
    row_pixels = {}
    posts_found = 0
    posts_projected = 0
    posts_in_frame = 0
    
    # Loop through Posts
    for feat in geojson_data['features']:
        if feat['geometry']['type'] != 'Point': continue
        
        posts_found += 1
        plon, plat = feat['geometry']['coordinates']
        palt = get_elevation_from_dem(plat, plon, dem_ds)
        
        if palt == 0: palt = exif['alt'] - 30 
        
        res = compute_projection(exif, plat, plon, palt, K)
        
        if res:
            posts_projected += 1
            px, py = res
            if 0 <= px < exif['width'] and 0 <= py < exif['height']:
                posts_in_frame += 1
                draw_post.ellipse((px-POST_RADIUS_BASE, py-POST_RADIUS_BASE, px+POST_RADIUS_BASE, py+POST_RADIUS_BASE), fill=255)
                
                rn = feat['properties']['row_number']
                if rn not in row_pixels: row_pixels[rn] = []
                row_pixels[rn].append((px, py))
    
    print(f"  Posts found: {posts_found}, Projected: {posts_projected}, In frame: {posts_in_frame}")
    
    # Draw Rows
    for rn, pts in row_pixels.items():
        if len(pts) > 1:
            pts.sort(key=lambda x: x[1])
            draw_row.line(pts, fill=255, width=ROW_WIDTH_BASE)
            
    # Save Outputs
    # 1. Prepare the blurred masks (Grayscale)
    # We need these to tell the computer WHERE to draw the colors
    mask_post_blurred = post_mask.filter(ImageFilter.GaussianBlur(BLUR_RADIUS_POST))
    mask_row_blurred = row_mask.filter(ImageFilter.GaussianBlur(BLUR_RADIUS_ROW))
    
    # Save raw training masks (always saved)
    output_name_post = f"posts_{img_name.rsplit('.', 1)[0]}.png"
    output_name_row = f"rows_{img_name.rsplit('.', 1)[0]}.png"
    mask_post_blurred.save(os.path.join(output_folder, output_name_post))
    mask_row_blurred.save(os.path.join(output_folder, output_name_row))

    # Optionally save overlay images
    if SAVE_OVERLAY_IMAGE:
        # 2. Create the Heatmaps (Colors)
        # This creates a square image that is mostly Blue (background) and Red/Yellow (posts)
        hm_post_np = cv2.applyColorMap(np.array(mask_post_blurred), cv2.COLORMAP_JET)
        hm_row_np = cv2.applyColorMap(np.array(mask_row_blurred), cv2.COLORMAP_JET)
        
        # Convert to PIL RGB
        hm_post_pil = Image.fromarray(cv2.cvtColor(hm_post_np, cv2.COLOR_BGR2RGB))
        hm_row_pil = Image.fromarray(cv2.cvtColor(hm_row_np, cv2.COLOR_BGR2RGB))
        
        # 3. COMPOSITE (The Fix for the Purple Background)
        # Instead of blending the whole square, we use the mask as an "Alpha Channel".
        # Logic: If mask pixel is Black -> Show Original Image.
        #        If mask pixel is White -> Show Heatmap Color.
        
        orig = Image.open(img_path).convert("RGB")
        
        # Overlay Rows first
        overlay = Image.composite(hm_row_pil, orig, mask_row_blurred)
        
        # Overlay Posts on top
        overlay = Image.composite(hm_post_pil, overlay, mask_post_blurred)
        
        overlay.save(os.path.join(overlay_output_folder, f"overlay_{img_name}"))

if dem_ds: dem_ds.close()
print("Done.")