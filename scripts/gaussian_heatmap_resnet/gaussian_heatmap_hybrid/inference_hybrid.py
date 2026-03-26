import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
import cv2
import json
from tqdm import tqdm
from skimage.feature import peak_local_max

# --- Import your GPS Utils ---
# Ensure image_gps_pixel_show_poles.py is in the same folder
import sys
sys.path.append(os.getcwd()) # Ensure current dir is in path
import image_gps_pixel_show_poles

# ======================
# 1. CONFIGURATION
# ======================
# Update this path to your actual best model
MODEL = "hybrid_train_resnet101_20260211_120756"
MODEL_PATH = f"results_hybrid/{MODEL}/best_hybrid_model.pth"
IMAGE_DATE = "august_2024"
ALTITUDE = 39
INPUT_DIR = f"../../../images/riseholme/{IMAGE_DATE}/{ALTITUDE}_feet/"

BACKBONE = "resnet101" # Options: "resnet50", "resnet101"

# Inference Size: Larger is better for separation, provided GPU fits it, orignal image size 4056x3040
# IMAGE_SIZE = (2028, 1520)
# IMAGE_SIZE = (1014, 760)
IMAGE_SIZE = (1280, 960)
# IMAGE_SIZE = (676, 506)
# IMAGE_SIZE = (640, 480) # For testing on CPU or smaller GPU, but expect worse separation and GPS accuracy

SAVE_OVERLAY_IMAGES = False
SAVE_HEATMAP_IMAGES = False
SAVE_NPZ = True  # Save raw heatmaps/probability maps for later thresholding

OUTPUT_DIR = f"inference_results/hybrid_test/{MODEL}/{IMAGE_DATE}/{ALTITUDE}_feet/image_size_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"

# Full Resolution (for scaling GPS coordinates back)
# Note: Code will auto-detect this from the image, but good to know
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds
CONFIDENCE_THRESHOLDS = {
    "pole": 0.4,       # Lower threshold because heatmaps are soft (MSE loss)
    "trunk": 0.4,
    "vine_row": 0.5    # Binary mask threshold
}

# GPS & Camera Specs (Riseholme H20)
FOCAL_LENGTH_MM = 4.5
SENSOR_WIDTH_MM = 6.17
SENSOR_HEIGHT_MM = 4.55

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# 2. MODEL DEFINITION
# ======================
class HybridUNetResNet(nn.Module):
    def __init__(self, backbone="resnet101"):
        super().__init__()
        
        # --- Encoder ---
        if backbone == "resnet50":
            resnet = models.resnet50(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        elif backbone == "resnet101":
            resnet = models.resnet101(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # --- Decoder ---
        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)

        # --- Heads ---
        self.head_reg = nn.Conv2d(64, 2, 1) # Heatmaps (Pole/Trunk)
        self.head_seg = nn.Conv2d(64, 1, 1) # Segmentation (Vine Row)

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        d4 = F.interpolate(self.upconv4(x5), size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)
        d3 = F.interpolate(self.upconv3(d4), size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        d2 = F.interpolate(self.upconv2(d3), size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        d1 = F.interpolate(self.upconv1(d2), size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out_reg = torch.sigmoid(self.head_reg(d1))
        out_reg = F.interpolate(out_reg, size=x.shape[2:], mode='bilinear', align_corners=True)
        out_seg = self.head_seg(d1) # Raw logits
        out_seg = F.interpolate(out_seg, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out_reg, out_seg

# ======================
# 3. LOAD MODEL
# ======================
print(f"🔄 Loading model from {MODEL_PATH}")
model = HybridUNetResNet(backbone=BACKBONE).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ======================
# 4. HELPER FUNCTIONS
# ======================
def get_peak_coordinates(heatmap, threshold, min_distance=10):
    """Finds local maxima in the Gaussian heatmap."""
    # Peak local max returns (y, x)
    coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
    return coords[:, ::-1] # Flip to (x, y)

def resize_heatmap(heatmap, target_shape):
    """Resizes heatmap to original image size for accurate GPS mapping."""
    return cv2.resize(heatmap, target_shape, interpolation=cv2.INTER_LINEAR)

# ======================
# 5. INFERENCE LOOP
# ======================
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))]
print(f"🚀 Processing {len(image_files)} images...")

# Lists to store all detections for GeoJSON
all_poles = []
all_trunks = []
all_rows = []

for file in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, file)
    
    # A. Load Image & EXIF
    image_pil = Image.open(img_path).convert("RGB")
    original_w, original_h = image_pil.size
    
    # Extract EXIF
    (flight_yaw, flight_pitch, flight_roll, gimbal_yaw, gimbal_pitch, gimbal_roll, 
     gps_lat, gps_lon, gps_alt, fov, _, _, _) = image_gps_pixel_show_poles.extract_exif(img_path)
    
    if gps_lat is None:
        print(f"Skipping {file}: No GPS data.")
        continue

    # Prepare GPS params
    flight_yaw_num = image_gps_pixel_show_poles.extract_number(flight_yaw)
    gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(gimbal_yaw)
    if gimbal_yaw_num == 0.0 or gimbal_yaw_num is None: gimbal_yaw_num = flight_yaw_num
    gps_alt_num = image_gps_pixel_show_poles.extract_number(gps_alt) if gps_alt else 0.0

    def to_gps(px, py):
        return image_gps_pixel_show_poles.get_gps_from_pixel(
            px, py, original_w, original_h, 
            flight_yaw_num, gimbal_yaw_num, 
            gps_lat, gps_lon, gps_alt_num, 
            FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM
        )
    
    # Resize for Model
    input_img = image_pil.resize(IMAGE_SIZE, Image.BILINEAR)
    image_np = np.array(input_img) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # B. Inference
    with torch.no_grad():
        pred_reg, pred_seg = model(image_tensor)
        
        # 1. Process Heatmaps (Poles/Trunks)
        heatmaps = pred_reg[0].cpu().permute(1, 2, 0).numpy()
        
        # 2. Process Segmentation (Vine Row)
        mask_logits = pred_seg[0, 0]
        # Keep probability map so thresholds can be changed later without re-running inference
        row_prob_map = torch.sigmoid(mask_logits).cpu().numpy()
        mask_row = row_prob_map > CONFIDENCE_THRESHOLDS["vine_row"]

    # C. Post-Processing & GPS Conversion
    
    # --- POLES (Channel 0) ---
    pole_map_lowres = heatmaps[:, :, 0]
    # Resize prediction back to FULL 12MP resolution
    pole_map_full = resize_heatmap(pole_map_lowres, (original_w, original_h))
    pole_peaks = get_peak_coordinates(pole_map_full, CONFIDENCE_THRESHOLDS["pole"], min_distance=25)
    
    for px, py in pole_peaks:
        lat, lon = to_gps(px, py)
        conf = float(pole_map_full[py, px])
        all_poles.append({
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"confidence": conf, "image": file}
        })

    # --- TRUNKS (Channel 1) ---
    trunk_map_lowres = heatmaps[:, :, 1]
    trunk_map_full = resize_heatmap(trunk_map_lowres, (original_w, original_h))
    trunk_peaks = get_peak_coordinates(trunk_map_full, CONFIDENCE_THRESHOLDS["trunk"], min_distance=25)

    for px, py in trunk_peaks:
        lat, lon = to_gps(px, py)
        conf = float(trunk_map_full[py, px])
        all_trunks.append({
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"confidence": conf, "image": file}
        })

    # --- VINE ROWS (Segmentation) ---
    row_mask_full = cv2.resize(mask_row.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(row_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            approx = cv2.approxPolyDP(cnt, 5.0, True)
            row_poly_px = approx.reshape(-1, 2)
            
            # Convert polygon to GPS
            row_gps = []
            for px, py in row_poly_px:
                lat, lon = to_gps(px, py)
                row_gps.append([lon, lat]) # GeoJSON uses [Lon, Lat]
            
            # Close loop
            if len(row_gps) > 2:
                row_gps.append(row_gps[0])
                all_rows.append({
                    "geometry": {"type": "Polygon", "coordinates": [row_gps]},
                    "properties": {"image": file}
                })

    # --- SAVE NPZ (RAW DATA) ---
    # Save low-res heatmaps and row probability map so thresholds can be changed later
    if SAVE_NPZ:
        try:
            npz_path = os.path.join(OUTPUT_DIR, f"{base_name}_outputs.npz")
            np.savez_compressed(
                npz_path,
                pole=pole_map_lowres.astype(np.float32),
                trunk=trunk_map_lowres.astype(np.float32),
                row=row_prob_map.astype(np.float32)
            )
        except Exception as e:
            print(f"Warning: failed to save npz for {file}: {e}")

    # D. VISUALIZATION & HEATMAP SAVING
    base_name = os.path.splitext(file)[0]
    
    # 1. Save Raw Heatmaps (Colorized for visibility)
    if SAVE_HEATMAP_IMAGES:
        # Pole Heatmap (Red/Yellow)
        pole_norm = (pole_map_full * 255).astype(np.uint8)
        pole_color = cv2.applyColorMap(pole_norm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_heatmap_pole.jpg"), pole_color)
        
        # Trunk Heatmap (Blue/Green)
        trunk_norm = (trunk_map_full * 255).astype(np.uint8)
        trunk_color = cv2.applyColorMap(trunk_norm, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_heatmap_trunk.jpg"), trunk_color)
    
    # 2. Create Overlay Image (Debug View)
    if SAVE_OVERLAY_IMAGES:
        vis_img = np.array(image_pil).copy()
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
        
        # Draw Vine Rows (Green Polygon)
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
        
        # Draw Poles (Red Circles)
        for x, y in pole_peaks:
            cv2.circle(vis_img, (x, y), 10, (0, 0, 255), -1) # Red filled
            cv2.circle(vis_img, (x, y), 10, (255, 255, 255), 2) # White rim
            
        # Draw Trunks (Blue Circles)
        for x, y in trunk_peaks:
            cv2.circle(vis_img, (x, y), 8, (255, 0, 0), -1) # Blue filled

        # Save final overlay
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_overlay.jpg"), vis_img)

# ======================
# 6. SAVE GEOJSON
# ======================
print("\n💾 Saving GeoJSON results...")

def save_geojson(features, filename):
    collection = {"type": "FeatureCollection", "features": []}
    for f in features:
        collection["features"].append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": f["properties"]
        })
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(collection, f, indent=2)

save_geojson(all_poles, "poles.geojson")
save_geojson(all_trunks, "trunks.geojson")
save_geojson(all_rows, "vine_rows.geojson")

print(f"✅ Finished! Results saved to {OUTPUT_DIR}")
print(f"   - Poles: {len(all_poles)}")
print(f"   - Trunks: {len(all_trunks)}")
print(f"   - Rows: {len(all_rows)}")