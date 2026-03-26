import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point

# --- Import your existing utils ---
import image_gps_pixel_show_poles

# --- Import your Model Class ---
# (Paste your UNetResNet class here, or import it if it's in a separate file)
from torchvision import models
import torch.nn as nn

# --- RE-DEFINING THE MODEL CLASS FOR STANDALONE RUNNING ---
# Ensure this matches EXACTLY what you trained with
class UNetResNet(nn.Module):
    def __init__(self, n_classes, backbone="resnet101"):
        super().__init__()
        if backbone == "resnet18":
            resnet = models.resnet18(weights=None) # Weights not needed for inference
            enc_ch = [64, 64, 128, 256, 512] 
        elif backbone == "resnet50":
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

        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)
        self.final = nn.Conv2d(64, n_classes, 1)

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
        d2 = self.upconv2(d3) # Fixed typo from previous discussion
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        d1 = F.interpolate(self.upconv1(d2), size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# ==========================================
# 1. FULL IMAGE INFERENCE FUNCTION (NO TILING)
# ==========================================
def predict_whole_image(model, image_path, device, num_classes=4):
    """
    Runs inference on the whole image without tiling.
    """
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    img_np = np.array(image) / 255.0
    
    model.eval()
    with torch.no_grad():
        # Convert entire image to tensor
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Run single inference on whole image
        output = model(tensor)
        full_probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy() # (C, H, W)
    
    return full_probs, original_w, original_h

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def extract_features(full_probs, class_indices, thresholds):
    """
    Extracts (x,y) coordinates for points (poles/trunks) and lines (rows).
    """
    idx_pole = class_indices["pole"]
    idx_trunk = class_indices["trunk"]
    idx_row = class_indices["vine_row"]

    # --- A. TRUNKS & POLES (Peak Finding) ---
    # Result is [[y, x], [y, x]...]
    pole_coords = peak_local_max(full_probs[idx_pole], min_distance=20, threshold_abs=thresholds["pole"])
    trunk_coords = peak_local_max(full_probs[idx_trunk], min_distance=20, threshold_abs=thresholds["trunk"])

    # Flip to (x, y) for GPS consistency
    pole_coords = pole_coords[:, ::-1]
    trunk_coords = trunk_coords[:, ::-1]

    # --- B. VINE ROWS (Skeletonization) ---
    # 1. Threshold to binary mask
    row_mask = full_probs[idx_row] > thresholds["vine_row"]
    # 2. Skeletonize to get a 1-pixel wide line
    skeleton = skeletonize(row_mask)
    # 3. Get coordinates (y, x) -> swap to (x, y)
    row_y, row_x = np.where(skeleton)
    row_coords = np.column_stack((row_x, row_y))

    return pole_coords, trunk_coords, row_coords, row_mask

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- CONFIG ---
    MODEL_PATH = "results_resnet/yolo_masks/vineyard_segmentation_paper_1/train_resnet101_20260203_135036/resnet101_vineyard_segmentation_paper_1_unet_image_size_1280x960_batch_size_2.pth" # UPDATE THIS
    IMAGE_FOLDER = "../../images/riseholme/july_2025/39_feet/"
    OUTPUT_FOLDER = "resnet_inference/vineyard_segmentation_paper_1/full_images/train_resnet101_20260203_135036/inference_results_full/39_feet/"
    
    # Camera Specs (Riseholme H20)
    FOCAL_LENGTH_MM = 4.5
    SENSOR_WIDTH_MM = 6.17
    SENSOR_HEIGHT_MM = 4.55
    
    # Class Config
    CLASS_INDICES = {"background": 0, "pole": 1, "trunk": 2, "vine_row": 3}
    THRESHOLDS = {"pole": 0.2, "trunk": 0.2, "vine_row": 0.2}

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNetResNet(n_classes=4, backbone="resnet101").to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))]

    # Initialize separate GeoJSON collections
    geojson_poles = {
        "type": "FeatureCollection",
        "features": []
    }
    geojson_trunks = {
        "type": "FeatureCollection",
        "features": []
    }
    geojson_rows = {
        "type": "FeatureCollection",
        "features": []
    }

    for img_file in tqdm(image_files):
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        
        # --- 2. Extract EXIF ---
        (flight_yaw_deg, flight_pitch_deg, flight_roll_deg, gimbal_yaw_deg, 
         gimbal_pitch_deg, gimbal_roll_deg, gps_lat, gps_lon, gps_alt, 
         fov, _, img_h, img_w) = image_gps_pixel_show_poles.extract_exif(img_path)

        if flight_yaw_deg is None or gps_lat is None:
            print(f"Skipping {img_file}: Missing EXIF")
            continue

        # Prepare Numeric Values for GPS Calc
        flight_yaw = image_gps_pixel_show_poles.extract_number(flight_yaw_deg)
        gimbal_yaw = image_gps_pixel_show_poles.extract_number(gimbal_yaw_deg)
        if gimbal_yaw == 0.0: gimbal_yaw = flight_yaw
        gps_alt_num = image_gps_pixel_show_poles.extract_number(gps_alt)
        if gps_alt_num is None: gps_alt_num = 0.0

        # --- 3. Run Inference ---
        full_probs, width, height = predict_whole_image(model, img_path, DEVICE)

        # --- 4. Extract Features (Pixel Coords) ---
        poles_px, trunks_px, rows_px, row_mask_vis = extract_features(full_probs, CLASS_INDICES, THRESHOLDS)

        # --- 5. Debug Visualization (Optional) ---
        # Save an image showing detected peaks and skeletons
        vis_img = cv2.imread(img_path)
        for x, y in poles_px:
            cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1) # Red Poles
        for x, y in trunks_px:
            cv2.circle(vis_img, (x, y), 3, (255, 0, 0), -1) # Blue Trunks
        vis_img[row_mask_vis] = (0, 255, 0) # Green Rows overlap
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"vis_{img_file}"), vis_img)

        # --- 6. Convert Pixels to GPS and Add to GeoJSON ---
        
        # Helper wrapper for the complex GPS function
        def to_gps(px, py):
            return image_gps_pixel_show_poles.get_gps_from_pixel(
                px, py, width, height, flight_yaw, gimbal_yaw, 
                gps_lat, gps_lon, gps_alt_num, 
                FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM
            )

        # Process Poles
        for x, y in poles_px:
            lat, lon = to_gps(x, y)
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"source_image": img_file}
            }
            geojson_poles["features"].append(feature)

        # Process Trunks
        for x, y in trunks_px:
            lat, lon = to_gps(x, y)
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"source_image": img_file}
            }
            geojson_trunks["features"].append(feature)

        # Process Rows (Saving as Multipoint for now, or simplify into Lines)
        # Note: Saving every single skeleton pixel as a GPS point is HEAVY.
        # Strategy: Save as a LineString if possible, or just subsample points.
        # Here we subsample every 10th pixel to keep file size manageable.
        if len(rows_px) > 0:
            row_gps_points = []
            for x, y in rows_px[::10]: # Subsample
                lat, lon = to_gps(x, y)
                row_gps_points.append([lon, lat])
            
            # Save as a MultiPoint or LineString
            feature = {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": row_gps_points},
                "properties": {"source_image": img_file}
            }
            geojson_rows["features"].append(feature)

    # Save separate GeoJSON files
    with open(os.path.join(OUTPUT_FOLDER, "poles.geojson"), "w") as f:
        json.dump(geojson_poles, f, indent=4)
    
    with open(os.path.join(OUTPUT_FOLDER, "trunks.geojson"), "w") as f:
        json.dump(geojson_trunks, f, indent=4)
    
    with open(os.path.join(OUTPUT_FOLDER, "vine_rows.geojson"), "w") as f:
        json.dump(geojson_rows, f, indent=4)

    print("Done! Results saved separately to poles.geojson, trunks.geojson, and vine_rows.geojson")
