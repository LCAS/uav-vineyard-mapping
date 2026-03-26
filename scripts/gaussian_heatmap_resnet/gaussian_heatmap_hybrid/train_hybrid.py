import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import csv
from datetime import datetime
import yaml

# =======================
# 1. CONFIGURATION
# =======================
# Dataset Selection Flag
USE_GPS_MASKS = False  # Set to True for GPS masks, False for YOLO labels

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Choose dataset based on flag
if USE_GPS_MASKS:
    DATASET_ROOT = os.path.join(
        SCRIPT_DIR,
        "..",
        "dataset_gps_masks",
        "riseholme",
        "full_res",
    )
    MASK_FOLDERS = {"posts_masks": "posts", "vine_masks": "vines", "rows_masks": "rows"}
else:
    DATASET_ROOT = os.path.join(
        SCRIPT_DIR,
        "..",
        "heatmap_masks_from_yolo_labels",
        "vineyard_object_detection_1",
    )
    MASK_FOLDERS = {"pole": "poles", "trunk": "trunks", "vine_row": "rows"}

BACKBONE = "resnet50"  # options: "resnet50", "resnet101"
DATASET_LABEL = "gps" if USE_GPS_MASKS else "yolo"
RUN_NAME = f"hybrid_train_{DATASET_LABEL}_{BACKBONE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_DIR = os.path.join("results_hybrid", RUN_NAME)
MODEL_OUTPUT_PATH = os.path.join(RESULTS_DIR, "best_hybrid_model.pth")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.csv")

# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
# IMAGE_SIZE = (640, 480)  # (width, height) - Must be multiples of 32
IMAGE_SIZE = (1280, 960)  # Larger size for better separation, but requires more GPU memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 20
NUM_WORKERS = 4

# Hybrid Loss Weights
# MSE (Regression) is usually very small (~0.001), so we multiply it to match BCE (~0.5)
# REDUCED from 1000.0 to prevent exploding gradients
LAMBDA_REG = 10.0  # Weight for Pole/Trunk Heatmaps
LAMBDA_SEG = 1.0   # Weight for Vine Row Segmentation
GRADIENT_CLIP = 1.0  # Clip gradients to prevent explosion

# Gaussian Target Generation
GAUSSIAN_SIGMA = 15  # Spread of the blob in pixels

os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================
# 2. DATASET (Generates Heatmaps on the Fly)
# =======================
class HybridVineyardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(640, 480), sigma=15):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.sigma = sigma
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        # Debug: Report paths
        print(f"📁 Images Dir: {images_dir}")
        print(f"📁 Masks Dir: {masks_dir}")
        print(f"📁 Images Dir exists: {os.path.exists(images_dir)}")
        print(f"📁 Masks Dir exists: {os.path.exists(masks_dir)}")
        if os.path.exists(masks_dir):
            print(f"📁 Subdirs in masks: {os.listdir(masks_dir)}")
        print(f"📊 Total images found: {len(self.image_files)}")
        if self.image_files:
            print(f"🖼️  Sample images: {self.image_files[:3]}")

        # Quick sanity check: count available masks per class
        self._mask_counts = {list(MASK_FOLDERS.keys())[0]: 0, list(MASK_FOLDERS.keys())[1]: 0, list(MASK_FOLDERS.keys())[2]: 0}
        for img_name in self.image_files:
            base_name = os.path.splitext(img_name)[0]
            for folder_name in MASK_FOLDERS.keys():
                mask_path = os.path.join(self.masks_dir, folder_name, f"{base_name}_mask.png")
                if os.path.exists(mask_path):
                    self._mask_counts[folder_name] += 1

        # Debug: Check first mask path construction for each class
        if self.image_files:
            first_img = self.image_files[0]
            base_name = os.path.splitext(first_img)[0]
            print(f"🔍 First image: {first_img} -> base_name: {base_name}")
            for folder_name, label in MASK_FOLDERS.items():
                expected_path = os.path.join(self.masks_dir, folder_name, f"{base_name}_mask.png")
                exists = os.path.exists(expected_path)
                print(f"   {label}: {expected_path} (exists: {exists})")
        counts_str = ", ".join([f"{list(MASK_FOLDERS.values())[i]}: {self._mask_counts[list(MASK_FOLDERS.keys())[i]]}/{len(self.image_files)}" for i in range(len(MASK_FOLDERS))])
        print(f"📊 Mask availability -> {counts_str}")

        # Sanity check: report non-empty stats for the first available mask of each class
        for folder_name, label in MASK_FOLDERS.items():
            sample_path = None
            for img_name in self.image_files:
                base_name = os.path.splitext(img_name)[0]
                candidate = os.path.join(self.masks_dir, folder_name, f"{base_name}_mask.png")
                if os.path.exists(candidate):
                    sample_path = candidate
                    break
            if sample_path:
                try:
                    mask = Image.open(sample_path).convert("L")
                    mask = mask.resize(self.target_size, Image.NEAREST)
                    mask_np = np.array(mask)
                    nonzero = np.count_nonzero(mask_np)
                    total = mask_np.size
                    print(
                        f"🧪 Sample {label} mask -> {os.path.basename(sample_path)}: "
                        f"min={mask_np.min()} max={mask_np.max()} nonzero={nonzero}/{total}"
                    )
                except Exception as e:
                    print(f"⚠️ Failed to read sample {cls} mask: {sample_path} ({e})")

    def __len__(self):
        return len(self.image_files)

    def generate_gaussian_target(self, mask_path):
        """Generates a Gaussian heatmap from a binary mask of objects."""
        # Initialize empty heatmap
        heatmap = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.float32)
        
        if not os.path.exists(mask_path):
            return heatmap

        # Load binary mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask_np = np.array(mask)

        # Find centroids of objects
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids_map = np.zeros_like(mask_np, dtype=np.float32)
        has_objects = False
        
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if 0 <= cY < self.target_size[1] and 0 <= cX < self.target_size[0]:
                    centroids_map[cY, cX] = 1.0
                    has_objects = True

        if not has_objects:
            return heatmap

        # Apply Gaussian Blur to create the "blob"
        k_size = int(6 * self.sigma) | 1  # Force odd kernel size
        heatmap = cv2.GaussianBlur(centroids_map, (k_size, k_size), self.sigma)
        
        # Normalize peak to 1.0
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
            
        return heatmap

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        try:
            # --- 1. Load Image ---
            image = Image.open(img_path).convert("RGB").resize(self.target_size, Image.BILINEAR)
            image = np.array(image) / 255.0  # Normalize 0-1
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        except Exception as e:
            print(f"⚠️ Failed to load image {img_name}: {e}. Using black image as fallback.")
            image = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        # --- 2. Generate Targets ---
        # Dynamically construct mask paths based on dataset configuration
        mask_folder_names = list(MASK_FOLDERS.keys())
        pole_path = os.path.join(self.masks_dir, mask_folder_names[0], f"{base_name}_mask.png")
        trunk_path = os.path.join(self.masks_dir, mask_folder_names[1], f"{base_name}_mask.png")
        row_path = os.path.join(self.masks_dir, mask_folder_names[2], f"{base_name}_mask.png")
        
        # A. Regression Targets (Heatmaps) -> Stack first two masks
        # If trunk masks don't exist, this just returns a black map (safe)
        pole_heatmap = self.generate_gaussian_target(pole_path)
        trunk_heatmap = self.generate_gaussian_target(trunk_path)
        
        # Shape: (2, Height, Width)
        target_reg = np.stack([pole_heatmap, trunk_heatmap], axis=0)
        target_reg = torch.from_numpy(target_reg).float()

        # B. Segmentation Target (Vine Row) -> Binary Mask
        if os.path.exists(row_path):
            row_mask = Image.open(row_path).convert("L").resize(self.target_size, Image.NEAREST)
            row_mask = (np.array(row_mask) > 127).astype(np.float32)
        else:
            row_mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.float32)
            
        # Shape: (1, Height, Width)
        target_seg = torch.from_numpy(row_mask).unsqueeze(0).float()

        return image, target_reg, target_seg

# =======================
# 3. HYBRID MODEL (Dual Head)
# =======================
class HybridUNetResNet(nn.Module):
    def __init__(self, backbone="resnet101"):
        super().__init__()
        
        # --- Encoder (ResNet) ---
        if backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            enc_ch = [64, 256, 512, 1024, 2048]
        elif backbone == "resnet101":
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            enc_ch = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("Backbone not supported")

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # --- Decoder (U-Net) ---
        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)

        # --- HEADS (The important change!) ---
        # Head 1: Regression (Pole, Trunk) -> 2 Channels
        self.head_reg = nn.Conv2d(64, 2, 1)
        
        # Head 2: Segmentation (Vine Row) -> 1 Channel
        self.head_seg = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        # Decoder
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

        # Outputs
        # 1. Regression (Sigmoid for 0-1 range heatmap)
        out_reg = torch.sigmoid(self.head_reg(d1))
        out_reg = F.interpolate(out_reg, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 2. Segmentation (Raw logits for BCEWithLogitsLoss)
        out_seg = self.head_seg(d1)
        out_seg = F.interpolate(out_seg, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out_reg, out_seg

# =======================
# 4. TRAINING SETUP
# =======================
def compute_metrics(pred_seg, target_seg):
    """Computes IoU for the Vine Row channel"""
    pred_mask = (torch.sigmoid(pred_seg) > 0.5).float()
    
    intersection = (pred_mask * target_seg).sum()
    union = (pred_mask + target_seg).sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

# Load Datasets
dataset_type = "GPS Masks" if USE_GPS_MASKS else "YOLO Labels"
print(f"📂 Loading Datasets ({dataset_type})...")
print(f"📍 Dataset Root: {DATASET_ROOT}")

# Determine mask directory path (GPS uses train/ directly, YOLO uses train/masks/)
TRAIN_MASKS_DIR = os.path.join(DATASET_ROOT, "train") if USE_GPS_MASKS else os.path.join(DATASET_ROOT, "train/masks")
VALID_MASKS_DIR = os.path.join(DATASET_ROOT, "valid") if USE_GPS_MASKS else os.path.join(DATASET_ROOT, "valid/masks")

train_dataset = HybridVineyardDataset(
    os.path.join(DATASET_ROOT, "train/images"), 
    TRAIN_MASKS_DIR,
    target_size=IMAGE_SIZE, sigma=GAUSSIAN_SIGMA
)
valid_dataset = HybridVineyardDataset(
    os.path.join(DATASET_ROOT, "valid/images"), 
    VALID_MASKS_DIR,
    target_size=IMAGE_SIZE, sigma=GAUSSIAN_SIGMA
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

# Initialize Model & Loss
print(f"🏗️ Building {BACKBONE} Hybrid Model...")
model = HybridUNetResNet(backbone=BACKBONE).to(DEVICE)

# Loss Functions
criterion_reg = nn.MSELoss()           # For Poles/Trunks
criterion_seg = nn.BCEWithLogitsLoss() # For Vine Rows

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# Logging
with open(METRICS_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_mse_reg", "val_iou_row"])

# Save Config
with open(os.path.join(RESULTS_DIR, "config.yaml"), "w") as f:
    yaml.dump({
        "backbone": BACKBONE, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        "image_size": IMAGE_SIZE, "sigma": GAUSSIAN_SIGMA,
        "lambda_reg": LAMBDA_REG, "lambda_seg": LAMBDA_SEG
    }, f)

# =======================
# 5. TRAINING LOOP
# =======================
best_val_loss = float('inf')
epochs_no_improve = 0

print(f"🚀 Starting training on {DEVICE}")

for epoch in range(NUM_EPOCHS):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for images, target_reg, target_seg in pbar:
        images = images.to(DEVICE)
        target_reg = target_reg.to(DEVICE)
        target_seg = target_seg.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            pred_reg, pred_seg = model(images)
            
            # Calculate Losses
            loss_reg = criterion_reg(pred_reg, target_reg)
            loss_seg = criterion_seg(pred_seg, target_seg)
            
            # Weighted Sum
            total_loss = (LAMBDA_REG * loss_reg) + (LAMBDA_SEG * loss_seg)
            
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", 
                         mse=f"{loss_reg.item():.5f}", 
                         bce=f"{loss_seg.item():.4f}")
    
    epoch_train_loss = running_loss / len(train_loader)
    
    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    running_mse = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, target_reg, target_seg in tqdm(valid_loader, desc="[Valid]"):
            images = images.to(DEVICE)
            target_reg = target_reg.to(DEVICE)
            target_seg = target_seg.to(DEVICE)
            
            pred_reg, pred_seg = model(images)
            
            loss_reg = criterion_reg(pred_reg, target_reg)
            loss_seg = criterion_seg(pred_seg, target_seg)
            total_loss = (LAMBDA_REG * loss_reg) + (LAMBDA_SEG * loss_seg)
            
            running_val_loss += total_loss.item()
            running_mse += loss_reg.item()
            running_iou += compute_metrics(pred_seg, target_seg)
            
    avg_val_loss = running_val_loss / len(valid_loader)
    avg_val_mse = running_mse / len(valid_loader)
    avg_val_iou = running_iou / len(valid_loader)
    
    print(f"   Summary -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MSE (Poles): {avg_val_mse:.6f} | IoU (Rows): {avg_val_iou:.4f}")
    
    # --- CHECKPOINTING ---
    with open(METRICS_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_train_loss, avg_val_loss, avg_val_mse, avg_val_iou])
        
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
        print(f"   ✨ Saved Best Model to {MODEL_OUTPUT_PATH}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"   ⚠️ No improvement ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
        
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print("🛑 Early stopping triggered.")
        break

print("✅ Training Finished.")