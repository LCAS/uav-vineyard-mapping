import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import shutil
import csv
from datetime import datetime
import yaml

# =======================
# CONFIGURATION
# =======================
DATASET_ROOT = "heatmap_masks_from_yolo_labels/vineyard_segmentation_paper_1"
BACKBONE = "resnet101"  # options: "resnet18" or "resnet50" or "resnet101"
MODEL_OUTPUT_PATH = f"{BACKBONE}_vineyard_segmentation_paper_1_unet_image_size_640x480_batch_size_2.pth"

BATCH_SIZE = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
# IMAGE_SIZE = (2028, 1520) # (width, height)
# IMAGE_SIZE = (1014, 760) # (width, height)
# IMAGE_SIZE = (1280, 960) # (width, height)
IMAGE_SIZE = (640, 480) # (width, height)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 20
NUM_CLASSES = 4
CLASS_NAMES = ["background", "pole", "trunk", "vine_row"]

# --- Metrics logging setup ---
RUN_NAME = f"train_{BACKBONE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_DIR = os.path.join("results_resnet/yolo_masks/vineyard_segmentation_paper_1/", RUN_NAME) # change folder name accordingly
os.makedirs(RESULTS_DIR, exist_ok=True)
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.csv")

with open(METRICS_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "pixel_acc", "mIoU"] + [f"IoU_{c}" for c in CLASS_NAMES])

# =======================
# SAVE TRAINING PARAMETERS
# =======================
TRAINING_CONFIG_PATH = os.path.join(RESULTS_DIR, "training_config.yaml")

training_config = {
    "dataset_root": DATASET_ROOT,
    "backbone": BACKBONE,
    "model_output_path": MODEL_OUTPUT_PATH,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "image_size": IMAGE_SIZE,
    "device": str(DEVICE),
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "amp_enabled": True,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(TRAINING_CONFIG_PATH, "w") as f:
    yaml.dump(training_config, f, default_flow_style=False)

print(f"📄 Training configuration saved to: {TRAINING_CONFIG_PATH}")

# =======================
# MODEL DEFINITION (Updated for ResNet-101)
# =======================
class UNetResNet(nn.Module):
    def __init__(self, n_classes, backbone="resnet18"):
        super().__init__()

        # Choose backbone
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            enc_ch = [64, 64, 128, 256, 512] 
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            enc_ch = [64, 256, 512, 1024, 2048]
        elif backbone == "resnet101":
            # --- NEW ADDITION ---
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            # ResNet101 has the same channel width as ResNet50
            enc_ch = [64, 256, 512, 1024, 2048] 
        else:
            raise ValueError(f"Backbone '{backbone}' not supported")

        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        # Note: We use dynamic sizes based on enc_ch, so this works for both 50 and 101 automatically
        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)

        # Final classification layer
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.encoder0(x)      # Initial features
        x1 = self.pool0(x0)        # Downsample
        x2 = self.encoder1(x1)     # Layer 1
        x3 = self.encoder2(x2)     # Layer 2
        x4 = self.encoder3(x3)     # Layer 3
        x5 = self.encoder4(x4)     # Layer 4 (Bottleneck)

        # --- Decoder ---
        
        # Block 4: Upsample x5, merge with x4
        d4 = self.upconv4(x5)
        d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        # Block 3: Upsample d4, merge with x3
        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        # Block 2: Upsample d3, merge with x2
        d2 = self.upconv2(d3)  # <--- Input is d3 (from below)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        # Block 1: Upsample d2, merge with x0
        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        # --- Output ---
        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# =======================
# DATASET DEFINITION
# =======================
import torchvision.transforms.functional as TF

class VineyardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(1024, 768)):
        # target_size must be multiples of 32, e.g., (1024, 768)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size 
        self.image_files = sorted([f for f in os.listdir(images_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # 1. Load Image and resize to target_size
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.target_size, Image.BILINEAR)
        
        # Convert to Tensor
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # 2. Handle Masks
        mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        base_name = os.path.splitext(img_name)[0]
        
        for cls_idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):
            cls_mask_dir = os.path.join(self.masks_dir, cls_name)
            if not os.path.isdir(cls_mask_dir):
                continue
            
            mask_files = [f for f in os.listdir(cls_mask_dir) if base_name in f]
            if mask_files:
                cls_mask_path = os.path.join(cls_mask_dir, mask_files[0])
                
                # Load mask and resize directly to target size
                cls_mask = Image.open(cls_mask_path).convert("L")
                cls_mask = cls_mask.resize(self.target_size, Image.NEAREST)
                
                cls_mask_np = np.array(cls_mask) > 127
                mask[cls_mask_np] = cls_idx
                
        mask = torch.from_numpy(mask).long()
        return image, mask

# =======================
# METRICS FUNCTIONS
# =======================
def compute_metrics(outputs, masks, num_classes=NUM_CLASSES):
    preds = torch.argmax(outputs, dim=1)
    preds_np = preds.cpu().numpy()
    masks_np = masks.cpu().numpy()
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(preds_np == cls, masks_np == cls).sum()
        union = np.logical_or(preds_np == cls, masks_np == cls).sum()
        iou = intersection / union if union != 0 else float('nan')
        ious.append(iou)
    correct = (preds_np == masks_np).sum()
    total = np.prod(masks_np.shape)
    pixel_acc = correct / total if total > 0 else 0
    mean_iou = np.nanmean(ious)
    return pixel_acc, mean_iou, ious

# =======================
# SETUP TRAINING
# =======================
print(f"Using device: {DEVICE}")
print(f"Backbone: {BACKBONE}")
print(f"Number of classes: {NUM_CLASSES} -> {CLASS_NAMES}")

train_dataset = VineyardDataset(os.path.join(DATASET_ROOT, "train/images"), os.path.join(DATASET_ROOT, "train/masks"), target_size=IMAGE_SIZE)
valid_dataset = VineyardDataset(os.path.join(DATASET_ROOT, "valid/images"), os.path.join(DATASET_ROOT, "valid/masks"), target_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

model = UNetResNet(n_classes=NUM_CLASSES, backbone=BACKBONE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()  # AMP scaler

# =======================
# TRAINING LOOP WITH AMP
# =======================
best_val_loss = float('inf')
epochs_no_improve = 0
best_epoch = 0
last_epoch = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    last_epoch = epoch + 1

    # --- Training ---
    model.train()
    running_train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Training", unit="batch")

    for images, masks in train_pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with autocast():  # AMP forward
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_train_loss = running_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    running_val_loss = 0.0
    total_pixel_acc = 0
    total_miou = 0
    total_batches = 0
    per_class_ious = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        val_pbar = tqdm(valid_loader, desc=f"Validation", unit="batch")
        for images, masks in val_pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            running_val_loss += loss.item()
            pixel_acc, mean_iou, ious = compute_metrics(outputs, masks)
            total_pixel_acc += pixel_acc
            total_miou += mean_iou
            per_class_ious += np.nan_to_num(ious)
            total_batches += 1
            val_pbar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{mean_iou:.4f}")

    epoch_val_loss = running_val_loss / len(valid_loader)
    avg_pixel_acc = total_pixel_acc / total_batches
    avg_miou = total_miou / total_batches
    avg_ious = per_class_ious / total_batches

    print(f"Epoch Summary -> Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_val_loss:.4f}, "
          f"Pixel Acc: {avg_pixel_acc:.4f}, mIoU: {avg_miou:.4f}")

    # --- Save metrics ---
    with open(METRICS_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, avg_pixel_acc, avg_miou] + avg_ious.tolist())
    shutil.copy(METRICS_PATH, os.path.join(RESULTS_DIR, "metrics_last.csv")) # Updated to save inside run dir

    # Save latest model every epoch
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "latest_model.pth"))

    # --- Early Stopping ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, os.path.basename(MODEL_OUTPUT_PATH)))
        print(f"✨ New best model saved with validation loss: {best_val_loss:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")


    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"🛑 Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
        break

# =======================
# SAVE FINAL TRAINING SUMMARY TO YAML
# =======================
training_config.update({
    "last_trained_epoch": last_epoch,
    "best_model_epoch": best_epoch,
    "best_val_loss": float(best_val_loss)
})

with open(TRAINING_CONFIG_PATH, "w") as f:
    yaml.dump(training_config, f, default_flow_style=False)

print("\nTraining complete.")
print(f"✅ Best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
print(f"📊 Metrics logged to: {METRICS_PATH}")
print(f"📝 Updated training config with epoch info at: {TRAINING_CONFIG_PATH}")