import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models
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
DATASET_ROOT = "../heatmap_masks_from_yolo_labels/vineyard_segmentation_17"
MODEL_OUTPUT_PATH = "resnext50_32x4d_unet_4_class_image_size_640x480_batch_size_4_best.pth"

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
IMAGE_SIZE = (640, 480) # (width, height)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 50

NUM_CLASSES = 4
CLASS_NAMES = ["background", "pole", "trunk", "vine_row"]

# --- Metrics logging setup ---
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_DIR = os.path.join("results_resnext", RUN_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.csv")

with open(METRICS_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "pixel_acc", "mIoU"] +
                    [f"IoU_{c}" for c in CLASS_NAMES])

# =======================
# SAVE TRAINING PARAMETERS
# =======================
TRAINING_CONFIG_PATH = os.path.join(RESULTS_DIR, "training_config.yaml")

training_config = {
    "dataset_root": DATASET_ROOT,
    "model_output_path": MODEL_OUTPUT_PATH,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "image_size": IMAGE_SIZE,
    "device": str(DEVICE),
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "model_architecture": "UNetResNet18",
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "amp_enabled": True,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(TRAINING_CONFIG_PATH, "w") as f:
    yaml.dump(training_config, f, default_flow_style=False)

print(f"📄 Training configuration saved to: {TRAINING_CONFIG_PATH}")

# =======================
# MODEL DEFINITION (U-Net with ResNeXt50 encoder)
# =======================
class UNetResNeXt50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec4 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec3 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        # Decoder with skip connections
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

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out


# =======================
# DATASET DEFINITION
# =======================
class VineyardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=IMAGE_SIZE):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.image_files = sorted([f for f in os.listdir(images_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB").resize(self.image_size)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        base_name = os.path.splitext(img_name)[0]
        for cls_idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):
            cls_mask_dir = os.path.join(self.masks_dir, cls_name)
            if not os.path.isdir(cls_mask_dir):
                continue
            mask_files = [f for f in os.listdir(cls_mask_dir) if base_name in f]
            if mask_files:
                cls_mask_path = os.path.join(cls_mask_dir, mask_files[0])
                cls_mask = Image.open(cls_mask_path).resize(self.image_size, Image.NEAREST).convert("L")
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
print(f"Number of classes: {NUM_CLASSES} -> {CLASS_NAMES}")

train_dataset = VineyardDataset(os.path.join(DATASET_ROOT, "train/images"),
                                os.path.join(DATASET_ROOT, "train/masks"))
valid_dataset = VineyardDataset(os.path.join(DATASET_ROOT, "valid/images"),
                                os.path.join(DATASET_ROOT, "valid/masks"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

model = UNetResNeXt50(n_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =======================
# TRAINING LOOP
# =======================
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

    # --- Training ---
    model.train()
    running_train_loss = 0.0
    train_pbar = tqdm(train_loader, desc="Training", unit="batch")
    for images, masks in train_pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
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
        val_pbar = tqdm(valid_loader, desc="Validation", unit="batch")
        for images, masks in val_pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
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

    print(f"Epoch Summary -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
          f"Pixel Acc: {avg_pixel_acc:.4f}, mIoU: {avg_miou:.4f}")

    # --- Save metrics ---
    with open(METRICS_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, avg_pixel_acc, avg_miou] + avg_ious.tolist())

    # --- Save latest model ---
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "latest_model.pth"))

    # --- Early Stopping ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
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
# FINAL MESSAGE
# =======================
print("\nTraining complete.")
print(f"Best model saved to '{RESULTS_DIR}' with validation loss: {best_val_loss:.4f}")
print(f"📊 Metrics logged to: {METRICS_PATH}")
