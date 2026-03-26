import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import csv
import numpy as np
from tqdm import tqdm
from dinov3.models.vision_transformer import vit_large
from datetime import datetime
import yaml

# =======================
# CONFIGURATION
# =======================
DATASET_ROOT = "dataset_gps_masks/riseholme/patches_960x1280"
MODEL_OUTPUT_PATH = "dinov3_vitb14_4_class_image_size_640x480_batch_size_2_best.pth"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_DIR = os.path.join(RESULTS_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(RUN_DIR, exist_ok=True)

METRICS_PATH = os.path.join(RUN_DIR, "metrics.csv")
TRAINING_CONFIG_PATH = os.path.join(RUN_DIR, "training_config.yaml")

BATCH_SIZE = 2
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
IMAGE_SIZE = (640, 480)  # width, height
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 50

NUM_CLASSES = 4
CLASS_NAMES = ["background", "pole", "trunk", "vine_row"]

# =======================
# SAVE TRAINING PARAMETERS
# =======================
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
    "model_architecture": "DINOv3Segmentation (ViT-Large)",
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "amp_enabled": True,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(TRAINING_CONFIG_PATH, "w") as f:
    yaml.dump(training_config, f, default_flow_style=False)

print(f"📄 Training configuration saved to: {TRAINING_CONFIG_PATH}")

# =======================
# METRICS CSV HEADER
# =======================
with open(METRICS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

# =======================
# DATASET
# =======================
class VineyardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(640, 480)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size

        # List all image files
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Map class names to their folder and filename prefix (exclude background)
        self.class_info = {
            "pole": ("posts_masks", "posts_mask_"),
            "trunk": ("rows_masks", "rows_mask_"),
            "vine_row": ("vine_masks", "vine_mask_")
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB").resize(self.image_size)
        image = np.array(image) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Initialize mask with background = 0
        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)

        # Load available masks for each class (1..NUM_CLASSES-1)
        for cls_idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):
            folder_name, prefix = self.class_info[cls_name]
            cls_mask_dir = os.path.join(self.masks_dir, folder_name)
            cls_mask_file = f"{prefix}{img_name}"
            cls_mask_path = os.path.join(cls_mask_dir, cls_mask_file)

            if os.path.exists(cls_mask_path):
                cls_mask = Image.open(cls_mask_path).resize(self.image_size, Image.NEAREST).convert("L")
                cls_mask_np = np.array(cls_mask) > 127
                mask[cls_mask_np] = cls_idx
            # else: missing mask → leave as background (0)

        return image, torch.from_numpy(mask).long()

# =======================
# MODEL DEFINITION
# =======================
class DINOv3Segmentation(nn.Module):
    def __init__(self, n_classes=4, pretrained_checkpoint=None):
        super().__init__()
        self.backbone = vit_large(
            patch_size=16,
            num_register_tokens=0,
            drop_path_rate=0.1
        )

        if pretrained_checkpoint:
            print(f"Loading pretrained weights from: {pretrained_checkpoint}")
            state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained DINOv3 weights (missing={len(missing)}, unexpected={len(unexpected)})")

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, 1)
        )

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        B, N, C = features.shape
        h = w = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, C, h, w)
        out = self.decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return out

# =======================
# DATA LOADERS
# =======================
train_dataset = VineyardDataset(
    images_dir=os.path.join(DATASET_ROOT, "train/images"),
    masks_dir=os.path.join(DATASET_ROOT, "train/masks")
)
valid_dataset = VineyardDataset(
    images_dir=os.path.join(DATASET_ROOT, "valid/images"),
    masks_dir=os.path.join(DATASET_ROOT, "valid/masks")
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
print(f"✅ Loaded {len(train_dataset)} training and {len(valid_dataset)} validation images")

# =======================
# TRAINING SETUP
# =======================
model = DINOv3Segmentation(
    n_classes=NUM_CLASSES,
    pretrained_checkpoint="../models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
scaler = torch.cuda.amp.GradScaler()

# =======================
# TRAINING LOOP
# =======================
best_val_loss = float("inf")
epochs_no_improve = 0
best_epoch = 0
last_epoch = 0  # track the last completed epoch

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training", unit="batch"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    epoch_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(valid_loader, desc="Validation", unit="batch"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(valid_loader)
    print(f"Epoch Summary → Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Update last_epoch
    last_epoch = epoch + 1

    # === Save metrics ===
    with open(METRICS_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch + 1, epoch_train_loss, epoch_val_loss])

    # === Save latest model ===
    torch.save(model.state_dict(), os.path.join(RUN_DIR, "latest_model.pth"))

    # === Best model / Early stopping ===
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1  # update best_epoch here
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))
        print(f"✨ New best model saved (val loss: {best_val_loss:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Val loss did not improve ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print("🛑 Early stopping triggered.")
        break

# =======================
# SAVE FINAL METRICS TO YAML
# =======================
final_metrics = {
    "best_val_loss": float(best_val_loss),
    "best_epoch": best_epoch,
    "total_epochs_trained": last_epoch,
    "stopped_early": epochs_no_improve >= EARLY_STOPPING_PATIENCE,
    "metrics_csv": os.path.basename(METRICS_PATH),
    "best_model_file": "best_model.pth"
}

training_config.update(final_metrics)
with open(TRAINING_CONFIG_PATH, "w") as f:
    yaml.dump(training_config, f, default_flow_style=False)

print(f"\n✅ Training complete. Best model saved at '{os.path.join(RUN_DIR, 'best_model.pth')}' (val loss: {best_val_loss:.4f})")
print(f"📊 Metrics saved to: {METRICS_PATH}")
print(f"📄 Updated training configuration with metrics saved to: {TRAINING_CONFIG_PATH}")
