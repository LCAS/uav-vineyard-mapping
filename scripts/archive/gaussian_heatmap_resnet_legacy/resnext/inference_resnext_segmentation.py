import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil

# ======================
# CONFIGURATION
# ======================
# MODEL_PATH = "./models/vineyard_segmentation_16/yolo_labels/resnext50_32x4d_unet_4_class_image_size_640x480_batch_size_2_epochs_100_100.pth"
# INPUT_DIR = "../heatmap_masks_from_yolo_labels/vineyard_segmentation_16/test/images/"
# OUTPUT_DIR = "inference_resnext50_unet_vineyard_heatmaps/yolo_labels/"

MODEL_PATH = "./results_resnext/train_20251021_192234/resnext50_32x4d_unet__4_class_image_size_640x480_batch_size_4_best.pth"
# INPUT_DIR = "../heatmap_masks_from_yolo_labels/vineyard_segmentation_16/test/images/"
# OUTPUT_DIR = "inference_resnext50_unet_vineyard_heatmaps/heatmaps/vineyard_segmentation_16_trained_on_vineyard_segmentation_17/yolo_and_labels/"

INPUT_DIR = "../../../images/riseholme/july_2025/39_feet/"
OUTPUT_DIR = "inference_resnext50_unet_vineyard_heatmaps/heatmaps/riseholme/trained_on_vineyard_segmentation_17/july_2025/39_feet/"

IMAGE_SIZE = (1280, 960) # (width, height)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 4
CLASS_NAMES = ["background", "pole", "trunk", "vine_row"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# COPY TRAINING CONFIG DYNAMICALLY
# ======================
model_dir = os.path.dirname(MODEL_PATH)
training_config_files = [f for f in os.listdir(model_dir) if f.endswith(".yaml")]

if training_config_files:
    training_config_src = os.path.join(model_dir, training_config_files[0])
    training_config_dst = os.path.join(OUTPUT_DIR, os.path.basename(training_config_src))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.copy(training_config_src, training_config_dst)
    print(f"📄 Copied training configuration to inference folder: {training_config_dst}")
else:
    print(f"⚠️ No training config YAML found in model directory: {model_dir}")

# ======================
# MODEL DEFINITION (same as training)
# ======================
class UNetResNeXt50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(weights=None)  # weights not needed for inference

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

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

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out


# ======================
# LOAD MODEL
# ======================
model = UNetResNeXt50(n_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")


# ======================
# PREPROCESS FUNCTION
# ======================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image


# ======================
# HEATMAP GENERATION
# ======================
def generate_single_class_heatmap(image_path, output_path, class_probs, class_idx, threshold=0.3, alpha=0.6):
    image_pil = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image_np = np.array(image_pil)

    prob_map = class_probs[class_idx, :, :]
    mask = prob_map >= threshold
    if not np.any(mask):
        return

    norm_map = np.zeros_like(prob_map)
    detected_probs = prob_map[mask]
    if detected_probs.max() > detected_probs.min():
        norm_map[mask] = (detected_probs - detected_probs.min()) / (detected_probs.max() - detected_probs.min())
    else:
        norm_map[mask] = 1.0

    heatmap_gray = (norm_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    blended[~mask] = image_np[~mask]

    Image.fromarray(blended).save(output_path)


# ======================
# INFERENCE LOOP
# ======================
print(f"\n🚀 Running inference on: {INPUT_DIR}")
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(INPUT_DIR, file)
    image_tensor = preprocess_image(img_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    base_name = os.path.splitext(file)[0]
    for idx, name in enumerate(CLASS_NAMES[1:], start=1):  # skip background
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_heatmap_{name}.jpg")
        generate_single_class_heatmap(img_path, out_path, probs, idx, threshold=0.3, alpha=0.6)

print(f"\n✅ All heatmaps generated successfully in '{OUTPUT_DIR}'!")
