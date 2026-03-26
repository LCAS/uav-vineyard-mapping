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
BACKBONE = "resnext50"  # options: "resnet18", "resnet50", "resnext50"
MODEL_PATH = "results_resnet/gps_masks/train_resnet50_20251027_114455/resnet50_unet_image_size_1280x960_batch_size_8.pth"
# MODEL_PATH = "results_resnet/gps_masks/train_resnet18_20251029_092840/resnet18_unet_image_size_640x480_batch_size_8.pth"
INPUT_DIR = "dataset_gps_masks/riseholme/full_res/test/images/"
OUTPUT_DIR = "resnet_inference/heatmaps/riseholme/train_resnet18_20251029_092840/inference_image_size_1080x960/full_res/"

IMAGE_SIZE = (1280, 960) # (width, height) full res images are 3040x4056
# IMAGE_SIZE = (4056, 3040)
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
# MODEL DEFINITION
# ======================
class UNetResNet(nn.Module):
    def __init__(self, n_classes, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            resnet = models.resnet18(weights=None)
            enc_ch = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        elif backbone == "resnext50":
            resnet = models.resnext50_32x4d(weights=None)
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

        # ------------------
        # V V V THIS IS THE CORRECTED BLOCK V V V
        # ------------------
        # Decoder: in_channels = upconv_out_channels + skip_channels
        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)  # final decoder before output

        # Final classification layer
        self.final = nn.Conv2d(64, n_classes, 1)
        # ------------------
        # ^ ^ ^ END OF CORRECTION ^ ^ ^
        # ------------------

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

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# ======================
# LOAD MODEL
# ======================
model = UNetResNet(n_classes=NUM_CLASSES, backbone=BACKBONE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH} using backbone {BACKBONE}")

# ======================
# PREPROCESS FUNCTION
# ======================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image

# ======================
# HEATMAP FUNCTION
# ======================
def generate_class_heatmap(image_path, output_path, class_probs, class_idx, threshold=0.3, alpha=0.6):
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(blended).save(output_path)

# ======================
# INFERENCE LOOP
# ======================
print(f"\n🚀 Running inference on images in: {INPUT_DIR}")
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(INPUT_DIR, file)
    image_tensor = preprocess_image(img_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    base_name = os.path.splitext(file)[0]
    for idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):  # skip background
        out_path = os.path.join(OUTPUT_DIR, cls_name, f"{base_name}_heatmap_{cls_name}.jpg")
        generate_class_heatmap(img_path, out_path, probs, idx, threshold=0.3, alpha=0.6)

print(f"\n✅ All heatmaps generated successfully in '{OUTPUT_DIR}'!")
