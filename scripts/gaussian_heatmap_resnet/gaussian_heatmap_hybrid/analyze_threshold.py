import torch
import numpy as np
import cv2
import os
from PIL import Image
from skimage.feature import peak_local_max
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# ================= CONFIG =================
# Use your best model path
MODEL_PATH = "results_hybrid/hybrid_train_resnet101_20260211_120756/best_hybrid_model.pth"
# Pick an image that has BOTH bright poles and faint/shadowy poles
TEST_IMAGE = "../../../images/riseholme/august_2024/65_feet/DJI_20240802140842_0076_W.JPG"
BACKBONE = "resnet101"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (1280, 960)
# ==========================================

# (Insert your HybridUNetResNet class definition here - same as before)
class HybridUNetResNet(nn.Module):
    # ... paste the exact class from your inference script ...
    def __init__(self, backbone="resnet101"):
        super().__init__()
        if backbone == "resnet101":
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
        self.head_reg = nn.Conv2d(64, 2, 1)
        self.head_seg = nn.Conv2d(64, 1, 1)

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
        out_seg = self.head_seg(d1)
        out_seg = F.interpolate(out_seg, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out_reg, out_seg

# Load Model
model = HybridUNetResNet(backbone=BACKBONE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Run Inference
img = Image.open(TEST_IMAGE).convert("RGB")
w, h = img.size
input_tensor = torch.from_numpy(np.array(img.resize(IMAGE_SIZE)) / 255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

with torch.no_grad():
    pred_reg, _ = model(input_tensor)
    # Get Pole Heatmap (Channel 0)
    heatmap = pred_reg[0, 0].cpu().numpy()

# Resize back to original
heatmap_full = cv2.resize(heatmap, (w, h))

# Find Peaks with a very low threshold to see EVERYTHING
low_threshold = 0.1
peaks = peak_local_max(heatmap_full, min_distance=20, threshold_abs=low_threshold)

print(f"--- Analysis of {os.path.basename(TEST_IMAGE)} ---")
print(f"Found {len(peaks)} potential peaks > {low_threshold}")

# Print the confidence of each peak
confidences = []
for y, x in peaks:
    conf = heatmap_full[y, x]
    confidences.append(conf)

confidences = sorted(confidences, reverse=True)

print("\nTop 10 Brightest Poles (Confidence):")
print(np.round(confidences[:10], 3))

print("\nFaintest 10 Poles (Confidence):")
print(np.round(confidences[-10:], 3))

print("\nStats:")
print(f"Mean: {np.mean(confidences):.3f}")
print(f"Median: {np.median(confidences):.3f}")