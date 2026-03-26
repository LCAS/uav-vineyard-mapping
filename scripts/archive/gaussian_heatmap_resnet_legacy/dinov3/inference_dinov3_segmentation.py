import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
from dinov3.models.vision_transformer import vit_large
import shutil

# ======================
# CONFIGURATION
# ======================
# MODEL_PATH = "./models/vineyard_segmentation_16/both_yolo_and_gps_labels/dinov3_vitb14_vineyard_4_class_image_size_640x480_batch_size_2_epochs_100_100.pth"
# INPUT_DIR = "../heatmap_masks_from_yolo_labels_and_gps_labels/vineyard_segmentation_16/test/images/"
# OUTPUT_DIR = "inference_dinov3_vineyard_heatmaps/"

MODEL_PATH = "./results/dinov3_vitb14_4_class_image_size_640x480_batch_size_2_best.pth"
INPUT_DIR = "../heatmap_masks_from_yolo_labels/vineyard_segmentation_17/test/images/"
OUTPUT_DIR = "inference_dinov3_vineyard_heatmaps/vineyard_segmentation_17/yolo_labels/"

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
# MODEL DEFINITION
# ======================
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
        patch_h, patch_w = self.backbone.patch_embed.patch_size
        h = x.shape[2] // patch_h
        w = x.shape[3] // patch_w

        # Remove class token if present
        if features.shape[1] == (h * w + 1):
            features = features[:, 1:, :]

        features = features.permute(0, 2, 1).reshape(B, C, h, w)

        out = self.decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return out


# ======================
# LOAD MODEL
# ======================
model = DINOv3Segmentation(n_classes=NUM_CLASSES).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()


# ======================
# NORMALIZATION FUNCTION
# ======================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # DINOv3 normalization
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image


# ======================
# HEATMAP GENERATION
# ======================
def generate_single_class_heatmap(image_path, output_path, model_outputs, class_idx, threshold=0.3, alpha=0.6):
    image_pil = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image_np = np.array(image_pil)
    
    class_prob_map = model_outputs[class_idx, :, :]
    mask = class_prob_map >= threshold
    if not np.any(mask):
        return  # skip if no pixels exceed threshold

    normalized_prob = np.zeros_like(class_prob_map)
    detected_probs = class_prob_map[mask]
    if detected_probs.max() > detected_probs.min():
        normalized_prob[mask] = (detected_probs - detected_probs.min()) / (detected_probs.max() - detected_probs.min())
    else:
        normalized_prob[mask] = 1.0

    heatmap_gray = (normalized_prob * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    blended[~mask] = image_np[~mask]
    Image.fromarray(blended).save(output_path)


# ======================
# INFERENCE LOOP
# ======================
print(f"\n🚀 Starting inference on images from '{INPUT_DIR}'...")
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
